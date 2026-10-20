using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.Resilience;
using Ai.Providers.Security;
using Ai.Providers.Usage;

namespace Ai.Providers.Routing;

/// <summary>Một lần thử một provider, kèm kết quả — để báo cáo và ghi log.</summary>
public sealed record RouteAttempt(string Provider, bool Success, AiErrorKind ErrorKind, string? Detail, TimeSpan Latency)
{
    public override string ToString() =>
        Success ? $"{Provider}: OK ({Latency.TotalMilliseconds:F0} ms)" : $"{Provider}: {ErrorKind} — {Detail}";
}

/// <summary>Câu trả lời cuối cùng, kèm toàn bộ đường đi đã thử.</summary>
public sealed record RoutedResponse(AiResponse Response, IReadOnlyList<RouteAttempt> Attempts)
{
    public bool UsedFallback => Attempts.Count > 1;

    public override string ToString() =>
        Attempts.Count <= 1 ? Response.ToString() : $"{Response} (sau {Attempts.Count} lần thử)";
}

public interface IAiRouter
{
    Task<RoutedResponse> GenerateAsync(AiRequest request, CancellationToken cancellationToken = default);

    /// <summary>Có provider nào dùng được lúc này không. Dùng cho màn hình bật/tắt LLM.</summary>
    Task<bool> HasAvailableProviderAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// BỘ ĐỊNH TUYẾN — chọn provider, thử lại, chuyển provider khi hỏng.
///
/// THỨ TỰ CHỌN, và mỗi bước đều bỏ bớt ứng viên:
///
///   1. đang bật trong cấu hình
///   2. nhận loại việc này (PreferredFor)
///   3. cho phép model được yêu cầu (danh sách model)
///   4. không đang trong khoảng nghỉ vì hỏng liên tiếp (health tracker)
///   5. xếp: DefaultProvider trước → local trước (nếu chiến lược LocalFirst)
///      → còn lại theo thứ tự cấu hình
///
/// Rồi thử lần lượt: mỗi provider có <c>RetryPolicy</c> riêng (lỗi tạm thời
/// thì thử lại tại chỗ), hỏng hẳn thì sang provider sau. Hết ứng viên thì trả
/// về LỖI CÓ KIỂM SOÁT — <see cref="AiErrorKind.NoProviderAvailable"/> — chứ
/// không ném ngoại lệ và cũng không trả về chuỗi rỗng trông như câu trả lời.
///
/// ĐIỂM QUAN TRỌNG NHẤT: người gọi KHÔNG BAO GIỜ thấy tên provider trong luồng
/// điều khiển. Không có <c>if (provider == "ollama")</c> ở bất cứ đâu ngoài lớp
/// provider tương ứng. Tất cả trả về cùng một <see cref="AiResponse"/>.
///
/// Và bộ định tuyến KHÔNG kiểm nội dung câu trả lời. Việc đó của tầng trên
/// (<c>LlmAnswerGenerator</c> + <c>AnswerValidator</c> của Phase 14) — trộn hai
/// việc vào đây sẽ làm lớp này phải biết về căn cứ, về Phase 13, về mọi thứ.
/// </summary>
public sealed class AiProviderRouter : IAiRouter
{
    private readonly AiProvidersOptions _options;
    private readonly IReadOnlyList<IAiProvider> _providers;
    private readonly ProviderHealthTracker _health;
    private readonly AiUsageTracker _usage;

    public AiProviderRouter(
        AiProvidersOptions options,
        IReadOnlyList<IAiProvider> providers,
        ProviderHealthTracker? health = null,
        AiUsageTracker? usage = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(providers);

        _options = options;
        _providers = providers;
        _health = health ?? new ProviderHealthTracker();
        _usage = usage ?? new AiUsageTracker();
    }

    public ProviderHealthTracker Health => _health;

    public AiUsageTracker Usage => _usage;

    public IReadOnlyList<IAiProvider> Providers => _providers;

    public async Task<bool> HasAvailableProviderAsync(CancellationToken cancellationToken = default)
    {
        foreach (var (provider, _) in Candidates(AiTaskType.GeneralChat, model: null))
        {
            if (await provider.IsAvailableAsync(cancellationToken).ConfigureAwait(false)) return true;
        }

        return false;
    }

    public async Task<RoutedResponse> GenerateAsync(AiRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        var attempts = new List<RouteAttempt>();

        // --- Chặn TRƯỚC khi gọi mạng: yêu cầu quá lớn ---
        //
        // Kiểm ở đây chứ không ở từng provider, vì đây là luật của ứng dụng
        // (đừng đẩy cả tài liệu ra ngoài), không phải giới hạn của nhà cung cấp.
        if (request.CharacterCount > _options.MaxPromptCharacters)
        {
            var rejected = AiResponse.Fail("router", request.Model, new AiError(AiErrorKind.Rejected,
                $"yêu cầu dài {request.CharacterCount} ký tự, vượt hạn mức {_options.MaxPromptCharacters}"));

            return new RoutedResponse(rejected,
                [new RouteAttempt("router", false, AiErrorKind.Rejected, "vượt hạn mức độ dài", TimeSpan.Zero)]);
        }

        // --- Lọc bí mật TRƯỚC khi gửi ra ngoài ---
        var safeRequest = Sanitize(request);

        var candidates = Candidates(request.TaskType, request.Model).ToList();

        if (candidates.Count == 0)
        {
            return NoProvider(request, attempts, "không có provider nào phù hợp và đang sẵn sàng");
        }

        foreach (var (provider, options) in candidates)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var policy = new RetryPolicy(options.MaxRetries + 1);

            var response = await policy
                .ExecuteAsync(token => provider.GenerateAsync(safeRequest, token), cancellationToken)
                .ConfigureAwait(false);

            _usage.Record(response, request.TaskType);

            if (response.Success)
            {
                _health.RecordSuccess(provider.ProviderName, response.Latency);
                attempts.Add(new RouteAttempt(provider.ProviderName, true, AiErrorKind.None, null, response.Latency));

                return new RoutedResponse(TrimIfNeeded(response), attempts);
            }

            var error = response.Error ?? new AiError(AiErrorKind.ProviderError, "không rõ nguyên nhân");

            _health.RecordFailure(provider.ProviderName, error, response.Latency);
            attempts.Add(new RouteAttempt(provider.ProviderName, false, error.Kind, error.Message, response.Latency));
        }

        return NoProvider(request, attempts, "mọi provider đã thử đều thất bại");
    }

    /// <summary>
    /// Danh sách ứng viên đã lọc và đã xếp thứ tự.
    ///
    /// Tách thành hàm riêng vì đây là phần đáng đọc nhất của lớp — và là phần
    /// duy nhất quyết định "vì sao provider này được chọn".
    /// </summary>
    private IEnumerable<(IAiProvider Provider, ProviderOptions Options)> Candidates(AiTaskType task, string? model)
    {
        var pairs = _providers
            .Select(p => (Provider: p, Options: _options.Find(p.ProviderName)))
            .Where(x => x.Options is not null)
            .Select(x => (x.Provider, Options: x.Options!))
            .Where(x => x.Options.Enabled)
            .Where(x => x.Options.Handles(task))
            .Where(x => x.Options.AllowsModel(model))
            .Where(x => _health.ShouldTry(x.Provider.ProviderName));

        if (_options.Strategy == ProviderStrategy.LocalOnly)
        {
            pairs = pairs.Where(x => x.Options.IsLocal);
        }

        return pairs
            .OrderByDescending(x => IsDefault(x.Options))
            .ThenByDescending(x => _options.Strategy == ProviderStrategy.LocalFirst && x.Options.IsLocal)
            .ThenByDescending(x => x.Options.PreferredFor.Contains(task))
            .ThenBy(x => IndexOf(x.Options));
    }

    private bool IsDefault(ProviderOptions options) =>
        _options.DefaultProvider is { } name
        && options.Name.Equals(name, StringComparison.OrdinalIgnoreCase);

    private int IndexOf(ProviderOptions options)
    {
        for (int i = 0; i < _options.Providers.Count; i++)
        {
            if (ReferenceEquals(_options.Providers[i], options)) return i;
        }

        return int.MaxValue;
    }

    /// <summary>Lọc bí mật khỏi mọi phần văn bản sẽ rời khỏi máy.</summary>
    private static AiRequest Sanitize(AiRequest request) => request with
    {
        UserPrompt = OutboundRedactor.Redact(request.UserPrompt),
        SystemPrompt = request.SystemPrompt is null ? null : OutboundRedactor.Redact(request.SystemPrompt),
        ConversationHistory = [.. request.ConversationHistory
            .Select(m => m with { Content = OutboundRedactor.Redact(m.Content) })],
    };

    /// <summary>Cắt câu trả lời quá dài. Provider hỏng có thể trả về hàng megabyte.</summary>
    private AiResponse TrimIfNeeded(AiResponse response) =>
        response.Content.Length <= _options.MaxResponseCharacters
            ? response
            : response with { Content = response.Content[.._options.MaxResponseCharacters] };

    private static RoutedResponse NoProvider(AiRequest request, List<RouteAttempt> attempts, string reason) =>
        new(AiResponse.Fail("router", request.Model, new AiError(AiErrorKind.NoProviderAvailable, reason)), attempts);
}
