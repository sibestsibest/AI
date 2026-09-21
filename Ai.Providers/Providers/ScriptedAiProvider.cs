using Ai.Providers.Contracts;

namespace Ai.Providers.Providers;

/// <summary>
/// PROVIDER THEO KỊCH BẢN — không gọi mạng, kết quả tất định.
///
/// Cùng lý lẽ với <c>StaticSearchProvider</c> của Phase 12: bộ test và demo
/// phụ thuộc mạng là bộ test hỏng — hôm nay xanh, mai đỏ, mà chẳng ai sửa gì.
/// Lớp này cho phép kiểm TOÀN BỘ đường đi (định tuyến, thử lại, tạm loại
/// provider hỏng, thống kê, và khâu diễn đạt lại câu trả lời) mà không cần
/// một máy chủ nào.
///
/// Nó cũng là chỗ dựng các tình huống hỏng mà nhà cung cấp thật rất khó bắt ta
/// gặp đúng lúc cần: hết thời gian, vượt hạn mức, khoá sai, phản hồi rác.
/// </summary>
public sealed class ScriptedAiProvider : IAiProvider
{
    private readonly Func<AiRequest, int, AiResponse> _script;
    private int _calls;

    /// <param name="script">Nhận yêu cầu và số lần đã gọi (bắt đầu từ 1), trả về phản hồi.</param>
    public ScriptedAiProvider(string name, Func<AiRequest, int, AiResponse> script,
        IReadOnlyList<string>? models = null, bool available = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(script);

        ProviderName = name;
        _script = script;
        SupportedModels = models ?? ["scripted-model"];
        Available = available;
    }

    public string ProviderName { get; }

    public IReadOnlyList<string> SupportedModels { get; }

    /// <summary>Đổi được lúc chạy, để test cảnh provider biến mất giữa phiên.</summary>
    public bool Available { get; set; }

    public int Calls => _calls;

    /// <summary>Độ trễ giả, để test hạn thời gian và để demo nhìn ra được thứ tự.</summary>
    public TimeSpan Delay { get; init; }

    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(Available);

    public async Task<AiResponse> GenerateAsync(AiRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        int call = Interlocked.Increment(ref _calls);

        if (Delay > TimeSpan.Zero)
        {
            await Task.Delay(Delay, cancellationToken).ConfigureAwait(false);
        }

        cancellationToken.ThrowIfCancellationRequested();

        return _script(request, call);
    }

    // ------------------------------------------------------------------
    // Kịch bản dựng sẵn cho những tình huống hay phải kiểm
    // ------------------------------------------------------------------

    /// <summary>Luôn trả về đúng một câu — dùng khi chỉ cần một provider "chạy được".</summary>
    public static ScriptedAiProvider AlwaysReplies(string name, string reply) =>
        new(name, (_, _) => AiResponse.Ok(name, "scripted-model", reply, TimeSpan.FromMilliseconds(1)));

    /// <summary>Luôn thất bại với một loại lỗi cho trước.</summary>
    public static ScriptedAiProvider AlwaysFails(string name, AiErrorKind kind, string message = "lỗi dựng sẵn") =>
        new(name, (_, _) => AiResponse.Fail(name, "scripted-model", new AiError(kind, message)));

    /// <summary>Hỏng <paramref name="failures"/> lần đầu rồi mới chạy được — để kiểm cơ chế thử lại.</summary>
    public static ScriptedAiProvider FailsThenSucceeds(string name, int failures, string reply,
        AiErrorKind kind = AiErrorKind.Timeout) =>
        new(name, (_, call) => call <= failures
            ? AiResponse.Fail(name, "scripted-model", new AiError(kind, $"hỏng lần {call}"))
            : AiResponse.Ok(name, "scripted-model", reply, TimeSpan.FromMilliseconds(1)));

    /// <summary>Trả về đúng nội dung được yêu cầu diễn đạt — dùng để kiểm khâu kiểm chứng câu trả lời.</summary>
    public static ScriptedAiProvider Echoes(string name) =>
        new(name, (request, _) => AiResponse.Ok(name, "scripted-model", request.UserPrompt,
            TimeSpan.FromMilliseconds(1)));
}
