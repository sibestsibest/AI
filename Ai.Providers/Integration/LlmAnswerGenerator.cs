using System.Globalization;
using Ai.Phase16.Grounding;
using Ai.Phase16.Verification;
using Ai.Providers.Contracts;
using Ai.Providers.Routing;

namespace Ai.Providers.Integration;

/// <summary>Chuyện gì đã xảy ra ở lần diễn đạt gần nhất — để màn hình hiện ra được.</summary>
public sealed record LlmComposeDiagnostics
{
    public bool UsedLlm { get; init; }

    public string? Provider { get; init; }

    public string? Model { get; init; }

    public TimeSpan Latency { get; init; }

    public bool UsedFallbackProvider { get; init; }

    public AiUsage Usage { get; init; } = AiUsage.Unknown;

    /// <summary>Vì sao KHÔNG dùng văn bản của LLM (null nếu đã dùng).</summary>
    public string? RejectedReason { get; init; }

    public IReadOnlyList<RouteAttempt> Attempts { get; init; } = [];

    public override string ToString() => UsedLlm
        ? string.Format(CultureInfo.InvariantCulture, "đã dùng LLM {0}/{1}, {2:F0} ms, {3}",
            Provider, Model, Latency.TotalMilliseconds, Usage)
        : $"dùng văn bản khuôn mẫu — {RejectedReason}";
}

public sealed record LlmComposeOptions
{
    /// <summary>Số câu tối đa cho model viết. Ngắn thì ít chỗ để trôi khỏi căn cứ.</summary>
    public int MaxSentences { get; init; } = 4;

    public double Temperature { get; init; } = 0.1;

    public int MaxTokens { get; init; } = 400;

    /// <summary>
    /// Hạn thời gian TỔNG cho khâu diễn đạt.
    ///
    /// Đây là hạn của cả đường ống, không phải của một provider: đường ống trả
    /// lời của Phase 14 là đồng bộ, nên nếu khâu này treo thì màn hình treo.
    /// Hết hạn thì dùng văn bản khuôn mẫu — người dùng vẫn có câu trả lời.
    /// </summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(20);

    /// <summary>
    /// Văn bản của LLM có phải qua <see cref="AnswerValidator"/> mới được dùng.
    ///
    /// Mặc định BẬT, và tắt nó là một quyết định nghiêm trọng: lúc đó văn bản
    /// do model sinh ra được phát trực tiếp, và ràng buộc "mọi câu truy được về
    /// căn cứ" mất hiệu lực. Công tắc này tồn tại để TEST quan sát được hành vi
    /// khi không có chốt, chứ không phải để dùng thật.
    /// </summary>
    public bool RequireValidation { get; init; } = true;

    public string SystemPrompt { get; init; } = EvidencePrompt.DefaultSystemPrompt;
}

/// <summary>
/// BỘ SINH CÂU TRẢ LỜI DÙNG LLM — thay đúng một khâu: DIỄN ĐẠT.
///
/// ============================================================
/// ĐÂY LÀ TOÀN BỘ CHỖ NỐI GIỮA LLM NGOÀI VÀ HỆ THỐNG
/// ============================================================
///
/// Nó hiện thực <see cref="IAnswerGenerator"/> của Phase 14, nên đường ống trả
/// lời không đổi một dòng nào:
///
///     Câu hỏi -> Evidence (P13) -> Reason -> Verify (P14)
///                                              ↓
///                                     [ LlmAnswerGenerator ]
///                                              ↓
///                                     AnswerValidator (P14)
///                                        ↓ đạt      ↓ không đạt
///                                    văn bản LLM   văn bản khuôn mẫu
///
/// BỐN GIỚI HẠN CỨNG, và chúng là lý do cách nối này an toàn:
///
///   1. LLM KHÔNG quyết định nội dung. <see cref="ExtractClaims"/> giao nguyên
///      cho bộ sinh khuôn mẫu — khẳng định vẫn được rút TẤT ĐỊNH từ căn cứ.
///      Nếu để LLM rút khẳng định thì mọi thứ phía sau (độ tự tin, fact vs
///      inference, id căn cứ) đều dựa trên thứ nó tự nghĩ ra.
///
///   2. LLM KHÔNG quyết định có trả lời hay không. Kết luận
///      <see cref="AnswerVerdict.Insufficient"/> và
///      <see cref="AnswerVerdict.Conflicted"/> KHÔNG được đưa cho model diễn
///      đạt: đó là hai câu trả lời mà cách diễn đạt quan trọng hơn cả nội dung
///      ("tôi không biết", "hai nguồn nói ngược nhau"), và một model viết mượt
///      rất dễ biến chúng thành câu nghe như đã có kết luận.
///
///   3. LLM KHÔNG gắn nguồn. Trích dẫn do hệ thống tự thêm
///      (<see cref="EvidencePrompt.BuildCitations"/>). Trích dẫn sai còn tệ hơn
///      không trích dẫn.
///
///   4. Văn bản của LLM phải QUA ĐƯỢC <see cref="AnswerValidator"/> mới được
///      dùng. Không qua thì dùng văn bản khuôn mẫu — tức là tệ nhất cũng chỉ
///      quay về đúng hành vi trước khi có lớp này. Không có đường nào để câu
///      trả lời trở nên tệ hơn trước.
///
/// Điểm 4 chính là điều mà chú thích của <see cref="AnswerValidator"/> đã nói
/// trước khi lớp này tồn tại: "bộ sinh sẽ được sửa, sẽ được thay… lúc đó tính
/// chất kia biến mất lặng lẽ, không có gì báo". Validator được viết độc lập với
/// bộ sinh chính vì ngày này.
/// </summary>
public sealed class LlmAnswerGenerator : IAnswerGenerator
{
    private readonly IAiRouter _router;
    private readonly IAnswerGenerator _template;
    private readonly IAnswerValidator _validator;
    private readonly LlmComposeOptions _options;

    public LlmAnswerGenerator(
        IAiRouter router,
        IAnswerGenerator? template = null,
        IAnswerValidator? validator = null,
        LlmComposeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(router);

        _router = router;
        _template = template ?? new AnswerGenerator();
        _validator = validator ?? new AnswerValidator();
        _options = options ?? new LlmComposeOptions();
    }

    /// <summary>Trạng thái lần diễn đạt gần nhất. Không có LLM nào được gọi thì UsedLlm = false.</summary>
    public LlmComposeDiagnostics Last { get; private set; } = new() { RejectedReason = "chưa gọi lần nào" };

    /// <summary>Số lần văn bản của LLM bị loại vì không qua kiểm — con số đáng theo dõi.</summary>
    public int RejectedCount { get; private set; }

    public int UsedCount { get; private set; }

    /// <summary>
    /// Rút khẳng định: GIAO NGUYÊN cho bộ sinh khuôn mẫu, không gọi LLM.
    ///
    /// Đây là giới hạn cứng số 1. Khẳng định là thứ mà độ tự tin, nhãn sự-thật-
    /// hay-suy-ra, và toàn bộ việc truy nguồn dựa vào; để một model sinh chữ
    /// quyết định chúng thì không còn gì để kiểm chứng nữa.
    /// </summary>
    public IReadOnlyList<Claim> ExtractClaims(EvidenceSet evidence, IReadOnlyList<Evidence> cited) =>
        _template.ExtractClaims(evidence, cited);

    public string Compose(string question, AnswerVerdict verdict, double confidence,
        IReadOnlyList<Claim> claims, IReadOnlyList<Evidence> cited,
        IReadOnlyList<Contradiction> contradictions, bool requiresDisclosure)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(claims);
        ArgumentNullException.ThrowIfNull(cited);

        string templateText = _template.Compose(question, verdict, confidence, claims, cited,
            contradictions, requiresDisclosure);

        // --- Giới hạn 2: không đưa "không biết" và "mâu thuẫn" cho model viết ---
        if (verdict is AnswerVerdict.Insufficient or AnswerVerdict.Conflicted)
        {
            Last = new LlmComposeDiagnostics
            {
                RejectedReason = $"kết luận {verdict} không được đưa cho LLM diễn đạt (luật cứng)",
            };

            return templateText;
        }

        if (cited.Count == 0 || claims.Count == 0)
        {
            Last = new LlmComposeDiagnostics { RejectedReason = "không có căn cứ nào để diễn đạt" };
            return templateText;
        }

        var routed = CallModel(question, claims, cited);

        if (routed is null)
        {
            Last = new LlmComposeDiagnostics
            {
                RejectedReason = $"khâu diễn đạt hết hạn {_options.Timeout.TotalSeconds:F0}s",
            };

            RejectedCount++;
            return templateText;
        }

        if (!routed.Response.Success)
        {
            Last = new LlmComposeDiagnostics
            {
                RejectedReason = $"không provider nào trả lời được: {routed.Response.Error}",
                Attempts = routed.Attempts,
                UsedFallbackProvider = routed.UsedFallback,
            };

            RejectedCount++;
            return templateText;
        }

        string candidate = Clean(routed.Response.Content);

        // Model làm đúng việc được giao khi căn cứ yếu: nói thẳng là không đủ.
        // Lúc đó dùng văn bản khuôn mẫu, vì nó nói điều đó chuẩn hơn.
        if (candidate.Length == 0
            || candidate.Contains("Không đủ căn cứ", StringComparison.OrdinalIgnoreCase))
        {
            Last = Diagnostics(routed, "model nói không đủ căn cứ");
            RejectedCount++;
            return templateText;
        }

        string answer = candidate + EvidencePrompt.BuildCitations(cited, requiresDisclosure);

        // --- Giới hạn 4: chốt chặn của Phase 14 ---
        if (_options.RequireValidation)
        {
            var scope = cited
                .Concat(contradictions.SelectMany(c => new[] { c.Left, c.Right }))
                .DistinctBy(e => e.Id)
                .ToList();

            var report = _validator.Validate(answer, claims, scope);

            if (!report.IsValid)
            {
                Last = Diagnostics(routed,
                    $"văn bản LLM KHÔNG qua kiểm chứng: {string.Join("; ", report.Problems)}");

                RejectedCount++;
                return templateText;
            }
        }

        Last = Diagnostics(routed, null);
        UsedCount++;

        return answer;
    }

    /// <summary>
    /// Gọi model qua bộ định tuyến, từ một hàm ĐỒNG BỘ.
    ///
    /// <see cref="IAnswerGenerator.Compose"/> của Phase 14 là đồng bộ, và ta
    /// KHÔNG sửa Phase 14 để nó thành async — đổi chữ ký đó sẽ lan ra cả đường
    /// ống, cả agent, cả bộ test của bốn phase.
    ///
    /// Nên chỗ này chặn luồng, và làm điều đó theo cách ít tệ nhất:
    ///
    ///   • <c>Task.Run</c> để lời gọi chạy ngoài ngữ cảnh đồng bộ hiện tại —
    ///     chờ trực tiếp một task async trên ngữ cảnh có SynchronizationContext
    ///     là công thức gây treo cứng (deadlock).
    ///   • <c>Wait(timeout)</c> chứ không chờ vô hạn: provider treo thì khâu
    ///     diễn đạt bỏ qua, không kéo cả màn hình xuống theo.
    ///   • huỷ token khi hết hạn để lời gọi HTTP không sống tiếp vô ích.
    ///
    /// Trả về null nghĩa là hết hạn.
    /// </summary>
    private RoutedResponse? CallModel(string question, IReadOnlyList<Claim> claims, IReadOnlyList<Evidence> cited)
    {
        var request = new AiRequest
        {
            SystemPrompt = _options.SystemPrompt,
            UserPrompt = EvidencePrompt.BuildUserPrompt(question, claims, cited, _options.MaxSentences),
            Temperature = _options.Temperature,
            MaxTokens = _options.MaxTokens,
            TaskType = AiTaskType.Summarization,
        };

        using var cancellation = new CancellationTokenSource(_options.Timeout);

        var task = Task.Run(() => _router.GenerateAsync(request, cancellation.Token), cancellation.Token);

        try
        {
            return task.Wait(_options.Timeout) ? task.Result : null;
        }
        catch (Exception ex) when (ex is AggregateException or OperationCanceledException)
        {
            return null;
        }
    }

    private LlmComposeDiagnostics Diagnostics(RoutedResponse routed, string? rejectedReason) => new()
    {
        UsedLlm = rejectedReason is null,
        Provider = routed.Response.Provider,
        Model = routed.Response.Model,
        Latency = routed.Response.Latency,
        Usage = routed.Response.Usage,
        UsedFallbackProvider = routed.UsedFallback,
        Attempts = routed.Attempts,
        RejectedReason = rejectedReason,
    };

    /// <summary>
    /// Dọn văn bản model trả về.
    ///
    /// Model hay thêm lời mở đầu ("Chắc chắn rồi! Đây là…") dù prompt đã cấm,
    /// và hay bọc câu trả lời trong dấu nháy hoặc khối mã. Dọn ở đây thay vì
    /// hy vọng prompt đủ nghiêm — prompt là lời đề nghị, code mới là luật.
    /// </summary>
    private static string Clean(string content)
    {
        string text = content.Trim();

        if (text.StartsWith("```", StringComparison.Ordinal))
        {
            int firstBreak = text.IndexOf('\n');
            if (firstBreak > 0) text = text[(firstBreak + 1)..];

            text = text.TrimEnd('`', '\n', '\r', ' ');
        }

        return text.Trim().Trim('"').Trim();
    }
}
