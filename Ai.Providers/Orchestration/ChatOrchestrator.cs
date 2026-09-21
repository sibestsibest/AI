using Ai.Providers.Configuration;
using Ai.Providers.Context;
using Ai.Providers.Contracts;
using Ai.Providers.Routing;
using Ai.Providers.Tools;

namespace Ai.Providers.Orchestration;

/// <summary>
/// MỘT YÊU CẦU CHAT — đúng khuôn của đề bài (conversationId / message / mode).
/// </summary>
public sealed record ChatRequest
{
    public required string Message { get; init; }

    public string ConversationId { get; init; } = "";

    public AiMode Mode { get; init; } = AiMode.Auto;

    /// <summary>Chỉ thị hệ thống thay thế. Null = dùng chỉ thị mặc định.</summary>
    public string? SystemInstructions { get; init; }
}

/// <summary>Một lượt gọi công cụ, dạng gọn để trả ra ngoài.</summary>
public sealed record ChatToolCall(string Name, bool Succeeded, bool Refused, double Milliseconds);

/// <summary>
/// CÂU TRẢ LỜI — đúng khuôn của đề bài, và KHÔNG có gì hơn.
///
/// Cố ý KHÔNG mang theo: chuỗi suy luận nội bộ của model, nội dung prompt đã
/// dựng, nội dung mẩu nhớ đã dùng. Đó đều là thứ hữu ích để dò lỗi và đều là
/// thứ không được đi ra ngoài cùng câu trả lời. Muốn xem thì có
/// <see cref="Diagnostics"/>, và nó nằm ở một trường riêng để không ai vô tình
/// serialize cả cụm ra ngoài.
/// </summary>
public sealed record ChatResult
{
    public required string ConversationId { get; init; }

    public required string Message { get; init; }

    public string? Model { get; init; }

    public string? Provider { get; init; }

    public IReadOnlyList<ChatToolCall> ToolCalls { get; init; } = [];

    public AiUsage Usage { get; init; } = AiUsage.Unknown;

    public bool Success { get; init; } = true;

    public AiError? Error { get; init; }

    /// <summary>Chi tiết chỉ dùng để dò lỗi và ghi log — KHÔNG gửi ra ngoài cho người dùng cuối.</summary>
    public ChatDiagnostics Diagnostics { get; init; } = new();
}

/// <summary>
/// Siêu dữ liệu có cấu trúc cho log — đúng danh sách đề bài yêu cầu ở mục
/// quan sát được, và không có nội dung nhạy cảm nào.
///
/// KHÔNG chứa: khoá, prompt, nội dung nhớ, nội dung kết quả công cụ. Chỉ chứa
/// con số và tên. Đây là khác biệt giữa một dòng log dùng được và một dòng log
/// vừa rò dữ liệu người dùng ra hệ thống thu thập log.
/// </summary>
public sealed record ChatDiagnostics
{
    public string RequestId { get; init; } = "";

    public AiMode Mode { get; init; }

    public AiTaskType TaskType { get; init; }

    public TimeSpan Latency { get; init; }

    public int ToolCount { get; init; }

    public int RetryCount { get; init; }

    public bool UsedFallbackProvider { get; init; }

    public bool HitToolLimit { get; init; }

    public int ContextCharacters { get; init; }

    public IReadOnlyList<string> ContextOmitted { get; init; } = [];

    public IReadOnlyList<string> ValidationProblems { get; init; } = [];

    public IReadOnlyList<RouteAttempt> Attempts { get; init; } = [];

    public override string ToString() =>
        $"req={RequestId} mode={Mode} task={TaskType} {Latency.TotalMilliseconds:F0}ms " +
        $"tools={ToolCount} retries={RetryCount}" +
        (UsedFallbackProvider ? " fallback" : "") +
        (ValidationProblems.Count > 0 ? $" problems={ValidationProblems.Count}" : "");
}

public sealed record OrchestratorOptions
{
    /// <summary>Số lần viết lại khi câu trả lời không qua kiểm. Đề bài: tối đa 2.</summary>
    public int MaxValidationRetries { get; init; } = 2;

    public string SystemInstructions { get; init; } =
        "Bạn là lớp diễn đạt của MiniAI. Trả lời ngắn gọn, đúng trọng tâm, bằng ngôn ngữ của câu hỏi. " +
        "Khi đã có kết quả công cụ, PHẢI dùng đúng con số trong đó — không tự tính lại, không làm tròn khác. " +
        "Không biết thì nói thẳng là không biết.";

    /// <summary>Hạn thời gian cho cả một lượt hỏi, kể cả công cụ và các lần viết lại.</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(90);
}

/// <summary>
/// BỘ ĐIỀU PHỐI — nối đủ các bước của đường ống đề bài, theo đúng thứ tự:
///
///     Câu hỏi
///        ↓
///     Phân loại việc        TaskClassifier (luật, không gọi model)
///        ↓
///     Chính sách chế độ     Modes.Policy
///        ↓
///     Dựng ngữ cảnh         ContextBuilder (nhớ + hội thoại, ĐÃ chọn lọc)
///        ↓
///     Chọn model            AiModelMap + AiProviderRouter
///        ↓
///     Vòng lặp công cụ      ToolLoop ↔ ToolRegistry (4 chốt)
///        ↓
///     Kiểm câu trả lời      ResponseValidator (tối đa 2 lần viết lại)
///        ↓
///     ChatResult
///
/// ĐÂY KHÔNG PHẢI MỘT KIẾN TRÚC AI THỨ HAI. Nó không có bộ nhớ riêng, không có
/// kho kiến thức riêng, không có bộ suy luận riêng — nó MƯỢN cả ba từ Phase
/// 10/13/9 qua <see cref="ContextBuilder"/> và <see cref="ToolRegistry"/>.
/// Việc duy nhất nó làm mà chưa ai làm là chạy vòng lặp model ↔ công cụ, vì
/// <see cref="IAiProvider"/> chỉ gọi được một lượt.
///
/// Đường trả lời CÓ KIỂM CHỨNG (căn cứ 5 kênh -> mâu thuẫn -> truy nguồn) vẫn
/// là của <c>AnswerPipeline</c> Phase 14, và vẫn được gọi qua
/// <c>LlmAnswerGenerator</c>. Hai đường tồn tại song song vì chúng trả lời hai
/// câu hỏi khác nhau: "làm hộ tôi việc này" và "điều này có đúng không".
/// </summary>
public sealed class ChatOrchestrator
{
    private readonly IAiRouter _router;
    private readonly IToolRegistry _registry;
    private readonly IContextBuilder _context;
    private readonly ResponseValidator _validator;
    private readonly AiModelMap _models;
    private readonly OrchestratorOptions _options;

    public ChatOrchestrator(
        IAiRouter router,
        IToolRegistry? registry = null,
        IContextBuilder? context = null,
        ResponseValidator? validator = null,
        AiModelMap? models = null,
        OrchestratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(router);

        _router = router;
        _registry = registry ?? new ToolRegistry();
        _context = context ?? new ContextBuilder();
        _validator = validator ?? ResponseValidator.Default();
        _models = models ?? new AiModelMap();
        _options = options ?? new OrchestratorOptions();
    }

    public async Task<ChatResult> AskAsync(ChatRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(request.Message);

        string requestId = Guid.NewGuid().ToString("N")[..12];
        string conversationId = string.IsNullOrWhiteSpace(request.ConversationId)
            ? Guid.NewGuid().ToString("N")[..12]
            : request.ConversationId;

        var startedAt = DateTimeOffset.UtcNow;

        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(_options.Timeout);

        // --- 1. Phân loại việc, rồi lấy chính sách ---
        var classified = TaskClassifier.Classify(request.Message);
        var policy = Modes.Policy(request.Mode, classified);

        // --- 2. Dựng ngữ cảnh (chỉ phần LIÊN QUAN) ---
        var context = _context.Build(request.Message,
            request.SystemInstructions ?? _options.SystemInstructions);

        // --- 3. Dựng yêu cầu ---
        var baseRequest = new AiRequest
        {
            UserPrompt = context.UserPrompt,
            SystemPrompt = context.SystemPrompt,
            ConversationHistory = context.History,
            Model = _models.For(policy.TaskType),
            Temperature = policy.Temperature,
            MaxTokens = policy.MaxTokens,
            TaskType = policy.TaskType,
            MaxToolCalls = policy.AllowTools ? policy.MaxToolCalls : 0,
            Tools = policy.AllowTools ? _registry.Declarations : [],
            Metadata = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["requestId"] = requestId,
                ["conversationId"] = conversationId,
                ["mode"] = request.Mode.ToString(),
            },
        };

        var loop = new ToolLoop(_router, _registry);

        ToolLoopResult result;
        ValidationOutcome validation;
        int retries = 0;
        var current = baseRequest;

        // --- 4 & 5. Chạy, kiểm, viết lại nếu cần (tối đa 2 lần) ---
        while (true)
        {
            try
            {
                result = await loop.RunAsync(current, deadline.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                return Timeout(conversationId, requestId, request, policy, startedAt);
            }

            if (!result.Response.Success)
            {
                // Bộ định tuyến đã thử mọi provider. Viết lại prompt không sửa
                // được việc không có ai trả lời, nên dừng ở đây.
                return Failed(conversationId, requestId, request, policy, result, startedAt, context);
            }

            validation = _validator.Validate(result.Response, result.Exchanges);

            if (validation.IsValid || retries >= _options.MaxValidationRetries) break;

            retries++;

            current = current with
            {
                UserPrompt = $"{baseRequest.UserPrompt}\n\n" +
                             $"[Bản nháp trước chưa dùng được: {string.Join("; ", validation.Problems)}. " +
                             "Viết lại cho đầy đủ, và dùng đúng kết quả công cụ đã có.]",

                // Lần viết lại KHÔNG cấp thêm lượt công cụ: công cụ đã chạy rồi
                // và kết quả vẫn còn đó. Cấp thêm chỉ mời model chạy lại cùng
                // một công cụ để "chắc ăn".
                MaxToolCalls = 0,
                Tools = [],

                // NHƯNG kết quả đã chạy thì phải MANG THEO. Thiếu nó, lần viết
                // lại bảo model "dùng đúng kết quả công cụ" mà không đưa kết
                // quả nào — và bộ kiểm cũng không còn gì để đối chiếu, nên một
                // câu trả lời bịa sẽ lặng lẽ qua được ở lần thứ hai.
                ToolHistory = result.Exchanges,
            };
        }

        return new ChatResult
        {
            ConversationId = conversationId,
            Message = result.Response.Content,
            Model = result.Response.Model,
            Provider = result.Response.Provider,
            Usage = result.Response.Usage,
            Success = true,
            ToolCalls = Describe(result),
            Diagnostics = new ChatDiagnostics
            {
                RequestId = requestId,
                Mode = request.Mode,
                TaskType = policy.TaskType,
                Latency = DateTimeOffset.UtcNow - startedAt,
                ToolCount = result.ToolCallCount,
                RetryCount = retries,
                HitToolLimit = result.HitLimit,
                ContextCharacters = context.CharacterCount,
                ContextOmitted = context.Omitted,
                ValidationProblems = validation.Problems,
            },
        };
    }

    private static IReadOnlyList<ChatToolCall> Describe(ToolLoopResult result) =>
        [.. result.Exchanges.Select(e => new ChatToolCall(
            e.Result.Name, e.Result.Succeeded, e.Result.WasRefused, e.Result.Duration.TotalMilliseconds))];

    private ChatResult Failed(string conversationId, string requestId, ChatRequest request,
        ModePolicy policy, ToolLoopResult result, DateTimeOffset startedAt, BuiltContext context) =>
        new()
        {
            ConversationId = conversationId,
            Success = false,
            Error = result.Response.Error,
            Provider = result.Response.Provider,
            Model = result.Response.Model,
            Message = $"Không gọi được model: {result.Response.Error?.Message ?? "không rõ nguyên nhân"}",
            ToolCalls = Describe(result),
            Diagnostics = new ChatDiagnostics
            {
                RequestId = requestId,
                Mode = request.Mode,
                TaskType = policy.TaskType,
                Latency = DateTimeOffset.UtcNow - startedAt,
                ToolCount = result.ToolCallCount,
                ContextCharacters = context.CharacterCount,
                ContextOmitted = context.Omitted,
            },
        };

    private static ChatResult Timeout(string conversationId, string requestId, ChatRequest request,
        ModePolicy policy, DateTimeOffset startedAt) =>
        new()
        {
            ConversationId = conversationId,
            Success = false,
            Error = new AiError(AiErrorKind.Timeout, "hết hạn thời gian cho cả lượt hỏi"),
            Message = "Lượt hỏi này chạy quá lâu và đã bị dừng.",
            Diagnostics = new ChatDiagnostics
            {
                RequestId = requestId,
                Mode = request.Mode,
                TaskType = policy.TaskType,
                Latency = DateTimeOffset.UtcNow - startedAt,
            },
        };
}
