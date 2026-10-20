using System.Globalization;

namespace Ai.Providers.Contracts;

/// <summary>Vì sao model dừng sinh. Chuẩn hoá về đây từ các tên gọi khác nhau của từng provider.</summary>
public enum AiFinishReason
{
    Unknown,
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Error,
}

/// <summary>
/// LOẠI LỖI — phân theo việc CÓ NÊN THỬ LẠI hay không, không phân theo mã HTTP.
///
/// Đây là phân loại mà <c>RetryPolicy</c> và <c>ProviderHealthTracker</c> cần:
/// mã HTTP 401 và 429 đều là "4xx" nhưng một cái thử lại vô nghĩa (sai khoá thì
/// thử mấy lần cũng sai) còn một cái chỉ cần chờ.
/// </summary>
public enum AiErrorKind
{
    None,

    /// <summary>Quá hạn thời gian. Thử lại được.</summary>
    Timeout,

    /// <summary>Không nối được / DNS / socket. Thử lại được.</summary>
    Network,

    /// <summary>HTTP 429 hoặc provider nói vượt hạn mức. Thử lại SAU khi chờ.</summary>
    RateLimited,

    /// <summary>HTTP 401/403. KHÔNG thử lại — khoá sai hoặc không có quyền.</summary>
    Unauthorized,

    /// <summary>Model không tồn tại ở provider này. KHÔNG thử lại cùng model.</summary>
    ModelUnavailable,

    /// <summary>Provider trả về thứ không đọc được (JSON sai khuôn, rỗng). Thử lại được một lần.</summary>
    InvalidResponse,

    /// <summary>Provider báo lỗi phía nó (5xx). Thử lại được.</summary>
    ProviderError,

    /// <summary>Không có provider nào dùng được. KHÔNG thử lại.</summary>
    NoProviderAvailable,

    /// <summary>Yêu cầu bị chính ứng dụng chặn (quá lớn, chứa dữ liệu không được gửi).</summary>
    Rejected,
}

/// <summary>
/// Lỗi đã được lọc sạch để ghi log và hiện cho người dùng.
///
/// <see cref="Message"/> đi qua bộ lọc bí mật trước khi tới đây. Chỗ này quan
/// trọng hơn nó trông: thông điệp lỗi của HttpClient và của nhiều SDK có kèm
/// URL đầy đủ, mà URL của một số provider mang khoá API trong query string.
/// Ghi nguyên văn message là ghi khoá vào log.
/// </summary>
public sealed record AiError(AiErrorKind Kind, string Message)
{
    /// <summary>Thời gian provider yêu cầu chờ, nếu nó có nói (HTTP Retry-After).</summary>
    public TimeSpan? RetryAfter { get; init; }

    public bool IsRetryable => Kind is AiErrorKind.Timeout or AiErrorKind.Network
        or AiErrorKind.RateLimited or AiErrorKind.ProviderError or AiErrorKind.InvalidResponse;

    public override string ToString() => $"{Kind}: {Message}";
}

/// <summary>
/// Số token đã dùng — CÓ THỂ KHÔNG BIẾT, và đó là trạng thái hợp lệ.
///
/// Ollama trả về số token thật; nhiều endpoint khác trả về, nhưng có endpoint
/// không trả gì. <see cref="IsMeasured"/> phân biệt "đo được 0 token" với
/// "không biết" — gộp hai thứ đó thành 0 sẽ làm mọi thống kê chi phí sai mà
/// trông như đúng.
/// </summary>
public sealed record AiUsage(int? InputTokens, int? OutputTokens)
{
    public static AiUsage Unknown { get; } = new(null, null);

    public bool IsMeasured => InputTokens is not null || OutputTokens is not null;

    public int? TotalTokens => InputTokens is null && OutputTokens is null
        ? null
        : (InputTokens ?? 0) + (OutputTokens ?? 0);

    public override string ToString() => IsMeasured
        ? string.Format(CultureInfo.InvariantCulture, "{0} vào / {1} ra",
            InputTokens?.ToString(CultureInfo.InvariantCulture) ?? "?",
            OutputTokens?.ToString(CultureInfo.InvariantCulture) ?? "?")
        : "không đo được";
}

/// <summary>
/// MỘT CÂU TRẢ LỜI từ model — cùng một kiểu cho MỌI provider.
///
/// Đây là điều kiện để phần còn lại của ứng dụng không phải viết
/// <c>if (provider == "ollama") … else if (provider == "gemini") …</c>. Mỗi
/// provider tự dịch khuôn dạng riêng của nó về đây, và chỗ gọi chỉ thấy kiểu
/// này.
///
/// <see cref="Content"/> LUÔN là chuỗi (rỗng khi lỗi) chứ không phải null, để
/// chỗ gọi không phải kiểm null ở mọi nhánh. Muốn biết có thành công không thì
/// xem <see cref="Success"/>.
/// </summary>
public sealed record AiResponse
{
    public required bool Success { get; init; }

    public string Content { get; init; } = "";

    public required string Provider { get; init; }

    public string? Model { get; init; }

    public AiUsage Usage { get; init; } = AiUsage.Unknown;

    public AiFinishReason FinishReason { get; init; } = AiFinishReason.Unknown;

    public TimeSpan Latency { get; init; }

    public AiError? Error { get; init; }

    /// <summary>
    /// Những công cụ model XIN gọi ở lượt này. Rỗng = không xin gì.
    ///
    /// Là một DANH SÁCH chứ không phải một mục, vì hợp đồng Responses cho phép
    /// model xin nhiều công cụ song song trong cùng một lượt. Ép về một mục sẽ
    /// khiến những lượt còn lại biến mất lặng lẽ — model chờ kết quả của một
    /// lượt gọi mà nó không bao giờ nhận được, rồi lặp lại y hệt.
    ///
    /// Đây vẫn chỉ là lời ĐỀ NGHỊ. Quyết định có chạy hay không là của
    /// <c>ToolRegistry</c>, không phải của model.
    /// </summary>
    public IReadOnlyList<AiToolCall> ToolCalls { get; init; } = [];

    /// <summary>Tên công cụ đầu tiên model xin gọi — tiện cho chỗ chỉ dùng một công cụ.</summary>
    public string? ToolName => ToolCalls.Count > 0 ? ToolCalls[0].Name : null;

    /// <summary>Tham số của lượt xin gọi đầu tiên, dạng JSON thô.</summary>
    public string? ToolArgumentsJson => ToolCalls.Count > 0 ? ToolCalls[0].ArgumentsJson : null;

    /// <summary>Model có đang chờ kết quả công cụ hay không.</summary>
    public bool WantsTools => ToolCalls.Count > 0;

    public static AiResponse Ok(string provider, string? model, string content, TimeSpan latency,
        AiUsage? usage = null, AiFinishReason finishReason = AiFinishReason.Stop) =>
        new()
        {
            Success = true,
            Provider = provider,
            Model = model,
            Content = content,
            Latency = latency,
            Usage = usage ?? AiUsage.Unknown,
            FinishReason = finishReason,
        };

    /// <summary>
    /// Model xin gọi công cụ thay vì trả lời.
    ///
    /// <see cref="Success"/> là TRUE: lời gọi đã thành công, model chỉ chưa
    /// nói xong. Đánh dấu thất bại ở đây sẽ khiến bộ định tuyến coi provider
    /// là hỏng và chuyển sang provider khác — giữa một lượt hội thoại đang dở.
    /// </summary>
    public static AiResponse Tools(string provider, string? model, IReadOnlyList<AiToolCall> calls,
        TimeSpan latency, AiUsage? usage = null, string content = "") =>
        new()
        {
            Success = true,
            Provider = provider,
            Model = model,
            Content = content,
            ToolCalls = calls,
            Latency = latency,
            Usage = usage ?? AiUsage.Unknown,
            FinishReason = AiFinishReason.ToolCall,
        };

    public static AiResponse Fail(string provider, string? model, AiError error, TimeSpan latency = default) =>
        new()
        {
            Success = false,
            Provider = provider,
            Model = model,
            Error = error,
            Latency = latency,
            FinishReason = AiFinishReason.Error,
        };

    public override string ToString() => Success
        ? string.Format(CultureInfo.InvariantCulture, "[{0}/{1}] {2} ký tự, {3:F0} ms, {4}",
            Provider, Model ?? "?", Content.Length, Latency.TotalMilliseconds, Usage)
        : $"[{Provider}/{Model ?? "?"}] THẤT BẠI — {Error}";
}
