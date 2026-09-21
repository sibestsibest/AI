using Ai.Providers.Contracts;

namespace Ai.Providers.Routing;

/// <summary>
/// CHẾ ĐỘ TRẢ LỜI — do NGƯỜI DÙNG chọn, không do model tự chọn.
///
/// Đây là điểm khác biệt đáng nói nhất giữa bộ định tuyến này và cách làm
/// "để model tự quyết": chế độ là một tham số đầu vào tường minh, và
/// <see cref="Auto"/> quyết định bằng LUẬT viết sẵn (xem <c>TaskClassifier</c>),
/// không bằng một lượt gọi model để hỏi "câu này thuộc loại gì".
///
/// Vì sao không hỏi model? Vì lượt hỏi đó cũng tốn tiền và cũng có thể sai, mà
/// sai ở đó thì sai cả những bước sau — và không có gì kiểm được nó. Một bảng
/// từ khoá thì đọc được, sửa được, và test được.
/// </summary>
public enum AiMode
{
    /// <summary>Bộ định tuyến tự phân loại theo luật. MẶC ĐỊNH.</summary>
    Auto,

    /// <summary>Ưu tiên nhanh và rẻ: model nhỏ, không công cụ, ít token.</summary>
    Fast,

    /// <summary>Việc cần suy luận nhiều bước: model mạnh, nhiều token ra hơn.</summary>
    Reasoning,

    /// <summary>Cần thông tin mới: bật công cụ tra web.</summary>
    Research,

    /// <summary>Việc lập trình: model thiên về mã, nhiệt độ thấp.</summary>
    Coding,
}

/// <summary>
/// CHÍNH SÁCH cho một lượt gọi: loại việc, tham số sinh, và có được dùng công
/// cụ hay không.
///
/// Gom vào một record thay vì rải thành bốn tham số vì cả bốn phải ĐỔI CÙNG
/// NHAU. Chế độ Fast mà vẫn bật công cụ thì nó không còn nhanh; chế độ Research
/// mà tắt công cụ thì nó không tra được gì. Tách rời ra là mời gọi đúng hai lỗi
/// đó.
/// </summary>
public sealed record ModePolicy(
    AiTaskType TaskType,
    double Temperature,
    int MaxTokens,
    bool AllowTools,
    bool AllowWebSearch)
{
    /// <summary>Số lượt gọi công cụ tối đa trong MỘT lượt hỏi. Chặn vòng lặp vô hạn.</summary>
    public int MaxToolCalls { get; init; } = 4;

    /// <summary>Mức "cố gắng suy luận" cho model có hỗ trợ. Null = để provider dùng mặc định.</summary>
    public string? ReasoningEffort { get; init; }
}

/// <summary>
/// Chế độ -> chính sách. Một bảng tra, cố ý không có logic.
///
/// Mọi con số ở đây đều là mặc định hợp lý, và đều bị cấu hình ghi đè được
/// (xem <c>AiModelMap</c>). Chỗ này chỉ bảo đảm hệ thống chạy được khi chưa ai
/// cấu hình gì.
/// </summary>
public static class Modes
{
    public static ModePolicy Policy(AiMode mode, AiTaskType? classified = null) => mode switch
    {
        AiMode.Fast => new ModePolicy(AiTaskType.GeneralChat, 0.2, 512, AllowTools: false, AllowWebSearch: false),

        AiMode.Reasoning => new ModePolicy(AiTaskType.Reasoning, 0.1, 2_048, AllowTools: true, AllowWebSearch: false)
        {
            ReasoningEffort = "medium",
        },

        AiMode.Research => new ModePolicy(AiTaskType.WebResearch, 0.2, 1_536, AllowTools: true, AllowWebSearch: true),

        AiMode.Coding => new ModePolicy(AiTaskType.Coding, 0.1, 2_048, AllowTools: true, AllowWebSearch: false),

        // Auto: loại việc do bộ phân loại quyết, rồi quay lại lấy chính sách
        // của chế độ tương ứng. Không có nhánh riêng cho Auto — nếu có thì sẽ
        // tồn tại hai định nghĩa cho cùng một loại việc.
        _ => (classified ?? AiTaskType.GeneralChat) switch
        {
            AiTaskType.Reasoning => Policy(AiMode.Reasoning),
            AiTaskType.WebResearch => Policy(AiMode.Research),
            AiTaskType.Coding => Policy(AiMode.Coding),
            var task => new ModePolicy(task, 0.2, 1_024, AllowTools: true, AllowWebSearch: false),
        },
    };

    /// <summary>Đọc tên chế độ từ chuỗi người dùng gõ. Không nhận ra thì Auto.</summary>
    public static AiMode Parse(string? text) => text?.Trim().ToLowerInvariant() switch
    {
        "fast" or "nhanh" => AiMode.Fast,
        "reasoning" or "suyluan" or "suy luận" => AiMode.Reasoning,
        "research" or "tracuu" or "tra cứu" => AiMode.Research,
        "coding" or "code" or "lapTrinh" or "lập trình" => AiMode.Coding,
        _ => AiMode.Auto,
    };
}

/// <summary>
/// PHÂN LOẠI VIỆC BẰNG LUẬT — bước "Task Classification" của đường ống.
///
/// Cố ý dùng từ khoá chứ không dùng mạng neural, và cũng không hỏi model. Bộ
/// phân loại ý định của Phase 11 đã tồn tại và làm việc khác (chọn CÔNG CỤ);
/// lớp này chọn LOẠI MODEL, một quyết định thô hơn nhiều — bốn nhánh, và đoán
/// sai chỉ làm câu trả lời đắt hơn hoặc rẻ hơn cần thiết, không làm nó sai.
///
/// Đó là lý do một bảng từ khoá là đủ ở đây, trong khi ở Phase 11 thì không.
/// </summary>
public static class TaskClassifier
{
    private static readonly string[] CodingWords =
        ["code", "hàm", "function", "class", "bug", "lỗi biên dịch", "compile", "c#", "csharp",
         "python", "sql", "regex", "api", "refactor", "unit test", "exception", "stack trace"];

    private static readonly string[] ResearchWords =
        ["hôm nay", "mới nhất", "hiện nay", "gần đây", "tin tức", "giá", "phiên bản",
         "latest", "today", "news", "current", "release", "2025", "2026"];

    private static readonly string[] ReasoningWords =
        ["vì sao", "tại sao", "so sánh", "phân tích", "chứng minh", "đánh giá", "nên chọn",
         "why", "compare", "analyze", "prove", "trade-off", "đánh đổi"];

    public static AiTaskType Classify(string question)
    {
        ArgumentNullException.ThrowIfNull(question);

        string text = question.ToLowerInvariant();

        // THỨ TỰ QUAN TRỌNG. "phiên bản mới nhất của .NET" chứa cả từ lập trình
        // lẫn từ tin mới — và cái quyết định ở đây là TÍNH MỚI, vì đó là thứ
        // model không tự biết. Nên Research phải được xét trước Coding.
        if (ResearchWords.Any(w => text.Contains(w, StringComparison.Ordinal))) return AiTaskType.WebResearch;
        if (CodingWords.Any(w => text.Contains(w, StringComparison.Ordinal))) return AiTaskType.Coding;
        if (ReasoningWords.Any(w => text.Contains(w, StringComparison.Ordinal))) return AiTaskType.Reasoning;

        return AiTaskType.GeneralChat;
    }
}
