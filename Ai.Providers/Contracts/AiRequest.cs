namespace Ai.Providers.Contracts;

/// <summary>Vai của một lượt trong hội thoại. Cố ý chỉ có ba vai — đủ cho mọi provider hiện nay.</summary>
public enum ChatRole
{
    System,
    User,
    Assistant,
    Tool,
}

/// <summary>Một lượt hội thoại gửi cho model.</summary>
public sealed record ChatMessage(ChatRole Role, string Content)
{
    /// <summary>Tên công cụ, khi <see cref="Role"/> là <see cref="ChatRole.Tool"/>.</summary>
    public string? ToolName { get; init; }

    public override string ToString() => $"{Role}: {Content}";
}

/// <summary>
/// Khai báo một công cụ cho model — KHÔNG phải bản thân công cụ.
///
/// Chỉ có tên, mô tả và lược đồ tham số. Model chỉ được XIN gọi công cụ; việc
/// gọi thật do phía ứng dụng làm, sau khi qua kiểm (xem tầng orchestration).
/// Đó là lý do ở đây không có delegate nào: một khai báo công cụ mà mang theo
/// code thực thi thì chỗ nào nhận nó cũng có thể chạy code đó.
/// </summary>
public sealed record AiToolDefinition(string Name, string Description)
{
    /// <summary>Lược đồ tham số dạng JSON Schema, để nguyên chuỗi cho provider chuyển tiếp.</summary>
    public string ParametersJsonSchema { get; init; } = """{"type":"object","properties":{}}""";
}

/// <summary>
/// LOẠI VIỆC — dùng để định tuyến, không dùng để đổi cách xử lý câu trả lời.
///
/// Bộ định tuyến chọn provider/model theo loại việc (xem <c>AiProviderRouter</c>).
/// Danh sách này cố tình khớp với đề bài, và <see cref="GeneralChat"/> là mặc
/// định — một loại việc chưa được cấu hình thì vẫn phải chạy được, chứ không
/// ném lỗi cấu hình vào mặt người dùng.
/// </summary>
public enum AiTaskType
{
    GeneralChat,
    Coding,
    Reasoning,
    Summarization,
    Translation,
    DocumentAnalysis,
    WebResearch,
    ToolCalling,
    ImageGeneration,
}

/// <summary>
/// MỘT YÊU CẦU gửi tới model — độc lập hoàn toàn với provider.
///
/// Không có trường nào mang tên nhà cung cấp, không có tham số riêng của một
/// API cụ thể. Nhờ vậy cùng một <see cref="AiRequest"/> gửi được cho Ollama,
/// cho một endpoint kiểu OpenAI, hay cho Gemini — việc dịch sang khuôn dạng
/// riêng là của từng <see cref="IAiProvider"/>, không phải của người gọi.
///
/// <see cref="Model"/> để null nghĩa là "dùng model mặc định của provider được
/// chọn". Đây là mặc định NÊN dùng: ghi cứng tên model ở chỗ gọi sẽ làm câu
/// hỏi chỉ chạy được với đúng một provider.
/// </summary>
public sealed record AiRequest
{
    public required string UserPrompt { get; init; }

    /// <summary>Chỉ thị hệ thống. Nội dung không tin cậy KHÔNG được đặt vào đây.</summary>
    public string? SystemPrompt { get; init; }

    public IReadOnlyList<ChatMessage> ConversationHistory { get; init; } = [];

    /// <summary>Null = để provider dùng model mặc định đã cấu hình.</summary>
    public string? Model { get; init; }

    public double Temperature { get; init; } = 0.2;

    public int MaxTokens { get; init; } = 1024;

    public IReadOnlyList<AiToolDefinition> Tools { get; init; } = [];

    public AiTaskType TaskType { get; init; } = AiTaskType.GeneralChat;

    /// <summary>
    /// Bắt model trả về JSON ĐÚNG MỘT LƯỢC ĐỒ. Null = trả lời tự do.
    ///
    /// Dùng cho những việc mà kết quả sẽ được MÁY đọc chứ không phải người:
    /// phân loại ý định, rút tham số công cụ, trích dữ liệu có cấu trúc. Ở
    /// những việc đó, "gần đúng" không dùng được — một dấu ngoặc thừa là hỏng
    /// cả bước sau.
    ///
    /// Provider nào không hỗ trợ thì BỎ QUA trường này, và phần phân tích JSON
    /// ở <c>StructuredOutput</c> vẫn chạy (chỉ là hay hỏng hơn). Đó là lý do
    /// lược đồ ở đây là gợi ý cho provider, còn chốt chặn thật nằm ở chỗ đọc
    /// kết quả.
    /// </summary>
    public StructuredFormat? ResponseFormat { get; init; }

    /// <summary>
    /// Số lượt gọi công cụ tối đa cho MỘT yêu cầu. Chặn vòng lặp model ↔ công cụ.
    /// </summary>
    public int MaxToolCalls { get; init; } = 4;

    /// <summary>
    /// Kết quả công cụ của những lượt TRƯỚC trong cùng một yêu cầu.
    ///
    /// Vòng lặp công cụ gửi lại toàn bộ yêu cầu kèm những gì đã chạy, thay vì
    /// giữ trạng thái trong provider. Nhờ vậy provider vẫn KHÔNG TRẠNG THÁI —
    /// và một lượt gọi bị chuyển sang provider dự phòng giữa chừng vẫn mang
    /// theo đủ ngữ cảnh.
    /// </summary>
    public IReadOnlyList<AiToolExchange> ToolHistory { get; init; } = [];

    /// <summary>
    /// Siêu dữ liệu của riêng ứng dụng (id phiên, id lượt…) để ghi log và truy vết.
    ///
    /// KHÔNG gửi ra ngoài: provider chỉ nhận prompt và tham số sinh. Nếu nó
    /// được gửi kèm thì mọi id nội bộ của hệ thống sẽ nằm trong log của một
    /// dịch vụ bên thứ ba.
    /// </summary>
    public IReadOnlyDictionary<string, string> Metadata { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>Toàn bộ lượt gửi cho model, theo đúng thứ tự: system → lịch sử → câu hỏi mới.</summary>
    public IReadOnlyList<ChatMessage> BuildMessages()
    {
        var messages = new List<ChatMessage>();

        if (!string.IsNullOrWhiteSpace(SystemPrompt))
        {
            messages.Add(new ChatMessage(ChatRole.System, SystemPrompt));
        }

        messages.AddRange(ConversationHistory);
        messages.Add(new ChatMessage(ChatRole.User, UserPrompt));

        return messages;
    }

    /// <summary>Số ký tự sẽ gửi đi — dùng để chặn trước khi gọi mạng, và để ước lượng token.</summary>
    public int CharacterCount => BuildMessages().Sum(m => m.Content.Length);
}
