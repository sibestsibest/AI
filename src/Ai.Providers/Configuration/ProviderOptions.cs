using Ai.Providers.Contracts;

namespace Ai.Providers.Configuration;

/// <summary>Loại provider — quyết định lớp nào được dựng (xem <c>AiProviderFactory</c>).</summary>
public enum ProviderKind
{
    /// <summary>Máy chủ Ollama chạy tại máy. Không tốn tiền mỗi request.</summary>
    Ollama,

    /// <summary>Bất kỳ endpoint theo hợp đồng kiểu OpenAI (/v1/chat/completions).</summary>
    OpenAiCompatible,

    /// <summary>
    /// OpenAI qua SDK chính thức, hợp đồng Responses.
    ///
    /// Khác <see cref="OpenAiCompatible"/> ở chỗ nó KHÔNG phải một lớp gọi HTTP
    /// tự viết: nó dùng <c>OpenAI.Responses.ResponsesClient</c>, nên có sẵn gọi
    /// công cụ, tra web, kết quả có lược đồ, và tham số suy luận. Đổi lại, nó
    /// chỉ nói chuyện được với endpoint thật sự cài đặt hợp đồng Responses —
    /// endpoint "tương thích OpenAI" của bên thứ ba thường thì không.
    /// </summary>
    OpenAiResponses,

    /// <summary>Provider giả, trả lời theo kịch bản. Dùng cho demo và test — không gọi mạng.</summary>
    Scripted,
}

/// <summary>
/// Thứ tự ưu tiên khi chọn provider.
///
/// Mặc định là <see cref="LocalFirst"/>, và đó là lựa chọn có lý do: model chạy
/// tại máy không tốn tiền mỗi lượt gọi, không gửi dữ liệu người dùng ra ngoài,
/// và không có hạn mức. Ba điều đó quan trọng hơn chất lượng câu trả lời ở
/// khâu mà lớp này đảm nhiệm — DIỄN ĐẠT LẠI căn cứ đã có, chứ không phải tự
/// nghĩ ra nội dung.
/// </summary>
public enum ProviderStrategy
{
    /// <summary>Local trước, rồi đến các provider còn lại theo thứ tự cấu hình.</summary>
    LocalFirst,

    /// <summary>Đúng thứ tự khai trong cấu hình, không ưu tiên local.</summary>
    ConfiguredOrder,

    /// <summary>Chỉ dùng provider local. Không có local thì không gọi LLM.</summary>
    LocalOnly,
}

/// <summary>
/// CẤU HÌNH MỘT PROVIDER.
///
/// <see cref="ApiKeyEnvironmentVariable"/> là TÊN BIẾN MÔI TRƯỜNG, không phải
/// khoá. Cả hệ thống không có chỗ nào nhận khoá dạng chuỗi thẳng từ file cấu
/// hình — <c>AiProvidersFile</c> từ chối file có vẻ chứa khoá thật, và
/// <c>SecretResolver</c> là chỗ duy nhất đọc được giá trị khoá.
///
/// Vì sao bắt buộc gián tiếp như vậy? Vì file cấu hình bị commit. Một trường
/// tên <c>ApiKey</c> nhận chuỗi thẳng thì sớm muộn cũng có người điền khoá thật
/// vào để "thử cho nhanh", rồi commit. Không có trường đó thì không có chỗ để
/// điền.
/// </summary>
public sealed record ProviderOptions
{
    public required string Name { get; init; }

    public required ProviderKind Type { get; init; }

    public bool Enabled { get; init; } = true;

    /// <summary>Địa chỉ gốc. Với Ollama mặc định là máy tại chỗ.</summary>
    public string Endpoint { get; init; } = "";

    /// <summary>TÊN biến môi trường chứa khoá. Null = provider không cần khoá (ví dụ Ollama).</summary>
    public string? ApiKeyEnvironmentVariable { get; init; }

    public string? DefaultModel { get; init; }

    /// <summary>Các model được phép dùng ở provider này — đóng vai danh sách cho phép.</summary>
    public IReadOnlyList<string> Models { get; init; } = [];

    public int TimeoutSeconds { get; init; } = 30;

    public int MaxRetries { get; init; } = 2;

    /// <summary>
    /// Model theo TỪNG LOẠI VIỆC — ứng với "AI:Models:*" của đề bài.
    ///
    /// Khác <see cref="Models"/>: trường kia là DANH SÁCH CHO PHÉP (model nào
    /// được gọi), trường này là LỰA CHỌN (loại việc nào dùng model nào). Hai
    /// thứ khác nhau, và gộp lại sẽ mất khả năng nói "cho phép bốn model này,
    /// nhưng việc lập trình thì dùng đúng cái thứ ba".
    /// </summary>
    public AiModelMap TaskModels { get; init; } = new();

    /// <summary>
    /// Trần token RA cho một lượt gọi. 0 = không đặt trần ở lớp cấu hình.
    ///
    /// Trần này CẮT trên giá trị mà chỗ gọi yêu cầu, chứ không thay nó: một
    /// chỗ gọi xin 4000 token trong khi cấu hình cho 1000 thì được 1000. Nhờ
    /// vậy hạn mức chi phí là của người vận hành, không phải của người viết
    /// chỗ gọi.
    /// </summary>
    public int MaxOutputTokens { get; init; }

    /// <summary>Số lượt gọi công cụ tối đa mỗi yêu cầu. 0 = dùng mặc định của chỗ gọi.</summary>
    public int MaxToolCalls { get; init; }

    /// <summary>Mức "cố gắng suy luận" cho model hỗ trợ: minimal / low / medium / high. Null = mặc định.</summary>
    public string? ReasoningEffort { get; init; }

    /// <summary>Có cho phép công cụ tra web phía nhà cung cấp hay không.</summary>
    public bool EnableWebSearch { get; init; }

    /// <summary>
    /// Cho phép nhà cung cấp LƯU nội dung lượt gọi ở phía họ.
    ///
    /// Mặc định TẮT, và mặc định đó là quyết định có chủ ý: bật lên nghĩa là
    /// prompt và câu trả lời — tức là dữ liệu người dùng — nằm lại trên máy
    /// chủ bên thứ ba. Muốn dùng tính năng nối lượt phía nhà cung cấp thì phải
    /// bật TƯỜNG MINH, chứ không được có sẵn.
    /// </summary>
    public bool AllowProviderSideStorage { get; init; }

    /// <summary>Trần token ra đã áp dụng, cho một giá trị chỗ gọi yêu cầu.</summary>
    public int CapOutputTokens(int requested) =>
        MaxOutputTokens > 0 ? Math.Min(requested, MaxOutputTokens) : requested;

    /// <summary>Provider này chạy tại máy (không tốn tiền, không gửi dữ liệu ra ngoài).</summary>
    public bool IsLocal => Type is ProviderKind.Ollama or ProviderKind.Scripted;

    /// <summary>
    /// Loại việc mà provider này được ưu tiên. Rỗng = dùng cho mọi loại việc.
    ///
    /// Đây là toàn bộ phần "định tuyến theo loại việc": một danh sách trong cấu
    /// hình, không phải logic trong code. Nhờ vậy thêm một provider chuyên cho
    /// việc lập trình không cần sửa bộ định tuyến.
    /// </summary>
    public IReadOnlyList<AiTaskType> PreferredFor { get; init; } = [];

    public TimeSpan Timeout => TimeSpan.FromSeconds(Math.Clamp(TimeoutSeconds, 1, 300));

    public bool Handles(AiTaskType task) => PreferredFor.Count == 0 || PreferredFor.Contains(task);

    /// <summary>Model này có được phép dùng ở provider này không (danh sách cho phép).</summary>
    public bool AllowsModel(string? model) =>
        model is null
        || Models.Count == 0
        || Models.Contains(model, StringComparer.OrdinalIgnoreCase);

    public override string ToString() =>
        $"{Name} ({Type}{(IsLocal ? ", local" : "")}, model {DefaultModel ?? "?"}, {(Enabled ? "bật" : "tắt")})";
}

/// <summary>Toàn bộ cấu hình của lớp provider.</summary>
public sealed record AiProvidersOptions
{
    public ProviderStrategy Strategy { get; init; } = ProviderStrategy.LocalFirst;

    /// <summary>Tên provider được thử trước tiên, bất kể chiến lược. Null = theo chiến lược.</summary>
    public string? DefaultProvider { get; init; }

    public IReadOnlyList<ProviderOptions> Providers { get; init; } = [];

    /// <summary>
    /// Số ký tự tối đa được gửi đi trong một yêu cầu.
    ///
    /// Có hai lý do, và lý do thứ hai mới là lý do chính: chặn hoá đơn bất ngờ,
    /// và chặn việc một tài liệu dài vô tình bị đẩy nguyên văn ra provider bên
    /// ngoài. Cả hai đều xảy ra âm thầm nếu không có hạn mức.
    /// </summary>
    public int MaxPromptCharacters { get; init; } = 12_000;

    /// <summary>Số ký tự tối đa nhận về từ một câu trả lời.</summary>
    public int MaxResponseCharacters { get; init; } = 8_000;

    /// <summary>Provider đang bật, đã lọc theo danh sách cho phép.</summary>
    public IReadOnlyList<ProviderOptions> EnabledProviders => [.. Providers.Where(p => p.Enabled)];

    public ProviderOptions? Find(string name) =>
        Providers.FirstOrDefault(p => p.Name.Equals(name, StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Cấu hình mặc định khi không có file nào: CHỈ Ollama tại máy, và không có khoá nào.
    ///
    /// Mặc định này chạy được ngay mà không cần ai điền gì, và cũng không gửi
    /// một byte nào ra Internet nếu máy không chạy Ollama — lúc đó provider
    /// đơn giản là "không sẵn sàng", và hệ thống dùng câu trả lời khuôn mẫu
    /// như trước khi có lớp này.
    /// </summary>
    public static AiProvidersOptions LocalOnlyDefault { get; } = new()
    {
        Strategy = ProviderStrategy.LocalFirst,
        Providers =
        [
            new ProviderOptions
            {
                Name = "ollama",
                Type = ProviderKind.Ollama,
                Endpoint = "http://localhost:11434",
                DefaultModel = null,   // lấy model đầu tiên máy đang có
                TimeoutSeconds = 60,
                MaxRetries = 1,
            },
        ],
    };
}
