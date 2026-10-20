using System.Diagnostics;
using System.Text.Json;
using Ai.Providers.Contracts;

namespace Ai.Providers.Tools;

/// <summary>
/// QUYỀN của một công cụ. Mặc định là mức NGẶT NHẤT, có chủ ý.
///
/// Một công cụ quên khai quyền phải là công cụ KHÔNG chạy được, chứ không phải
/// công cụ chạy tự do. Nhầm lẫn theo chiều ngược lại là nhầm lẫn không sửa được
/// sau khi nó đã xảy ra một lần.
/// </summary>
public enum ToolPermission
{
    /// <summary>Phải hỏi ý người dùng trước mỗi lượt chạy. MẶC ĐỊNH.</summary>
    RequiresApproval = 0,

    /// <summary>Chỉ đọc, không đổi gì, không tốn tiền — chạy thẳng.</summary>
    ReadOnly = 1,

    /// <summary>Đã khai nhưng đang bị chặn. Model vẫn thấy tên, nhưng gọi thì bị từ chối.</summary>
    Denied = 2,
}

/// <summary>
/// MỘT CÔNG CỤ ĐÃ ĐĂNG KÝ — khai báo CỘNG phần thực thi.
///
/// Khác <see cref="AiToolDefinition"/> ở đúng một điểm, và điểm đó là cả lý do
/// hai kiểu này tách nhau: kiểu kia KHÔNG mang code, vì nó được gửi RA NGOÀI.
/// Kiểu này mang <see cref="Handler"/> và KHÔNG BAO GIỜ rời khỏi máy.
///
/// <c>ToolRegistry</c> là chỗ duy nhất giữ ánh xạ giữa hai kiểu. Nhờ vậy model
/// biết tên công cụ và lược đồ tham số, nhưng không có đường nào chạm tới
/// delegate phía sau.
/// </summary>
public sealed record RegisteredTool(
    string Name,
    string Description,
    string ParametersJsonSchema,
    Func<JsonElement, CancellationToken, Task<string>> Handler)
{
    public ToolPermission Permission { get; init; } = ToolPermission.RequiresApproval;

    /// <summary>Hạn thời gian cho MỘT lượt chạy. Công cụ treo không được kéo cả hội thoại theo.</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>Khai báo gửi cho model — KHÔNG mang theo <see cref="Handler"/>.</summary>
    public AiToolDefinition Declare() =>
        new(Name, Description) { ParametersJsonSchema = ParametersJsonSchema };
}

/// <summary>Quyết định cho phép hay không một lượt gọi cần duyệt.</summary>
public interface IToolAuthorizer
{
    /// <summary>Trả về null nếu cho phép, hoặc lý do từ chối.</summary>
    string? Authorize(RegisteredTool tool, JsonElement arguments);
}

/// <summary>
/// Bộ duyệt mặc định: TỪ CHỐI mọi công cụ cần duyệt.
///
/// Nghe có vẻ vô dụng, nhưng đây chính là hành vi đúng khi chưa ai cắm bộ duyệt
/// thật vào. Mặc định "cho qua" sẽ khiến một ứng dụng quên cấu hình chạy mọi
/// công cụ nguy hiểm mà không ai biết — và nó sẽ chạy đúng như vậy suốt nhiều
/// tháng trước khi có người để ý.
/// </summary>
public sealed class DenyByDefaultAuthorizer : IToolAuthorizer
{
    public string? Authorize(RegisteredTool tool, JsonElement arguments) =>
        tool.Permission == ToolPermission.ReadOnly
            ? null
            : $"công cụ '{tool.Name}' cần được duyệt, nhưng chưa có bộ duyệt nào được cấu hình";
}

/// <summary>Bộ duyệt dựa trên một hàm — để màn hình hỏi người dùng, và để test dựng tình huống.</summary>
public sealed class DelegateToolAuthorizer(Func<RegisteredTool, JsonElement, string?> decide) : IToolAuthorizer
{
    private readonly Func<RegisteredTool, JsonElement, string?> _decide =
        decide ?? throw new ArgumentNullException(nameof(decide));

    public string? Authorize(RegisteredTool tool, JsonElement arguments) => _decide(tool, arguments);

    /// <summary>Cho phép mọi thứ. CHỈ dùng trong test — tên dài để không ai gõ nhầm vào code thật.</summary>
    public static DelegateToolAuthorizer AllowEverythingForTesting() => new((_, _) => null);
}

public interface IToolRegistry
{
    /// <summary>Khai báo để gửi cho model. Công cụ <see cref="ToolPermission.Denied"/> KHÔNG có ở đây.</summary>
    IReadOnlyList<AiToolDefinition> Declarations { get; }

    /// <summary>Chạy một lượt model xin gọi. KHÔNG ném — mọi hỏng hóc trả về trong kết quả.</summary>
    Task<AiToolResult> ExecuteAsync(AiToolCall call, CancellationToken cancellationToken = default);
}

/// <summary>
/// SỔ ĐĂNG KÝ CÔNG CỤ — MiniAI quyết định, model chỉ đề nghị.
///
/// Mọi lượt gọi đi qua đúng bốn chốt, theo thứ tự, và thứ tự này có lý do:
///
///   1. CÓ ĐĂNG KÝ KHÔNG?   tên lạ -> từ chối. Model bịa ra tên công cụ là
///                          chuyện thường; chạy theo tên nó bịa thì không.
///   2. CÓ BỊ CHẶN KHÔNG?   Denied -> từ chối, kể cả khi đã đăng ký.
///   3. THAM SỐ CÓ ĐỌC ĐƯỢC KHÔNG?
///                          JSON sai khuôn -> từ chối TRƯỚC khi gọi handler.
///                          Đây là chỗ duy nhất chuỗi thô của model thành dữ
///                          liệu có kiểu, nên nó cũng là chỗ duy nhất phải
///                          chịu lỗi phân tích.
///   4. CÓ ĐƯỢC DUYỆT KHÔNG?
///                          <see cref="IToolAuthorizer"/> nói không -> từ chối.
///
/// Qua hết bốn chốt mới chạy, và chạy có hạn thời gian.
///
/// LỖI KHÔNG ĐƯỢC NÉM RA NGOÀI. Người gọi là vòng lặp công cụ, và nó phải gửi
/// được MỘT ĐIỀU GÌ ĐÓ về cho model ở mọi nhánh — kể cả nhánh "công cụ của bạn
/// vừa nổ tung". Ném ngoại lệ ở đây sẽ làm hỏng cả lượt hội thoại vì một công
/// cụ hỏng.
/// </summary>
public sealed class ToolRegistry : IToolRegistry
{
    private readonly Dictionary<string, RegisteredTool> _tools = new(StringComparer.OrdinalIgnoreCase);
    private readonly IToolAuthorizer _authorizer;

    public ToolRegistry(IToolAuthorizer? authorizer = null) =>
        _authorizer = authorizer ?? new DenyByDefaultAuthorizer();

    public IReadOnlyList<AiToolDefinition> Declarations =>
        [.. _tools.Values.Where(t => t.Permission != ToolPermission.Denied).Select(t => t.Declare())];

    public IReadOnlyCollection<RegisteredTool> All => _tools.Values;

    public int Count => _tools.Count;

    /// <summary>Số lượt bị từ chối — con số đáng theo dõi, vì nó tăng khi model đang dò dẫm.</summary>
    public int RefusedCount { get; private set; }

    public int ExecutedCount { get; private set; }

    public ToolRegistry Register(RegisteredTool tool)
    {
        ArgumentNullException.ThrowIfNull(tool);
        ArgumentException.ThrowIfNullOrWhiteSpace(tool.Name);

        // Đăng ký đè là lỗi LẬP TRÌNH, nên chỗ này ném thật: hai công cụ khác
        // nhau cùng tên nghĩa là model gọi một cái và nhận cái kia.
        if (!_tools.TryAdd(tool.Name, tool))
        {
            throw new InvalidOperationException($"công cụ '{tool.Name}' đã được đăng ký rồi");
        }

        return this;
    }

    public RegisteredTool? Find(string name) =>
        _tools.TryGetValue(name, out var tool) ? tool : null;

    public async Task<AiToolResult> ExecuteAsync(AiToolCall call, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(call);

        // --- Chốt 1: có đăng ký không ---
        if (!_tools.TryGetValue(call.Name, out var tool))
        {
            return Refuse(call, $"không có công cụ nào tên '{call.Name}'");
        }

        // --- Chốt 2: có bị chặn không ---
        if (tool.Permission == ToolPermission.Denied)
        {
            return Refuse(call, $"công cụ '{tool.Name}' đang bị chặn");
        }

        // --- Chốt 3: tham số có đọc được không ---
        JsonElement arguments;

        try
        {
            using var document = JsonDocument.Parse(
                string.IsNullOrWhiteSpace(call.ArgumentsJson) ? "{}" : call.ArgumentsJson);

            arguments = document.RootElement.Clone();
        }
        catch (JsonException ex)
        {
            return Refuse(call, $"tham số không phải JSON hợp lệ: {ex.Message}");
        }

        if (arguments.ValueKind != JsonValueKind.Object)
        {
            return Refuse(call, $"tham số phải là một object, nhận được {arguments.ValueKind}");
        }

        // --- Chốt 4: có được duyệt không ---
        if (_authorizer.Authorize(tool, arguments) is { } refusal)
        {
            return Refuse(call, refusal);
        }

        // --- Chạy, có hạn thời gian ---
        var stopwatch = Stopwatch.StartNew();

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(tool.Timeout);

        try
        {
            string output = await tool.Handler(arguments, timeout.Token).ConfigureAwait(false);

            ExecutedCount++;
            stopwatch.Stop();

            return new AiToolResult(call.CallId, tool.Name, true, output) { Duration = stopwatch.Elapsed };
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            stopwatch.Stop();

            return new AiToolResult(call.CallId, tool.Name, false,
                $"công cụ chạy quá {tool.Timeout.TotalSeconds:F0}s và đã bị dừng")
            {
                Duration = stopwatch.Elapsed,
            };
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            stopwatch.Stop();

            // Thông điệp của ngoại lệ đi NGƯỢC ra model, nên nó phải được lọc
            // như mọi thứ khác rời khỏi máy: ngoại lệ hay mang theo đường dẫn
            // đầy đủ, chuỗi kết nối, và đôi khi cả khoá.
            return new AiToolResult(call.CallId, tool.Name, false,
                Security.OutboundRedactor.Redact($"{ex.GetType().Name}: {ex.Message}"))
            {
                Duration = stopwatch.Elapsed,
            };
        }
    }

    private AiToolResult Refuse(AiToolCall call, string reason)
    {
        RefusedCount++;

        return new AiToolResult(call.CallId, call.Name, false, reason) { WasRefused = true };
    }
}
