using System.Text.Json;
using Ai.Phase16.Tools;

namespace Ai.Providers.Tools;

/// <summary>
/// NỐI CÔNG CỤ SẴN CÓ CỦA MINIAI VÀO SỔ ĐĂNG KÝ — không viết lại công cụ nào.
///
/// Công cụ của Phase 11–12 nhận MỘT CHUỖI (<c>ITool.Execute(string)</c>), còn
/// model gọi công cụ bằng MỘT OBJECT JSON. Lớp này làm đúng một việc: dịch
/// giữa hai khuôn đó, và không làm gì khác.
///
/// VÌ SAO KHÔNG SỬA <c>ITool</c> CHO NHẬN JSON? Vì công cụ của các phase phải
/// chạy y như cũ khi KHÔNG có LLM nào — đó là điều kiện để Phase 1–16 còn đọc
/// được độc lập. Một bộ điều hợp ở lớp provider thì chỉ lớp provider phải
/// biết; sửa giao diện gốc thì mọi phase đều phải biết.
///
/// QUYỀN được gán ở đây, không phải trong công cụ: cùng một công cụ có thể
/// chỉ-đọc trong ngữ cảnh này và phải-duyệt trong ngữ cảnh khác, và bản thân
/// công cụ không có cách nào biết nó đang ở ngữ cảnh nào.
/// </summary>
public static class PhaseTools
{
    /// <summary>
    /// Lược đồ cho công cụ nhận một câu hỏi bằng chữ.
    ///
    /// Cố ý đúng MỘT trường bắt buộc. Lược đồ càng nhiều nhánh thì model càng
    /// có nhiều cách điền sai, mà phía sau vẫn chỉ là một chuỗi.
    /// </summary>
    private static string QuerySchema(string description) =>
        "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\",\"description\":"
        + JsonSerializer.Serialize(description)
        + "}},\"required\":[\"query\"],\"additionalProperties\":false}";

    /// <summary>Bọc một công cụ đồng bộ của phase.</summary>
    public static RegisteredTool From(ITool tool, string argumentDescription,
        ToolPermission permission = ToolPermission.ReadOnly)
    {
        ArgumentNullException.ThrowIfNull(tool);

        return new RegisteredTool(
            tool.Name,
            tool.Description,
            QuerySchema(argumentDescription),
            (arguments, _) => Task.FromResult(Run(() => tool.Execute(ReadQuery(arguments)))))
        {
            Permission = permission,
        };
    }

    /// <summary>Bọc một công cụ bất đồng bộ của phase (ví dụ Internet).</summary>
    public static RegisteredTool From(IAsyncTool tool, string argumentDescription,
        ToolPermission permission = ToolPermission.ReadOnly, TimeSpan? timeout = null)
    {
        ArgumentNullException.ThrowIfNull(tool);

        return new RegisteredTool(
            tool.Name,
            tool.Description,
            QuerySchema(argumentDescription),
            async (arguments, token) =>
            {
                var result = await tool.ExecuteAsync(ReadQuery(arguments), token).ConfigureAwait(false);
                return Describe(result);
            })
        {
            Permission = permission,
            Timeout = timeout ?? TimeSpan.FromSeconds(45),
        };
    }

    /// <summary>
    /// Đăng ký mọi công cụ của một agent đã dựng sẵn.
    ///
    /// Nhận danh sách <c>ITool</c>/<c>IAsyncTool</c> thay vì các kiểu cụ thể,
    /// vì <c>AiController</c> phơi ra đúng hai danh sách đó — và agent được
    /// DỰNG LẠI mỗi lần huấn luyện lại, nên sổ đăng ký cũng phải dựng lại từ
    /// agent hiện tại chứ không giữ tham chiếu cũ.
    ///
    /// Mọi công cụ của Phase 11–12 đều CHỈ ĐỌC: tính toán, tra tài liệu, đếm
    /// đơn hàng, tra web. Không cái nào ghi gì, nên tất cả vào mức ReadOnly.
    /// Công cụ GHI về sau phải tự khai <see cref="ToolPermission.RequiresApproval"/>.
    /// </summary>
    public static ToolRegistry ForAgent(
        IReadOnlyList<ITool> tools,
        IReadOnlyList<IAsyncTool>? asyncTools = null,
        IToolAuthorizer? authorizer = null)
    {
        ArgumentNullException.ThrowIfNull(tools);

        var registry = new ToolRegistry(authorizer);

        foreach (var tool in tools)
        {
            registry.Register(From(tool, tool.Description));
        }

        foreach (var tool in asyncTools ?? [])
        {
            registry.Register(From(tool, tool.Description));
        }

        return registry;
    }

    /// <summary>
    /// Đăng ký bộ công cụ chuẩn của một agent.
    ///
    /// Cả bốn đều là CHỈ ĐỌC: tính toán, tra tài liệu, đếm đơn hàng và tra web
    /// đều không sửa gì. Công cụ nào GHI thì phải tự khai
    /// <see cref="ToolPermission.RequiresApproval"/> khi đăng ký — và đó là lý
    /// do hàm này không nhận một tham số "cho phép tất".
    /// </summary>
    public static ToolRegistry Standard(
        CalculatorTool calculator,
        SearchTool search,
        DatabaseTool database,
        IAsyncTool? internet = null,
        IToolAuthorizer? authorizer = null)
    {
        ArgumentNullException.ThrowIfNull(calculator);
        ArgumentNullException.ThrowIfNull(search);
        ArgumentNullException.ThrowIfNull(database);

        var registry = new ToolRegistry(authorizer)
            .Register(From(calculator, "biểu thức số học cần tính, ví dụ \"2 + 3 * 4\""))
            .Register(From(search, "khái niệm cần tra trong kho tài liệu nội bộ"))
            .Register(From(database, "câu hỏi về đơn hàng: đếm, tổng, lớn nhất, liệt kê"));

        if (internet is not null)
        {
            registry.Register(From(internet, "từ khoá cần tra trên Internet"));
        }

        return registry;
    }

    /// <summary>
    /// Lấy trường <c>query</c>. Thiếu thì lấy nguyên object làm chuỗi.
    ///
    /// Đường lui này có chủ ý: model đôi khi gửi <c>{"input": "..."}</c> hoặc
    /// <c>{"expression": "..."}</c> dù lược đồ nói khác. Công cụ phía sau nhận
    /// chuỗi tự do và tự phân tích, nên đưa nó chuỗi gần đúng vẫn tốt hơn là
    /// từ chối cả lượt gọi vì sai tên trường.
    /// </summary>
    private static string ReadQuery(JsonElement arguments)
    {
        if (arguments.TryGetProperty("query", out var query) && query.ValueKind == JsonValueKind.String)
        {
            return query.GetString() ?? "";
        }

        foreach (var property in arguments.EnumerateObject())
        {
            if (property.Value.ValueKind == JsonValueKind.String)
            {
                return property.Value.GetString() ?? "";
            }
        }

        return arguments.ToString();
    }

    private static string Run(Func<ToolResult> execute) => Describe(execute());

    /// <summary>
    /// Kết quả công cụ thành chữ cho model đọc.
    ///
    /// Giữ cả <c>Detail</c> khi có, vì đó là phần truy nguồn (tên tài liệu, câu
    /// SQL, các bước suy luận). Bỏ nó đi thì model có con số mà không có cách
    /// nào nói con số đó ở đâu ra.
    /// </summary>
    private static string Describe(ToolResult result) => result.Success
        ? result.Detail is { Length: > 0 } detail
            ? $"{result.Output}\n(nguồn: {detail})"
            : result.Output
        : $"CÔNG CỤ HỎNG: {result.Output}";
}
