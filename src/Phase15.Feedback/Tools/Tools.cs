using System.Globalization;
using Ai.Phase15.Memory;
using Ai.Phase15.Reasoning;

namespace Ai.Phase15.Tools;

public sealed record ToolResult(bool Success, string Output, string? Detail = null)
{
    public static ToolResult Fail(string reason) => new(false, reason);
}

/// <summary>
/// Công cụ: thứ agent gọi ra khi tự nó không trả lời được.
///
/// Đây chính là điểm mấu chốt của kiến trúc agent. Một mạng neural đơn độc
/// phải NHỚ mọi thứ trong trọng số và TÍNH mọi thứ bằng phép nhân ma trận —
/// nên nó nhớ sai và tính sai. Agent thì không cần: nó chỉ cần nhận ra
/// "việc này nên giao cho máy tính bỏ túi", rồi gọi một hàm tính đúng 100%.
/// </summary>
public interface ITool
{
    string Name { get; }

    string Description { get; }

    ToolResult Execute(string input);
}

/// <summary>
/// Máy tính — bọc reasoning engine của Phase 9.
///
/// Vì sao không để mạng neural tự tính? Vì nó sẽ trả về 29.97 thay vì 30.
/// Mạng neural nội suy, không tính toán. Việc số học phải giao cho số học.
/// </summary>
public sealed class CalculatorTool(ReasoningEngine engine) : ITool
{
    public string Name => "calculator";

    public string Description => "tính toán số học và so sánh, có suy luận qua các biến đã biết";

    public ReasoningEngine Engine { get; } = engine ?? throw new ArgumentNullException(nameof(engine));

    public ToolResult Execute(string input)
    {
        ArgumentNullException.ThrowIfNull(input);

        try
        {
            var answer = Engine.Ask(input);
            string trace = string.Join("\n           ", answer.Steps.Select(s => s.Description));

            return new ToolResult(true, answer.Text, trace);
        }
        catch (Exception ex) when (ex is ReasoningException or ParseException)
        {
            return ToolResult.Fail(ex.Message);
        }
    }

    /// <summary>Nạp một sự thật vào engine ("A = 10").</summary>
    public ToolResult Define(string statement)
    {
        try
        {
            var fact = Engine.Learn(statement);
            return new ToolResult(true, $"đã ghi nhận {fact}");
        }
        catch (Exception ex) when (ex is ReasoningException or ParseException)
        {
            return ToolResult.Fail(ex.Message);
        }
    }
}

/// <summary>
/// Tra cứu kiến thức — kho tài liệu tĩnh, tìm theo từ khoá.
///
/// Đây là RAG (retrieval-augmented generation) ở dạng tối giản: thay vì bắt
/// model nhớ mọi định nghĩa trong trọng số, ta để nó TÌM trong tài liệu.
/// Ưu điểm: cập nhật tài liệu là xong, không phải train lại; và trích dẫn
/// được nguồn.
/// </summary>
public sealed class SearchTool : ITool
{
    private readonly List<(string Title, string Body, HashSet<string> TitleKeywords, HashSet<string> AllKeywords)> _documents = [];

    public string Name => "search";

    public string Description => "tra cứu định nghĩa và khái niệm trong kho tài liệu";

    public int DocumentCount => _documents.Count;

    /// <summary>
    /// Trùng dưới ngưỡng này thì coi như không tìm thấy.
    ///
    /// Không có ngưỡng thì "làm thế nào để nấu phở" vẫn trả về tài liệu về
    /// Attention, chỉ vì cả hai cùng chứa chữ "để" — một câu trả lời sai mà
    /// trông rất tự tin. Thà nói không tìm thấy.
    /// </summary>
    public double MinimumRelevance { get; init; } = 0.3;

    public void AddDocument(string title, string body)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(title);
        ArgumentException.ThrowIfNullOrWhiteSpace(body);

        var titleKeywords = Keywords.Extract(title);
        var allKeywords = new HashSet<string>(titleKeywords, StringComparer.OrdinalIgnoreCase);
        allKeywords.UnionWith(Keywords.Extract(body));

        _documents.Add((title, body, titleKeywords, allKeywords));
    }

    /// <summary>
    /// Điểm = độ khớp toàn văn + nửa độ khớp TIÊU ĐỀ.
    ///
    /// Vì sao cộng thêm phần tiêu đề? Hỏi "softmax" thì cả tài liệu "Softmax"
    /// lẫn tài liệu "Attention" (có nhắc tới softmax trong thân bài) đều khớp
    /// 100% theo từ khoá. Nhưng rõ ràng tài liệu ĐỀ TÊN softmax mới là thứ
    /// người dùng muốn. Tiêu đề nói lên tài liệu NÓI VỀ cái gì, thân bài chỉ
    /// nói nó CÓ NHẮC tới cái gì.
    /// </summary>
    public ToolResult Execute(string input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var query = Keywords.Extract(input);

        var best = _documents
            .Select(d => new
            {
                Doc = d,
                BodyScore = Keywords.Relevance(query, d.AllKeywords),
                TitleScore = Keywords.Relevance(query, d.TitleKeywords),
            })
            .Where(x => x.BodyScore >= MinimumRelevance)
            .OrderByDescending(x => x.BodyScore + 0.5 * x.TitleScore)
            .ThenBy(x => x.Doc.Title, StringComparer.Ordinal)
            .FirstOrDefault();

        return best is null
            ? ToolResult.Fail("không tìm thấy tài liệu nào khớp")
            : new ToolResult(true, best.Doc.Body,
                $"nguồn: \"{best.Doc.Title}\" (khớp {best.BodyScore:P0}, tiêu đề {best.TitleScore:P0})");
    }
}

public sealed record Order(int Id, string Customer, double Amount);

/// <summary>
/// Truy vấn cơ sở dữ liệu — bảng đơn hàng trong bộ nhớ.
///
/// Khác SearchTool ở chỗ: dữ liệu ở đây CÓ CẤU TRÚC và câu trả lời phải
/// CHÍNH XÁC. "Có bao nhiêu đơn hàng" chỉ có đúng một đáp án, và nó phải
/// được đếm chứ không được đoán.
/// </summary>
public sealed class DatabaseTool : ITool
{
    private readonly List<Order> _orders = [];

    public string Name => "database";

    public string Description => "đếm, tính tổng và liệt kê dữ liệu đơn hàng";

    public IReadOnlyList<Order> Orders => _orders;

    public void Insert(Order order) => _orders.Add(order);

    public ToolResult Execute(string input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var words = Keywords.Extract(input);

        // Thứ tự kiểm tra rất quan trọng: "doanh thu tổng cộng là bao nhiêu"
        // chứa cả từ của phép TỔNG lẫn từ của phép ĐẾM. Phép cụ thể hơn
        // (tổng, lớn nhất, liệt kê) phải được xét TRƯỚC phép chung nhất (đếm).
        if (Matches(words, "doanh", "thu", "tổng", "total", "revenue", "sum"))
        {
            double total = _orders.Sum(o => o.Amount);
            return new ToolResult(true, total.ToString("N0", CultureInfo.InvariantCulture),
                "SELECT SUM(amount) FROM orders");
        }

        if (Matches(words, "lớn", "nhất", "largest", "max", "biggest"))
        {
            if (_orders.Count == 0) return ToolResult.Fail("bảng rỗng");

            var largest = _orders.MaxBy(o => o.Amount)!;
            return new ToolResult(true,
                $"đơn #{largest.Id} của {largest.Customer}, {largest.Amount:N0}",
                "SELECT * FROM orders ORDER BY amount DESC LIMIT 1");
        }

        if (Matches(words, "liệt", "kê", "list", "show", "tất", "cả", "all"))
        {
            string list = string.Join("; ", _orders.Select(o => $"#{o.Id} {o.Customer} {o.Amount:N0}"));
            return new ToolResult(true, list, "SELECT * FROM orders");
        }

        if (Matches(words, "bao", "nhiêu", "đếm", "many", "count", "số"))
        {
            return new ToolResult(true, _orders.Count.ToString(CultureInfo.InvariantCulture),
                "SELECT COUNT(*) FROM orders");
        }

        return ToolResult.Fail("không hiểu truy vấn — chỉ hỗ trợ đếm, tổng, lớn nhất, liệt kê");
    }

    private static bool Matches(IReadOnlySet<string> words, params string[] candidates) =>
        candidates.Any(words.Contains);
}
