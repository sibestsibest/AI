using System.Text;
using Ai.Phase12.Memory;

namespace Ai.Phase12.Internet;

/// <summary>Biến câu hỏi của người dùng thành (các) truy vấn gửi cho máy tìm kiếm.</summary>
public interface ISearchQueryGenerator
{
    IReadOnlyList<SearchQuery> Generate(string question, int maxResults = 5);
}

/// <summary>
/// Sinh truy vấn bằng từ khoá — dùng lại nguyên bộ tách từ của Phase 10.
///
/// VÌ SAO KHÔNG GỬI THẲNG CÂU HỎI?
///
/// "bạn có thể cho tôi biết attention hoạt động như thế nào không" — gửi
/// nguyên câu này thì máy tìm kiếm phải cân 14 từ, trong đó 11 từ là lời dẫn
/// không mang thông tin. Rút còn "attention hoạt động" thì đúng trọng tâm hơn.
///
/// Sinh HAI biến thể, không phải một:
///
///   1. Bản đầy đủ  — mọi từ khoá, giữ nguyên thứ tự trong câu gốc
///   2. Bản rút gọn — chỉ các từ khoá nặng nhất (bỏ lời dẫn hỏi)
///
/// Vì sao cần cả hai? Bản đầy đủ chính xác hơn nhưng dễ ra 0 kết quả; bản rút
/// gọn rộng hơn nên hầu như luôn có gì đó. Thứ tự này cũng là thứ tự ưu tiên:
/// dùng bản đầu, cạn kết quả thì mới xuống bản sau.
///
/// Thứ tự từ được GIỮ NGUYÊN theo câu gốc. Keywords.Extract trả về HashSet
/// nên mất thứ tự — mà "học máy" khác "máy học", nên ở đây phải tự tách từ
/// theo thứ tự rồi mới lọc bằng tập từ khoá của Phase 10.
/// </summary>
public sealed class KeywordSearchQueryGenerator : ISearchQueryGenerator
{
    /// <summary>
    /// Từ dẫn của câu hỏi. Chúng KHÔNG nằm trong stop word của Phase 10 (ở đó
    /// chúng có ích cho việc so khớp ký ức), nhưng với máy tìm kiếm thì chỉ là
    /// nhiễu.
    ///
    /// Hai nhóm, và nhóm thứ hai mới là nhóm dễ bị bỏ sót:
    ///
    ///   1. Từ dẫn hỏi     — "cho tôi biết", "giải thích", "tell me about"
    ///   2. Từ chỉ KÊNH    — "trên mạng", "internet", "online", "thông tin",
    ///                       "mới nhất". Chúng nói agent phải đi đâu tìm, chứ
    ///                       không nói tìm CÁI GÌ.
    ///
    /// Nhóm 2 quan trọng vì đúng những từ này là thứ giúp bộ phân loại nhận ra
    /// ý định SearchInternet — nhưng gửi chúng cho máy tìm kiếm thì tai hại:
    /// "tìm trên mạng thông tin về SSRF" có 6 từ, chỉ 1 từ là chủ đề thật.
    /// Độ khớp bị chia cho 6 nên tài liệu đúng cũng tụt xuống dưới ngưỡng, và
    /// agent kết luận "không tìm thấy gì" trong khi tài liệu nằm ngay đó.
    /// </summary>
    private static readonly HashSet<string> QuestionLeadWords = new(StringComparer.OrdinalIgnoreCase)
    {
        // --- nhóm 1: từ dẫn hỏi ---
        // tiếng Việt
        "cho", "biết", "giải", "thích", "hãy", "giúp", "tôi", "bạn", "mình", "thế",
        "sao", "nào", "vậy", "hỏi", "tìm", "tra", "cứu", "kiếm", "xem", "muốn",
        "thể", "như", "nghĩa", "khái", "niệm", "định",
        // tiếng Anh
        "can", "could", "would", "tell", "me", "about", "please", "explain",
        "search", "find", "look", "up", "know", "want", "definition", "mean",
        "means", "meaning", "concept", "give", "show",

        // --- nhóm 2: từ chỉ kênh và tính mới ---
        "trên", "mạng", "lên", "internet", "online", "web", "google",
        "thông", "tin", "trực", "tuyến",
        "mới", "nhất", "hiện", "nay", "tại", "latest", "news", "current", "now",
    };

    /// <summary>Tối đa bao nhiêu từ trong một truy vấn — dài hơn thì máy tìm kiếm ra rỗng.</summary>
    public int MaxTermsPerQuery { get; init; } = 8;

    public IReadOnlyList<SearchQuery> Generate(string question, int maxResults = 5)
    {
        ArgumentNullException.ThrowIfNull(question);

        var meaningful = Keywords.Extract(question);
        if (meaningful.Count == 0) return [];

        // Tách theo THỨ TỰ, rồi giữ lại từ nào có trong tập từ khoá.
        var ordered = OrderedTerms(question).Where(meaningful.Contains).ToList();
        if (ordered.Count == 0) return [];

        var queries = new List<SearchQuery>();

        string full = string.Join(' ', ordered.Take(MaxTermsPerQuery));
        queries.Add(new SearchQuery(full, maxResults));

        // Bản rút gọn: bỏ từ dẫn hỏi. Chỉ thêm nếu nó thật sự khác bản đầy đủ
        // và không rút đến mức rỗng.
        var core = ordered.Where(t => !QuestionLeadWords.Contains(t)).Take(MaxTermsPerQuery).ToList();

        if (core.Count > 0)
        {
            string narrowed = string.Join(' ', core);
            if (!narrowed.Equals(full, StringComparison.OrdinalIgnoreCase))
            {
                queries.Add(new SearchQuery(narrowed, maxResults));
            }
        }

        return queries;
    }

    /// <summary>Tách từ theo thứ tự xuất hiện, giữ '#' và '+' để "C#"/"C++" không bị cắt.</summary>
    public static IReadOnlyList<string> OrderedTerms(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var terms = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var buffer = new StringBuilder();

        void Flush()
        {
            if (buffer.Length == 0) return;

            string word = buffer.ToString();
            buffer.Clear();

            if (word.Length > 1 && seen.Add(word)) terms.Add(word);
        }

        foreach (char c in text.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c) || c == '#' || c == '+') buffer.Append(c);
            else Flush();
        }

        Flush();
        return terms;
    }
}
