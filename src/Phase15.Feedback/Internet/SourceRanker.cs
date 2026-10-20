using System.Globalization;
using Ai.Phase15.Memory;

namespace Ai.Phase15.Internet;

/// <summary>Xếp hạng lại kết quả theo tiêu chí của chính agent, không tin thứ tự của máy tìm kiếm.</summary>
public interface ISourceRanker
{
    IReadOnlyList<RankedResult> Rank(string question, IEnumerable<SearchResult> results, DeduplicationReport? dedup = null);
}

/// <summary>
/// Chấm điểm nguồn.
///
/// VÌ SAO PHẢI XẾP HẠNG LẠI? Máy tìm kiếm xếp theo mục tiêu của NÓ — trong đó
/// có cả quảng cáo và tối ưu SEO. Agent cần thứ tự theo mục tiêu của MÌNH:
/// trả lời đúng câu hỏi đang hỏi.
///
/// CÔNG THỨC — bốn tín hiệu, cùng dạng tổ hợp có trọng số như Ranker của
/// Phase 10 (score = 0.55·relevance + 0.20·importance + ...):
///
///     score = 0.45·relevance     tiêu đề + đoạn trích khớp câu hỏi bao nhiêu
///           + 0.20·authority     tên miền có đáng tin không
///           + 0.20·agreement     bao nhiêu máy tìm kiếm độc lập nhắc tới
///           + 0.15·position      máy tìm kiếm xếp nó thứ mấy
///
/// relevance nặng nhất vì một nguồn uy tín mà nói về chuyện khác thì vô dụng.
/// position nhẹ nhất vì nó chính là ý kiến của máy tìm kiếm — thứ ta đang cố
/// không phụ thuộc vào.
///
/// TRỌNG SỐ TIÊU ĐỀ: giống hệt lý do ở SearchTool của Phase 11 — tiêu đề nói
/// trang NÓI VỀ cái gì, thân bài chỉ nói nó CÓ NHẮC tới cái gì.
///
/// CẢNH BÁO QUAN TRỌNG: điểm này là mức ĐÁNG XEM TRƯỚC, không phải mức
/// ĐÁNG TIN. Một trang .edu vẫn có thể sai. Phase 14 mới là chỗ kiểm chứng
/// nội dung; ở Phase 12 ta chỉ xếp thứ tự đọc.
/// </summary>
public sealed class SourceRanker : ISourceRanker
{
    /// <summary>
    /// Bảng uy tín theo tên miền — CẤU HÌNH ĐƯỢC, và mặc định phải khiêm tốn.
    ///
    /// Nhúng cứng "tôi tin trang X" vào mã nguồn là một dạng thiên lệch khó
    /// thấy. Bảng này chỉ ưu tiên nhẹ vài loại tên miền có quy trình biên
    /// tập/bình duyệt, và trừ điểm nhẹ những nơi nội dung do người dùng tự
    /// đăng. Mọi thứ khác là 0.5 — trung tính.
    /// </summary>
    public IReadOnlyDictionary<string, double> DomainAuthority { get; init; } =
        new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase)
        {
            [".gov"] = 0.90,
            [".edu"] = 0.85,
            [".ac.uk"] = 0.85,
            ["arxiv.org"] = 0.80,
            ["wikipedia.org"] = 0.75,
            ["learn.microsoft.com"] = 0.80,
            ["docs.python.org"] = 0.80,
            ["developer.mozilla.org"] = 0.80,
            ["stackoverflow.com"] = 0.60,
            ["github.com"] = 0.60,
            ["medium.com"] = 0.40,
            ["blogspot.com"] = 0.35,
            ["quora.com"] = 0.30,
        };

    public double DefaultAuthority { get; init; } = 0.50;

    /// <summary>HTTPS được cộng thêm chút — không phải vì nội dung đúng hơn, mà vì nội dung không bị sửa trên đường truyền.</summary>
    public double HttpsBonus { get; init; } = 0.05;

    public IReadOnlyList<RankedResult> Rank(string question, IEnumerable<SearchResult> results, DeduplicationReport? dedup = null)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(results);

        var queryWords = Keywords.Extract(question);

        return [.. results
            .Select(r => Score(r, queryWords, dedup?.AgreementFor(r.Url) ?? 1))
            .OrderByDescending(r => r.Score)
            .ThenBy(r => r.Result.Url.AbsoluteUri, StringComparer.Ordinal)];
    }

    private RankedResult Score(SearchResult result, IReadOnlySet<string> queryWords, int agreement)
    {
        // --- 1. relevance ---
        double titleMatch = Keywords.Relevance(queryWords, result.Title.Keywords());
        double snippetMatch = Keywords.Relevance(queryWords, result.Snippet.Keywords());
        double relevance = Math.Min(1.0, 0.6 * titleMatch + 0.4 * snippetMatch + 0.3 * titleMatch * snippetMatch);

        // --- 2. authority ---
        double authority = Math.Min(1.0, AuthorityOf(result.Url) +
            (result.Url.Scheme.Equals("https", StringComparison.OrdinalIgnoreCase) ? HttpsBonus : 0));

        // --- 3. agreement: 1 nguồn -> 0, 2 -> 0.5, 3 -> 0.75, ... (bão hoà, không tuyến tính) ---
        double agreementScore = 1.0 - 1.0 / Math.Max(1, agreement);

        // --- 4. position: 1/rank, giảm nhanh rồi phẳng ra ---
        double position = 1.0 / Math.Max(1, result.Rank);

        double score = 0.45 * relevance + 0.20 * authority + 0.20 * agreementScore + 0.15 * position;

        // Nội dung có dấu hiệu tiêm chỉ thị -> trừ mạnh. Một trang đang cố ra
        // lệnh cho agent thì không đáng được đọc trước.
        if (result.Title.LooksLikeInjection || result.Snippet.LooksLikeInjection)
        {
            score *= 0.5;
        }

        string explanation = string.Format(CultureInfo.InvariantCulture,
            "khớp {0:P0} (tiêu đề {1:P0}), uy tín {2:F2}, {3} nguồn, hạng {4}",
            relevance, titleMatch, authority, agreement, result.Rank);

        if (result.Title.LooksLikeInjection || result.Snippet.LooksLikeInjection)
        {
            explanation += ", BỊ TRỪ ĐIỂM vì có dấu hiệu tiêm chỉ thị";
        }

        return new RankedResult(result, score, explanation);
    }

    /// <summary>
    /// Uy tín của tên miền. Khớp theo đuôi để ".edu" bao được cả
    /// "cs.stanford.edu", nhưng không để "notevil.edu.attacker.com" lọt qua.
    /// </summary>
    public double AuthorityOf(Uri url)
    {
        ArgumentNullException.ThrowIfNull(url);

        string host = url.Host.ToLowerInvariant();

        double best = DefaultAuthority;
        bool found = false;

        foreach (var (pattern, value) in DomainAuthority)
        {
            bool matches = pattern.StartsWith('.')
                ? host.EndsWith(pattern, StringComparison.Ordinal)
                : host.Equals(pattern, StringComparison.Ordinal) ||
                  host.EndsWith("." + pattern, StringComparison.Ordinal);

            if (!matches) continue;

            // Nhiều mẫu cùng khớp -> lấy mẫu THẤP nhất. "abc.medium.com" vừa
            // khớp medium.com (0.40); nếu có mẫu nào khác cao hơn thì vẫn nên
            // thận trọng, vì rủi ro nằm ở mẫu thấp.
            best = found ? Math.Min(best, value) : value;
            found = true;
        }

        return best;
    }
}
