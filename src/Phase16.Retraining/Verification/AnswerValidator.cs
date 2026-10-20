using Ai.Phase16.Grounding;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Verification;

/// <summary>Một câu trong câu trả lời mà không có căn cứ nào đỡ.</summary>
public sealed record UnsupportedClaim(string Text, double Coverage, string Reason)
{
    public override string ToString() => $"\"{Text}\" (phủ {Coverage:P0}) — {Reason}";
}

/// <summary>
/// Kết quả kiểm tra cuối. <see cref="IsValid"/> false nghĩa là câu trả lời
/// KHÔNG được phát ra như hiện tại.
/// </summary>
public sealed record ValidationReport
{
    public required bool IsValid { get; init; }

    public IReadOnlyList<UnsupportedClaim> UnsupportedClaims { get; init; } = [];

    public IReadOnlyList<string> Problems { get; init; } = [];

    public int SentencesChecked { get; init; }

    public int SentencesGrounded { get; init; }

    public double GroundedRatio => SentencesChecked == 0 ? 1 : (double)SentencesGrounded / SentencesChecked;

    public override string ToString() =>
        IsValid
            ? $"hợp lệ ({SentencesGrounded}/{SentencesChecked} câu có căn cứ)"
            : $"KHÔNG hợp lệ: {string.Join("; ", Problems)}";
}

public interface IAnswerValidator
{
    ValidationReport Validate(string answerText, IReadOnlyList<Claim> claims, IReadOnlyList<Evidence> citedEvidence);
}

/// <summary>
/// KIỂM TRA CUỐI — chốt chặn trước khi câu trả lời được phát ra.
///
/// VÌ SAO CẦN, KHI CÂU TRẢ LỜI ĐƯỢC DỰNG TỪ CHÍNH CĂN CỨ?
///
/// Vì "dựng từ căn cứ" là tính chất của <see cref="AnswerGenerator"/> hôm nay,
/// không phải bảo đảm của hệ thống. Bộ sinh câu trả lời sẽ được sửa, sẽ được
/// thay, và ở Phase 16 có thể được thay bằng một model đã train. Lúc đó tính
/// chất kia biến mất lặng lẽ, không có gì báo.
///
/// Nên validator được viết ĐỘC LẬP với bộ sinh: nó chỉ nhận văn bản câu trả
/// lời cùng danh sách căn cứ, rồi tự kiểm. Nó không tin bộ sinh, và nhờ vậy
/// bắt được cả trường hợp bộ sinh bịa — kể cả một bộ sinh viết sau này.
///
/// CÁCH KIỂM "KHÔNG CÓ CĂN CỨ": tách câu trả lời thành câu, với mỗi câu lấy
/// từ khoá, rồi xem từ khoá đó có nằm trong hợp các từ khoá của căn cứ đã dẫn
/// hay không.
///
///     coverage = |từ khoá của câu ∩ từ khoá mọi căn cứ| / |từ khoá của câu|
///
/// Thô, và có giới hạn thật: cách này bắt được câu NÓI VỀ chuyện không có
/// trong căn cứ, nhưng KHÔNG bắt được câu dùng đúng những từ đó mà ghép sai
/// nghĩa ("SSRF KHÔNG phải lỗ hổng" dùng y hệt bộ từ của câu đúng). Bắt được
/// loại sau cần hiểu ngữ nghĩa, không phải đếm từ. Phần nào bắt được thì
/// <see cref="ContradictionDetector"/> lo qua dấu hiệu phủ định.
/// </summary>
public sealed class AnswerValidator : IAnswerValidator
{
    /// <summary>
    /// Câu phải có ít nhất bao nhiêu phần từ khoá nằm trong căn cứ.
    ///
    /// 0.6 chứ không phải 1.0, vì câu trả lời có từ nối và từ dẫn của riêng nó
    /// ("theo nguồn", "ngoài ra") — những từ không thể có trong căn cứ.
    /// </summary>
    public double MinimumCoverage { get; init; } = 0.6;

    /// <summary>Câu ngắn hơn số từ khoá này thì bỏ qua (tiêu đề, dòng dẫn).</summary>
    public int MinimumKeywordsToCheck { get; init; } = 3;

    /// <summary>
    /// TỪ VỰNG KHUNG ĐỠ — những từ agent được phép tự dùng.
    ///
    /// Vì sao phải có danh sách này? Vì câu trả lời không chỉ gồm nội dung căn
    /// cứ; nó còn có phần khung do agent viết:
    ///
    ///     "Theo các căn cứ tôi có (độ tự tin 84%):"
    ///     "↳ kho kiến thức nội bộ: "SSRF"  [tin cậy 0.95]"
    ///     "LƯU Ý: toàn bộ căn cứ trên đến từ nguồn ngoài CHƯA KIỂM CHỨNG."
    ///
    /// Không có từ nào trong mấy câu đó nằm trong căn cứ — đúng như vậy, vì
    /// chúng KHÔNG khẳng định gì về thế giới. Chúng là câu nói VỀ CHÍNH CÂU
    /// TRẢ LỜI. Nếu không tách chúng ra thì validator báo mọi câu trả lời hợp
    /// lệ là bịa, đường ống hạ hết xuống "không đủ căn cứ", và agent im lặng
    /// hoàn toàn — đúng lỗi đã xảy ra khi làm Phase 14.
    ///
    /// Danh sách này phải ĐƯỢC KHAI BÁO TƯỜNG MINH và giữ NHỎ. Nó là bề mặt
    /// tấn công: mỗi từ thêm vào là một từ mà một bộ sinh bịa được dùng miễn
    /// phí. Nên ở đây chỉ có từ nối, từ chỉ siêu dữ liệu, và tên các kênh —
    /// không có từ nào mang nội dung sự kiện.
    /// </summary>
    public IReadOnlySet<string> ScaffoldVocabulary { get; init; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            // nói về chính câu trả lời
            "theo", "căn", "cứ", "tôi", "độ", "tự", "tin", "cậy", "mức",
            "sự", "thật", "suy", "ra", "kênh", "cùng", "xác", "nhận",
            "lưu", "ý", "toàn", "bộ", "trên", "đến", "từ", "nguồn",
            "trả", "lời", "câu", "hỏi", "này", "nên", "đủ", "quá", "yếu",
            "chắc", "nhưng", "còn", "khác", "nữa", "để", "kết", "luận",
            "mẩu", "liên", "quan", "chúng", "đã", "đưa", "gồm",
            // tên kênh và cách trích dẫn
            "kho", "kiến", "thức", "nội", "bạn", "bảo", "ghi", "nhớ",
            "vừa", "nói", "lượt", "web", "internet", "ngoài", "chưa",
            "kiểm", "chứng", "hãy", "đối", "chiếu", "trước", "dựa", "vào",
            // trình bày mâu thuẫn
            "phía", "dấu", "hiệu", "mâu", "thuẫn", "đáng", "chọn", "bên",
            "cả", "hai", "ngược", "nhau", "đây", "tranh", "chấp",
            "lại", "kém", "hơn", "biết", "nó", "vì",
        };

    public ValidationReport Validate(string answerText, IReadOnlyList<Claim> claims,
        IReadOnlyList<Evidence> citedEvidence)
    {
        ArgumentNullException.ThrowIfNull(answerText);
        ArgumentNullException.ThrowIfNull(claims);
        ArgumentNullException.ThrowIfNull(citedEvidence);

        var problems = new List<string>();
        var unsupported = new List<UnsupportedClaim>();

        // --- Kiểm 1: mọi khẳng định phải dẫn được căn cứ ---
        foreach (var claim in claims)
        {
            if (claim.EvidenceIds.Count == 0)
            {
                problems.Add($"khẳng định '{claim.Id}' không dẫn căn cứ nào");
            }
            else if (!claim.IsSupported)
            {
                problems.Add($"khẳng định '{claim.Id}' chỉ được đỡ {claim.Support:F2}, " +
                             $"dưới ngưỡng {Claim.MinimumSupport:F2}");
            }
        }

        // --- Kiểm 2: id căn cứ được dẫn phải thực sự tồn tại ---
        var availableIds = citedEvidence.Select(e => e.Id).ToHashSet(StringComparer.Ordinal);

        foreach (var claim in claims)
        {
            foreach (string id in claim.EvidenceIds.Where(id => !availableIds.Contains(id)))
            {
                problems.Add($"khẳng định '{claim.Id}' dẫn căn cứ '{id}' không có trong tập đã dẫn");
            }
        }

        // --- Kiểm 3: từng câu trong văn bản phải có căn cứ ---
        // Từ được phép xuất hiện = từ của căn cứ + từ khung đỡ của agent.
        var allowedWords = new HashSet<string>(ScaffoldVocabulary, StringComparer.OrdinalIgnoreCase);

        foreach (var evidence in citedEvidence)
        {
            allowedWords.UnionWith(evidence.Keywords());
            allowedWords.UnionWith(Keywords.Extract(evidence.Source.Name));
        }

        int checkedCount = 0;
        int groundedCount = 0;

        foreach (string sentence in SplitSentences(answerText))
        {
            var words = Keywords.Extract(sentence);
            if (words.Count < MinimumKeywordsToCheck) continue;

            checkedCount++;

            double coverage = Keywords.Relevance(words, allowedWords);

            if (coverage >= MinimumCoverage)
            {
                groundedCount++;
                continue;
            }

            unsupported.Add(new UnsupportedClaim(sentence.Trim(), coverage,
                $"chỉ {coverage:P0} từ khoá của câu tìm thấy trong căn cứ đã dẫn"));
        }

        if (unsupported.Count > 0)
        {
            problems.Add($"{unsupported.Count} câu không truy được về căn cứ nào");
        }

        return new ValidationReport
        {
            IsValid = problems.Count == 0,
            UnsupportedClaims = unsupported,
            Problems = problems,
            SentencesChecked = checkedCount,
            SentencesGrounded = groundedCount,
        };
    }

    /// <summary>
    /// Tách văn bản thành câu.
    ///
    /// Tách cả theo dấu chấm và theo dòng, vì câu trả lời của agent là văn bản
    /// có gạch đầu dòng chứ không phải đoạn văn liền mạch.
    ///
    /// BỎ URL TRƯỚC KHI TÁCH — và đây là bản sửa cho một lỗi thật.
    ///
    /// URL chứa dấu chấm, nên tách theo dấu chấm sẽ băm nó thành những mảnh vô
    /// nghĩa. Mảnh "com/@someone/bitcoin-price-today" được đem đi kiểm như một
    /// câu khẳng định, và tất nhiên các từ trong đó (com, someone, price,
    /// today) không có trong nội dung căn cứ — nên validator báo câu trả lời
    /// là bịa, đường ống hạ kết luận xuống "không đủ căn cứ", và agent im lặng
    /// về một thông tin nó thực sự có.
    ///
    /// URL là DẤU CHỈ NGUỒN, không phải lời khẳng định về thế giới. Tên miền
    /// và slug của nó không cần — và không nên — xuất hiện trong nội dung căn
    /// cứ. Nên chúng bị bỏ ra khỏi phần văn bản đem đi kiểm.
    /// </summary>
    public static IReadOnlyList<string> SplitSentences(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        string withoutUrls = UrlPattern.Replace(text, " ");

        return [.. withoutUrls
            .Split(['.', '!', '?', '\n', ';'], StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim(' ', '\t', '•', '-', '*'))
            .Where(s => s.Length > 0)];
    }

    private static readonly System.Text.RegularExpressions.Regex UrlPattern =
        new(@"\b[a-z][a-z0-9+.-]*://\S+", System.Text.RegularExpressions.RegexOptions.IgnoreCase
            | System.Text.RegularExpressions.RegexOptions.CultureInvariant);
}
