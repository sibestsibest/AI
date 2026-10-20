using System.Globalization;
using System.Text;

namespace Ai.Phase10.Memory;

/// <summary>
/// Rút từ khoá để tìm kiếm.
///
/// Cách làm đơn giản nhất còn dùng được: hạ chữ thường, bỏ dấu câu, bỏ từ
/// dừng (stop word). Hệ thống thật sẽ dùng embedding của Phase 5 để bắt
/// được cả "xe hơi" ~ "ô tô" — ở đây so khớp chuỗi nên hai từ đó là khác
/// nhau hoàn toàn. Đổi lại: nhanh, tất định, và giải thích được vì sao một
/// mẩu ký ức được lấy ra.
/// </summary>
public static class Keywords
{
    private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        // tiếng Anh
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
        "of", "to", "in", "on", "at", "for", "with", "and", "or", "but",
        "i", "you", "he", "she", "it", "we", "they", "this", "that",
        "do", "does", "did", "what", "which", "who", "how",
        // tiếng Việt
        "là", "và", "của", "có", "cho", "với", "thì", "mà", "ở", "các",
        "một", "những", "được", "gì", "nào", "không", "về",
        // từ đệm — người dùng gõ rất nhiều nhưng không mang thông tin nào
        "ok", "okay", "yes", "yeah", "nope", "hmm", "haha", "hehe",
        "thanks", "thank", "please", "sure", "fine", "cool",
        "ừ", "ờ", "vâng", "dạ", "nhé", "nha", "thôi", "vậy",
    };

    /// <summary>Tách văn bản thành tập từ khoá đã chuẩn hoá.</summary>
    public static HashSet<string> Extract(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var buffer = new StringBuilder();

        void Flush()
        {
            if (buffer.Length == 0) return;

            string word = buffer.ToString();
            buffer.Clear();

            if (word.Length > 1 && !StopWords.Contains(word))
            {
                words.Add(word);
            }
        }

        foreach (char c in text.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c) || c == '#' || c == '+')
            {
                buffer.Append(c);   // giữ '#' và '+' để "C#" và "C++" không bị cắt
            }
            else
            {
                Flush();
            }
        }

        Flush();
        return words;
    }

    /// <summary>
    /// Tỉ lệ từ khoá của câu hỏi tìm thấy trong ký ức:
    ///
    ///     relevance = |query ∩ fact| / |query|
    ///
    /// Ví dụ: hỏi "Gundam" (1 từ), ký ức "User likes Gundam" chứa nó
    /// -> relevance = 1/1 = 1.0. Hỏi "Gundam Azure" (2 từ) mà ký ức chỉ có
    /// một -> 0.5.
    ///
    /// Chia cho kích thước CÂU HỎI chứ không phải ký ức, vì một ký ức dài
    /// không nên bị phạt chỉ vì nó dài.
    /// </summary>
    public static double Relevance(IReadOnlySet<string> query, IReadOnlySet<string> fact)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(fact);

        if (query.Count == 0) return 0;

        int matches = query.Count(fact.Contains);
        return (double)matches / query.Count;
    }

    /// <summary>Chuẩn hoá nội dung để phát hiện trùng lặp.</summary>
    public static string Canonical(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var words = Extract(text).OrderBy(w => w, StringComparer.Ordinal);
        return string.Join(' ', words).ToLower(CultureInfo.InvariantCulture);
    }
}
