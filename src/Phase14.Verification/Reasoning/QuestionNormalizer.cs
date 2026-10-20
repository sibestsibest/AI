namespace Ai.Phase14.Reasoning;

/// <summary>
/// Gọt câu hỏi tự nhiên về biểu thức thuần.
///
///     "C bằng bao nhiêu?"   ->  "C"
///     "A có lớn hơn B không?" -> "A > B"
///     "what is A + B"       ->  "A + B"
///
/// Đây CỐ Ý là thứ thô sơ: chỉ so khớp mẫu chuỗi, không hiểu ngôn ngữ.
/// Nó cho thấy giới hạn thật của hệ thống ký hiệu — mọi cách diễn đạt mới
/// đều phải được thêm tay vào danh sách dưới đây.
///
/// Đây chính xác là khoảng trống mà mô hình ngôn ngữ lấp vào: LLM ánh xạ
/// muôn vàn cách nói về cùng một ý định. Phase 11 sẽ ghép hai thứ lại —
/// LLM hiểu câu hỏi, reasoning engine tính đáp án chính xác.
/// </summary>
public static class QuestionNormalizer
{
    /// <summary>Các cụm bị xoá thẳng (đã sắp dài trước ngắn để khớp cụm dài trước).</summary>
    private static readonly string[] NoisePhrases =
    [
        "bằng bao nhiêu", "là bao nhiêu", "có giá trị bao nhiêu", "bao nhiêu",
        "hãy tính", "tính giá trị", "tính xem", "tính",
        "cho biết", "giá trị của", "kết quả của",
        "what is the value of", "what is", "what's", "how much is", "how much",
        "calculate", "compute", "evaluate",
        "là gì", "là mấy",
        "không", "nhỉ", "vậy", "thì",
    ];

    /// <summary>Cụm so sánh viết bằng chữ -> ký hiệu.</summary>
    private static readonly (string Phrase, string Symbol)[] ComparisonPhrases =
    [
        ("có lớn hơn", ">"), ("lớn hơn hay bằng", ">"), ("lớn hơn", ">"),
        ("có nhỏ hơn", "<"), ("nhỏ hơn hay bằng", "<"), ("nhỏ hơn", "<"), ("bé hơn", "<"),
        ("có bằng", "="), ("bằng với", "="),
        ("is greater than", ">"), ("greater than", ">"),
        ("is less than", "<"), ("less than", "<"),
        ("is equal to", "="), ("equal to", "="), ("equals", "="),
        ("cộng", "+"), ("trừ", "-"), ("nhân", "*"), ("chia", "/"),
        ("plus", "+"), ("minus", "-"), ("times", "*"), ("divided by", "/"),
    ];

    public static string Normalize(string question)
    {
        ArgumentNullException.ThrowIfNull(question);

        string text = question.Trim().TrimEnd('?', '.', '!').Trim();

        // Đổi cụm so sánh thành ký hiệu TRƯỚC, vì có cụm chứa từ nhiễu
        // ("có lớn hơn ... không" — nếu xoá "không" trước thì vẫn ổn,
        // nhưng đổi trước cho chắc chắn).
        foreach (var (phrase, symbol) in ComparisonPhrases)
        {
            text = ReplaceWholeWords(text, phrase, $" {symbol} ");
        }

        foreach (string phrase in NoisePhrases)
        {
            text = ReplaceWholeWords(text, phrase, " ");
        }

        // Gom khoảng trắng thừa.
        return string.Join(' ', text.Split(' ', StringSplitOptions.RemoveEmptyEntries)).Trim();
    }

    /// <summary>
    /// Chỉ thay khi khớp NGUYÊN TỪ, để "không" trong câu không ăn mất chữ
    /// "không" nằm giữa một tên biến như "Khong_Gian".
    /// </summary>
    private static string ReplaceWholeWords(string text, string phrase, string replacement)
    {
        int index = 0;

        while (index <= text.Length - phrase.Length)
        {
            int found = text.IndexOf(phrase, index, StringComparison.OrdinalIgnoreCase);
            if (found < 0) break;

            bool leftOk = found == 0 || !IsWordChar(text[found - 1]);
            int after = found + phrase.Length;
            bool rightOk = after >= text.Length || !IsWordChar(text[after]);

            if (leftOk && rightOk)
            {
                text = text[..found] + replacement + text[after..];
                index = found + replacement.Length;
            }
            else
            {
                index = found + 1;
            }
        }

        return text;
    }

    private static bool IsWordChar(char c) => char.IsLetterOrDigit(c) || c == '_';
}
