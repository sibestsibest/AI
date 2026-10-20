namespace Ai.Phase14.Grounding;

/// <summary>
/// CHIỀU KHẲNG ĐỊNH / PHỦ ĐỊNH của một đoạn văn.
///
/// ====================================================================
/// VÌ SAO KHÔNG DÙNG ĐƯỢC Keywords.Extract Ở ĐÂY
/// ====================================================================
///
/// Đây là một cái bẫy rất gọn, và nó đã bẫy thật trong lúc làm Phase 14.
///
/// Từ phủ định mạnh nhất của tiếng Việt — "không" — nằm trong danh sách STOP
/// WORD của Phase 10. Nó nằm đó hoàn toàn hợp lý: với việc TÌM KIẾM thì
/// "không" xuất hiện ở khắp mọi nơi nên chẳng phân biệt được gì, bỏ đi là đúng.
///
/// Nhưng với việc PHÁT HIỆN MÂU THUẪN thì "không" là từ quan trọng NHẤT trong
/// câu. Hai câu dưới đây có tập từ khoá gần như y hệt nhau:
///
///     "SSRF là lỗ hổng cho phép máy chủ gửi request tới địa chỉ nội bộ"
///     "SSRF KHÔNG phải lỗ hổng cho phép máy chủ gửi request tới địa chỉ nội bộ"
///
/// Sau khi lọc stop word, chúng gần như KHÔNG KHÁC GÌ NHAU — nên hệ thống vừa
/// bỏ sót mâu thuẫn, vừa còn tệ hơn: bộ bỏ trùng coi chúng là cùng một thông
/// tin và GỘP chúng lại, làm mâu thuẫn biến mất hoàn toàn trước khi có ai kịp
/// nhìn thấy.
///
/// Bài học chung: cùng một văn bản cần được tách từ theo những cách khác nhau
/// cho những mục đích khác nhau. Lọc bỏ cái gì là quyết định thuộc về mục
/// đích, không thuộc về bộ tách từ.
///
/// Nên lớp này tự tách từ trên văn bản THÔ, không bỏ stop word nào.
/// </summary>
public static class Polarity
{
    /// <summary>Từ phủ định. Phần lớn chúng là stop word của Phase 10 — đó chính là vấn đề.</summary>
    private static readonly HashSet<string> Negations = new(StringComparer.OrdinalIgnoreCase)
    {
        // tiếng Việt
        "không", "chẳng", "chưa", "đừng", "khỏi", "chả", "đâu",
        // tiếng Anh
        "not", "no", "never", "cannot", "cant", "isnt", "arent", "doesnt",
        "dont", "wont", "didnt", "none", "neither", "nor",
    };

    /// <summary>
    /// Cặp từ loại trừ nhau. Bảng nhỏ và viết tay — cố ý.
    ///
    /// Suy ra trái nghĩa một cách tự động (kiểu "cứ thêm 'không' vào") tạo ra
    /// rất nhiều cặp vô nghĩa. Danh sách viết tay thì ít, nhưng mỗi cặp trong
    /// đó đều đúng, và đọc là biết ngay hệ thống phát hiện được gì.
    ///
    /// Để ở đây cùng với phủ định, vì cả hai trả lời CÙNG MỘT câu hỏi: hai
    /// đoạn văn này có đang nói ngược chiều nhau hay không. Và cả hai đều cần
    /// tách từ trên văn bản thô — "cho phép" chứa "cho", vốn là stop word.
    /// </summary>
    public static readonly (string A, string B)[] OppositePairs =
    [
        ("tăng", "giảm"),
        ("an", "nguy"),                  // an toàn / nguy hiểm
        ("đúng", "sai"),
        ("nhanh", "chậm"),
        ("cao", "thấp"),
        ("nhiều", "ít"),
        ("bật", "tắt"),
        ("phép", "chặn"),                // cho phép / chặn
        ("increase", "decrease"),
        ("safe", "unsafe"),
        ("true", "false"),
        ("faster", "slower"),
        ("enable", "disable"),
        ("allow", "block"),
    ];

    /// <summary>Đoạn văn này có chứa từ phủ định nào không.</summary>
    public static bool ContainsNegation(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        return Tokenize(text).Any(Negations.Contains);
    }

    /// <summary>Hai đoạn văn có chiều khẳng định/phủ định NGƯỢC nhau.</summary>
    public static bool Differ(string left, string right) =>
        ContainsNegation(left) != ContainsNegation(right);

    /// <summary>
    /// Tìm cặp từ đối nghịch mà hai đoạn văn dùng ngược nhau: một bên chỉ có
    /// A, bên kia chỉ có B. Trả về null nếu không có.
    /// </summary>
    public static (string A, string B)? FindOpposite(string left, string right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        var leftWords = Tokenize(left).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var rightWords = Tokenize(right).ToHashSet(StringComparer.OrdinalIgnoreCase);

        foreach (var (a, b) in OppositePairs)
        {
            bool leftA = leftWords.Contains(a), leftB = leftWords.Contains(b);
            bool rightA = rightWords.Contains(a), rightB = rightWords.Contains(b);

            if ((leftA && !leftB && rightB && !rightA) ||
                (leftB && !leftA && rightA && !rightB))
            {
                return (a, b);
            }
        }

        return null;
    }

    /// <summary>
    /// Hai đoạn văn nói NGƯỢC CHIỀU nhau, dù theo phủ định hay theo cặp từ
    /// đối nghịch.
    ///
    /// Bộ bỏ trùng dùng đúng hàm này để KHÔNG BAO GIỜ gộp hai mẩu như vậy —
    /// gộp chúng là làm mâu thuẫn biến mất trước khi có ai kịp thấy.
    /// </summary>
    public static bool Conflict(string left, string right) =>
        Differ(left, right) || FindOpposite(left, right) is not null;

    /// <summary>
    /// Tách từ trên văn bản thô — KHÔNG bỏ stop word, KHÔNG bỏ từ một ký tự.
    ///
    /// Dấu nháy bị bỏ để "doesn't" thành "doesnt", khớp được với danh sách trên.
    /// </summary>
    public static IEnumerable<string> Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var buffer = new System.Text.StringBuilder();

        foreach (char c in text.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c))
            {
                buffer.Append(c);
            }
            else if (c is '\'' or '’')
            {
                // bỏ dấu nháy, không cắt từ
            }
            else if (buffer.Length > 0)
            {
                yield return buffer.ToString();
                buffer.Clear();
            }
        }

        if (buffer.Length > 0) yield return buffer.ToString();
    }
}
