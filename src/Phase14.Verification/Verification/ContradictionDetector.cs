using System.Globalization;
using Ai.Phase14.Grounding;
using Ai.Phase14.Internet;
using Ai.Phase14.Memory;

namespace Ai.Phase14.Verification;

public enum ContradictionKind
{
    /// <summary>Một bên phủ định điều bên kia khẳng định.</summary>
    Negation,

    /// <summary>Cùng một đại lượng nhưng hai con số khác nhau.</summary>
    NumericMismatch,

    /// <summary>Hai bên dùng cặp từ loại trừ nhau (tăng/giảm, an toàn/nguy hiểm).</summary>
    Opposite,
}

/// <summary>
/// Hai mẩu căn cứ nói ngược nhau.
///
/// <see cref="Severity"/> quyết định agent làm gì: mâu thuẫn nhẹ thì chỉ hạ độ
/// tự tin, mâu thuẫn nặng thì phải NGỪNG trả lời dứt khoát và trình bày cả hai
/// phía. Chọn một phía rồi im lặng về phía kia là hành vi tệ nhất có thể —
/// người đọc mất luôn cơ hội biết rằng có tranh chấp.
/// </summary>
public sealed record Contradiction
{
    public required Evidence Left { get; init; }

    public required Evidence Right { get; init; }

    public required ContradictionKind Kind { get; init; }

    public required string Explanation { get; init; }

    /// <summary>0..1. Càng cao khi hai bên càng cùng chủ đề và càng đáng tin.</summary>
    public double Severity { get; init; }

    /// <summary>Mâu thuẫn giữa hai nguồn ĐÃ KIỂM là chuyện nghiêm trọng hơn nhiều.</summary>
    public bool BetweenVerifiedSources =>
        Left.Trust == TrustLevel.Verified && Right.Trust == TrustLevel.Verified;

    public override string ToString() =>
        $"{Kind} ({Severity:F2}): {Left.Source.Name} ⟷ {Right.Source.Name} — {Explanation}";
}

public interface IContradictionDetector
{
    IReadOnlyList<Contradiction> Detect(IReadOnlyList<Evidence> evidence);
}

/// <summary>
/// PHÁT HIỆN MÂU THUẪN bằng dấu hiệu ký hiệu — không dùng mạng neural.
///
/// BA DẤU HIỆU:
///
///   1. Phủ định  — cùng chủ đề, một bên có từ phủ định, bên kia không
///   2. Lệch số   — cùng chủ đề, hai con số khác nhau
///   3. Đối nghịch— cùng chủ đề, hai bên dùng một cặp từ loại trừ nhau
///
/// ĐIỀU KIỆN TIÊN QUYẾT CHO CẢ BA: hai mẩu phải THỰC SỰ CÙNG CHỦ ĐỀ.
///
/// Đây là chỗ dễ sai nhất của toàn bộ Phase 14, và sai theo hướng tệ. Bỏ điều
/// kiện cùng chủ đề đi thì:
///
///     "Softmax biến logits thành phân bố xác suất"     (không có 'không')
///     "Overfitting là khi model KHÔNG học được quy luật" (có 'không')
///
/// bị báo là mâu thuẫn, chỉ vì một câu chứa từ phủ định và câu kia không. Hai
/// câu đó chẳng liên quan gì đến nhau. Báo động giả kiểu này còn tệ hơn không
/// phát hiện gì: nó làm agent từ chối trả lời những câu nó trả lời được, và
/// làm người dùng học cách bỏ qua cảnh báo.
///
/// Nên ngưỡng chủ đề (<see cref="MinimumTopicOverlap"/>) đặt khá cao, và
/// phương châm ở đây là THẬN TRỌNG KHI BÁO: thà bỏ sót một mâu thuẫn thật hơn
/// là dựng ra một mâu thuẫn không có.
/// </summary>
public sealed class ContradictionDetector : IContradictionDetector
{
    /// <summary>Dưới mức trùng chủ đề này thì KHÔNG xét mâu thuẫn.</summary>
    public double MinimumTopicOverlap { get; init; } = 0.45;

    /// <summary>Lệch số dưới tỉ lệ này coi như làm tròn khác nhau, không phải mâu thuẫn.</summary>
    public double NumericTolerance { get; init; } = 0.05;

    public IReadOnlyList<Contradiction> Detect(IReadOnlyList<Evidence> evidence)
    {
        ArgumentNullException.ThrowIfNull(evidence);

        var found = new List<Contradiction>();

        for (int i = 0; i < evidence.Count; i++)
        {
            for (int j = i + 1; j < evidence.Count; j++)
            {
                var contradiction = Compare(evidence[i], evidence[j]);
                if (contradiction is not null) found.Add(contradiction);
            }
        }

        return [.. found.OrderByDescending(c => c.Severity)];
    }

    /// <summary>So hai mẩu. Trả về null nếu không thấy mâu thuẫn nào.</summary>
    public Contradiction? Compare(Evidence left, Evidence right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        var leftWords = left.Keywords();
        var rightWords = right.Keywords();

        // --- Điều kiện tiên quyết: phải cùng chủ đề ---
        double overlap = ResultDeduplicator.Jaccard(leftWords, rightWords);
        if (overlap < MinimumTopicOverlap) return null;

        // --- Dấu hiệu 1: phủ định lệch nhau ---
        //
        // Xét trên văn bản THÔ, không xét trên tập từ khoá: "không" là stop
        // word của Phase 10 nên nó đã bị lọc mất khỏi Keywords. Xem
        // <see cref="Polarity"/> để biết vì sao đây là một cái bẫy thật.
        if (Polarity.Differ(left.Content, right.Content))
        {
            return Build(left, right, ContradictionKind.Negation, overlap,
                $"trùng chủ đề {overlap:P0} nhưng một bên phủ định, bên kia không");
        }

        // --- Dấu hiệu 2: lệch số ---
        var numericClash = FindNumericClash(left.Content, right.Content);
        if (numericClash is not null)
        {
            var (a, b) = numericClash.Value;

            return Build(left, right, ContradictionKind.NumericMismatch, overlap,
                string.Format(CultureInfo.InvariantCulture,
                    "trùng chủ đề {0:P0} nhưng nêu hai con số khác nhau: {1} và {2}", overlap, a, b));
        }

        // --- Dấu hiệu 3: cặp từ đối nghịch ---
        // Cũng xét trên văn bản thô, vì "cho phép" chứa "cho" — một stop word.
        var opposite = Polarity.FindOpposite(left.Content, right.Content);

        if (opposite is not null)
        {
            var (a, b) = opposite.Value;

            return Build(left, right, ContradictionKind.Opposite, overlap,
                $"trùng chủ đề {overlap:P0} nhưng dùng cặp từ loại trừ nhau: '{a}' và '{b}'");
        }

        return null;
    }

    /// <summary>
    /// Mức nghiêm trọng = mức trùng chủ đề × độ tin cậy của bên YẾU HƠN.
    ///
    /// Vì sao lấy bên yếu hơn? Vì một mẩu rất đáng tin mâu thuẫn với một mẩu
    /// rác thì đó gần như không phải tranh chấp — chỉ cần bỏ mẩu rác đi. Tranh
    /// chấp thật sự là khi CẢ HAI bên đều đáng tin, và lúc đó min() mới cao.
    /// </summary>
    private static Contradiction Build(Evidence left, Evidence right, ContradictionKind kind,
        double overlap, string explanation)
    {
        double weakest = Math.Min(left.Reliability, right.Reliability);

        return new Contradiction
        {
            Left = left,
            Right = right,
            Kind = kind,
            Explanation = explanation,
            Severity = Math.Clamp(overlap * weakest, 0, 1),
        };
    }

    /// <summary>
    /// Tìm một cặp số xung đột giữa hai đoạn văn.
    ///
    /// Chỉ báo xung đột khi một bên có số mà bên kia KHÔNG có số nào gần nó.
    /// Cách này thô — nó không biết con số đang đo cái gì — nên nó bị khoá sau
    /// điều kiện cùng chủ đề. Không có điều kiện đó thì mọi trang có số khác
    /// nhau đều thành mâu thuẫn.
    /// </summary>
    private (double A, double B)? FindNumericClash(string left, string right)
    {
        var leftNumbers = ExtractNumbers(left);
        var rightNumbers = ExtractNumbers(right);

        if (leftNumbers.Count == 0 || rightNumbers.Count == 0) return null;

        foreach (double a in leftNumbers)
        {
            bool matched = rightNumbers.Any(b => AreClose(a, b));
            if (matched) continue;

            // a không có bản tương ứng bên phải -> lấy số gần nhất làm đối chứng.
            double closest = rightNumbers.MinBy(b => Math.Abs(b - a));
            return (a, closest);
        }

        return null;
    }

    private bool AreClose(double a, double b)
    {
        if (a.Equals(b)) return true;

        double scale = Math.Max(Math.Abs(a), Math.Abs(b));
        return scale > 0 && Math.Abs(a - b) / scale <= NumericTolerance;
    }

    /// <summary>Rút các số xuất hiện trong văn bản.</summary>
    public static List<double> ExtractNumbers(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var numbers = new List<double>();
        int i = 0;

        while (i < text.Length)
        {
            if (!char.IsDigit(text[i]))
            {
                i++;
                continue;
            }

            int start = i;
            while (i < text.Length && (char.IsDigit(text[i]) || text[i] == '.' || text[i] == ',')) i++;

            // Bỏ dấu phân cách hàng nghìn, giữ dấu thập phân cuối cùng nếu có.
            string raw = text[start..i].TrimEnd('.', ',').Replace(",", string.Empty);

            if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
            {
                numbers.Add(value);
            }
        }

        return numbers;
    }
}
