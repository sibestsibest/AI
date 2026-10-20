using System.Globalization;

namespace Ai.Phase12.Reasoning;

public sealed class ReasoningException(string message) : Exception(message);

/// <summary>Một bước trong chuỗi suy luận — để giải thích cho người đọc.</summary>
public sealed record ReasoningStep(string Description);

/// <summary>Kết quả trả lời, kèm toàn bộ đường đi đã dùng để đến đó.</summary>
public sealed record Answer(
    double? Value,
    bool? Truth,
    IReadOnlyList<ReasoningStep> Steps,
    IReadOnlyList<Fact> UsedFacts)
{
    public bool IsComparison => Truth.HasValue;

    public string Text => Truth.HasValue
        ? Truth.Value ? "ĐÚNG" : "SAI"
        : Format(Value!.Value);

    public static string Format(double value) =>
        value == Math.Floor(value) && Math.Abs(value) < 1e15
            ? ((long)value).ToString(CultureInfo.InvariantCulture)
            : value.ToString("G10", CultureInfo.InvariantCulture);
}

/// <summary>
/// Bộ máy suy luận ký hiệu.
///
/// Quy trình đúng như đề bài:
///
///     Parse facts  -> Parser.ParseFact
///     Store facts  -> KnowledgeBase
///     Find relevant facts -> KnowledgeBase.FindRelevant
///     Apply rule   -> Evaluate (đệ quy theo cây biểu thức)
///     Calculate    -> phép toán số học
///     Return answer -> Answer, kèm đầy đủ các bước
///
/// ĐIỂM MẠNH so với neural network:
///   • Chính xác tuyệt đối — 10 + 20 luôn ra 30, không phải 29.97
///   • Giải thích được từng bước
///   • Không cần huấn luyện
///   • Thêm một sự thật là biết ngay, không phải train lại
///
/// ĐIỂM YẾU:
///   • Chỉ làm được đúng những gì đã lập trình
///   • Không chịu được đầu vào mơ hồ hay sai chính tả
///   • Không học được từ ví dụ
///
/// Vì thế Phase 11 sẽ để một AI Controller quyết định khi nào dùng cái nào.
/// </summary>
public sealed class ReasoningEngine
{
    private const int MaxDepth = 64;

    public KnowledgeBase Knowledge { get; } = new();

    public Fact Learn(string statement) => Knowledge.Assert(statement);

    public void LearnAll(params string[] statements) => Knowledge.AssertAll(statements);

    /// <summary>
    /// Trả lời một câu hỏi. Chấp nhận cả biểu thức thuần ("A + B", "A > B")
    /// lẫn câu hỏi tiếng Việt/tiếng Anh ("C bằng bao nhiêu?", "what is C").
    /// </summary>
    public Answer Ask(string question)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(question);

        string normalized = QuestionNormalizer.Normalize(question);

        if (string.IsNullOrWhiteSpace(normalized))
        {
            throw new ReasoningException($"Không tìm thấy nội dung hỏi trong \"{question}\".");
        }

        var expression = Parser.ParseExpression(normalized);
        var steps = new List<ReasoningStep>();
        var used = Knowledge.FindRelevant(expression);

        foreach (var fact in used)
        {
            steps.Add(new ReasoningStep($"đã biết: {fact.Source}"));
        }

        if (expression is BinaryExpression { IsComparison: true } comparison)
        {
            double left = Evaluate(comparison.Left, steps, 0);
            double right = Evaluate(comparison.Right, steps, 0);

            bool truth = comparison.Op switch
            {
                Operator.Equal => Math.Abs(left - right) < 1e-9,
                Operator.Greater => left > right,
                Operator.Less => left < right,
                _ => throw new ReasoningException($"Phép so sánh không hỗ trợ: {comparison.Op}"),
            };

            string symbol = BinaryExpression.SymbolOf(comparison.Op);
            steps.Add(new ReasoningStep(
                $"so sánh: {Answer.Format(left)} {symbol} {Answer.Format(right)} -> {(truth ? "ĐÚNG" : "SAI")}"));

            return new Answer(null, truth, Distinct(steps), used);
        }

        double value = Evaluate(expression, steps, 0);
        steps.Add(new ReasoningStep($"kết quả: {expression} = {Answer.Format(value)}"));

        return new Answer(value, null, Distinct(steps), used);
    }

    /// <summary>
    /// Tính giá trị của một biểu thức, đi đệ quy xuống các biến mà nó phụ thuộc.
    ///
    /// Pseudocode:
    ///     nếu là số        -> trả về chính nó
    ///     nếu là biến      -> tra kho, rồi tính đệ quy định nghĩa của nó
    ///     nếu là phép toán -> tính hai vế rồi áp phép toán
    /// </summary>
    public double Evaluate(Expression expression, List<ReasoningStep>? steps = null, int depth = 0)
    {
        ArgumentNullException.ThrowIfNull(expression);

        if (depth > MaxDepth)
        {
            throw new ReasoningException(
                "Suy luận quá sâu — nhiều khả năng các sự thật tham chiếu vòng tròn (ví dụ A = B, B = A).");
        }

        switch (expression)
        {
            case NumberExpression number:
                return number.Value;

            case VariableExpression variable:
            {
                if (!Knowledge.TryGet(variable.Name, out var fact))
                {
                    throw new ReasoningException(
                        $"Chưa biết '{variable.Name}'. Hãy cho biết giá trị của nó trước.");
                }

                double value = Evaluate(fact.Value, steps, depth + 1);

                if (!fact.IsLiteral)
                {
                    steps?.Add(new ReasoningStep(
                        $"suy ra:   {variable.Name} = {fact.Value} = {Answer.Format(value)}"));
                }

                return value;
            }

            case BinaryExpression binary:
            {
                double left = Evaluate(binary.Left, steps, depth + 1);
                double right = Evaluate(binary.Right, steps, depth + 1);

                if (binary.Op == Operator.Divide && Math.Abs(right) < 1e-12)
                {
                    throw new ReasoningException($"Chia cho 0 trong \"{binary}\".");
                }

                return binary.Op switch
                {
                    Operator.Add => left + right,
                    Operator.Subtract => left - right,
                    Operator.Multiply => left * right,
                    Operator.Divide => left / right,
                    Operator.Equal => Math.Abs(left - right) < 1e-9 ? 1 : 0,
                    Operator.Greater => left > right ? 1 : 0,
                    Operator.Less => left < right ? 1 : 0,
                    _ => throw new ReasoningException($"Phép toán không hỗ trợ: {binary.Op}"),
                };
            }

            default:
                throw new ReasoningException($"Loại biểu thức không hỗ trợ: {expression.GetType().Name}");
        }
    }

    /// <summary>Giá trị của một biến, không cần dựng câu hỏi.</summary>
    public double ValueOf(string name) => Evaluate(new VariableExpression(name));

    /// <summary>
    /// Bỏ các bước trùng lặp khi trình bày.
    ///
    /// Một biến được nhiều biểu thức cùng dùng sẽ bị tính lại nhiều lần —
    /// ví dụ Subtotal xuất hiện cả trong Total lẫn trong Tax. Kết quả vẫn
    /// đúng, chỉ là lời giải thích bị lặp. (Nếu cần nhanh hơn thì phải nhớ
    /// lại kết quả đã tính — nhưng ở quy mô này chưa đáng đánh đổi độ rõ ràng.)
    /// </summary>
    private static List<ReasoningStep> Distinct(List<ReasoningStep> steps)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var result = new List<ReasoningStep>();

        foreach (var step in steps)
        {
            if (seen.Add(step.Description)) result.Add(step);
        }

        return result;
    }
}
