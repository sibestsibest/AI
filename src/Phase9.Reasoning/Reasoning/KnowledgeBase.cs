namespace Ai.Phase9.Reasoning;

/// <summary>
/// Một sự thật đã biết: tên biến và định nghĩa của nó.
/// <paramref name="Source"/> giữ lại câu gốc để giải thích cho người đọc.
/// </summary>
public sealed record Fact(string Name, Expression Value, string Source)
{
    /// <summary>Sự thật này có phụ thuộc biến khác không, hay là hằng số?</summary>
    public bool IsLiteral => Value is NumberExpression;

    public override string ToString() => $"{Name} = {Value}";
}

/// <summary>
/// Kho sự thật — bộ nhớ làm việc của engine.
///
/// Khác gì bảng embedding ở Phase 5? Ở đây mỗi mục là một khẳng định RỜI RẠC,
/// chính xác, ghi vào là dùng được ngay. Embedding thì phải học dần qua hàng
/// nghìn epoch và cũng chỉ cho ra quan hệ mờ.
/// </summary>
public sealed class KnowledgeBase
{
    private readonly Dictionary<string, Fact> _facts = new(StringComparer.Ordinal);

    public int Count => _facts.Count;

    public IReadOnlyCollection<Fact> Facts => _facts.Values;

    /// <summary>Thêm sự thật; ghi đè nếu tên đã tồn tại.</summary>
    public Fact Assert(string statement)
    {
        var fact = Parser.ParseFact(statement);
        _facts[fact.Name] = fact;
        return fact;
    }

    public void AssertAll(params string[] statements)
    {
        ArgumentNullException.ThrowIfNull(statements);

        foreach (string statement in statements)
        {
            Assert(statement);
        }
    }

    public bool TryGet(string name, out Fact fact) => _facts.TryGetValue(name, out fact!);

    public bool Contains(string name) => _facts.ContainsKey(name);

    public bool Remove(string name) => _facts.Remove(name);

    public void Clear() => _facts.Clear();

    /// <summary>
    /// Tìm các sự thật LIÊN QUAN tới một biểu thức — tức là tập biến mà việc
    /// tính biểu thức này thực sự cần, lấy đệ quy.
    ///
    /// Đây là bước "Find relevant facts". Với vài chục sự thật thì duyệt hết
    /// cũng được, nhưng với hàng triệu thì phải biết lọc — và đây chính là
    /// nguyên lý của retrieval trong các hệ thống lớn.
    /// </summary>
    public IReadOnlyList<Fact> FindRelevant(Expression expression)
    {
        ArgumentNullException.ThrowIfNull(expression);

        var found = new List<Fact>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        void Walk(Expression node)
        {
            switch (node)
            {
                case VariableExpression variable:
                    if (!seen.Add(variable.Name)) return;
                    if (_facts.TryGetValue(variable.Name, out var fact))
                    {
                        found.Add(fact);
                        Walk(fact.Value);
                    }

                    break;

                case BinaryExpression binary:
                    Walk(binary.Left);
                    Walk(binary.Right);
                    break;
            }
        }

        Walk(expression);
        return found;
    }
}
