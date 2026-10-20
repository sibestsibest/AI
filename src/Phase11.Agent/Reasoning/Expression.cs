namespace Ai.Phase11.Reasoning;

/// <summary>Các phép toán mà engine hiểu được.</summary>
public enum Operator
{
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    Greater,
    Less,
}

/// <summary>
/// Cây biểu thức (AST).
///
/// Đây là điểm khác biệt căn bản so với Phase 1–8. Mạng neural biểu diễn tri
/// thức bằng các con số mờ (trọng số) và suy luận bằng phép nhân ma trận —
/// không truy vết được, không giải thích được.
///
/// Reasoning engine biểu diễn tri thức bằng CẤU TRÚC tường minh: "C là tổng
/// của A và B" được lưu đúng như vậy. Nhờ đó nó suy luận CHÍNH XÁC (không
/// xấp xỉ), GIẢI THÍCH ĐƯỢC từng bước, và đúng ngay từ một ví dụ — không cần
/// train.
///
/// Đổi lại, nó chỉ làm được đúng những gì được lập trình. Không học được
/// từ dữ liệu, không xử lý được cái mơ hồ. Hai cách tiếp cận bù trừ cho
/// nhau — Phase 11 sẽ ghép cả hai lại.
/// </summary>
public abstract record Expression
{
    public abstract override string ToString();
}

/// <summary>Một hằng số: 10, 3.14</summary>
public sealed record NumberExpression(double Value) : Expression
{
    public override string ToString() =>
        Value == Math.Floor(Value) && Math.Abs(Value) < 1e15
            ? ((long)Value).ToString()
            : Value.ToString("G6");
}

/// <summary>Tham chiếu tới một biến khác: A, B, Total</summary>
public sealed record VariableExpression(string Name) : Expression
{
    public override string ToString() => Name;
}

/// <summary>Phép toán hai ngôi: A + B, A > B</summary>
public sealed record BinaryExpression(Expression Left, Operator Op, Expression Right) : Expression
{
    public static string SymbolOf(Operator op) => op switch
    {
        Operator.Add => "+",
        Operator.Subtract => "-",
        Operator.Multiply => "*",
        Operator.Divide => "/",
        Operator.Equal => "=",
        Operator.Greater => ">",
        Operator.Less => "<",
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    public bool IsComparison => Op is Operator.Equal or Operator.Greater or Operator.Less;

    public override string ToString() => $"{Left} {SymbolOf(Op)} {Right}";
}
