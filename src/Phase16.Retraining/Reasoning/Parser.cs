using System.Globalization;
using System.Text;

namespace Ai.Phase16.Reasoning;

public sealed class ParseException(string message) : Exception(message);

/// <summary>
/// Bộ phân tích cú pháp đệ quy xuống (recursive descent).
///
/// Ngữ pháp:
///     statement  := IDENT '=' comparison
///     comparison := sum ( ('=' | '>' | '<') sum )?
///     sum        := product ( ('+' | '-') product )*
///     product    := atom ( ('*' | '/') atom )*
///     atom       := NUMBER | IDENT | '(' comparison ')' | '-' atom
///
/// Thứ tự các quy tắc chính là THỨ TỰ ƯU TIÊN: phép nào nằm sâu hơn trong
/// ngữ pháp thì được tính trước. Vì product nằm dưới sum, nên
/// "2 + 3 * 4" được phân tích thành 2 + (3*4) = 14, không phải (2+3)*4 = 20.
/// </summary>
public sealed class Parser
{
    private readonly List<string> _tokens;
    private int _position;

    private Parser(List<string> tokens) => _tokens = tokens;

    /// <summary>Phân tích một phát biểu dạng "TÊN = biểu thức".</summary>
    public static Fact ParseFact(string statement)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(statement);

        var parser = new Parser(Tokenize(statement));

        string name = parser.NextToken()
            ?? throw new ParseException($"Phát biểu rỗng: \"{statement}\"");

        if (!IsIdentifier(name))
        {
            throw new ParseException($"\"{name}\" không phải tên biến hợp lệ trong \"{statement}\".");
        }

        if (parser.NextToken() != "=")
        {
            throw new ParseException($"Thiếu dấu '=' trong \"{statement}\".");
        }

        var value = parser.ParseComparison();
        parser.RequireEnd(statement);

        return new Fact(name, value, statement.Trim());
    }

    /// <summary>Phân tích một biểu thức bất kỳ (dùng cho câu hỏi).</summary>
    public static Expression ParseExpression(string text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);

        var parser = new Parser(Tokenize(text));
        var expression = parser.ParseComparison();
        parser.RequireEnd(text);

        return expression;
    }

    public static List<string> Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var tokens = new List<string>();
        var buffer = new StringBuilder();

        void Flush()
        {
            if (buffer.Length > 0)
            {
                tokens.Add(buffer.ToString());
                buffer.Clear();
            }
        }

        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                Flush();
            }
            else if ("+-*/=<>()".Contains(c))
            {
                Flush();
                tokens.Add(c.ToString());
            }
            else
            {
                buffer.Append(c);
            }
        }

        Flush();
        return tokens;
    }

    public static bool IsIdentifier(string token) =>
        token.Length > 0 && (char.IsLetter(token[0]) || token[0] == '_')
        && token.All(c => char.IsLetterOrDigit(c) || c == '_');

    private Expression ParseComparison()
    {
        var left = ParseSum();

        string? op = Peek();
        if (op is "=" or ">" or "<")
        {
            NextToken();
            var right = ParseSum();

            return new BinaryExpression(left, op switch
            {
                "=" => Operator.Equal,
                ">" => Operator.Greater,
                _ => Operator.Less,
            }, right);
        }

        return left;
    }

    private Expression ParseSum()
    {
        var left = ParseProduct();

        while (Peek() is "+" or "-")
        {
            string op = NextToken()!;
            left = new BinaryExpression(left, op == "+" ? Operator.Add : Operator.Subtract, ParseProduct());
        }

        return left;
    }

    private Expression ParseProduct()
    {
        var left = ParseAtom();

        while (Peek() is "*" or "/")
        {
            string op = NextToken()!;
            left = new BinaryExpression(left, op == "*" ? Operator.Multiply : Operator.Divide, ParseAtom());
        }

        return left;
    }

    private Expression ParseAtom()
    {
        string token = NextToken() ?? throw new ParseException("Biểu thức kết thúc đột ngột.");

        if (token == "(")
        {
            var inner = ParseComparison();

            if (NextToken() != ")")
            {
                throw new ParseException("Thiếu dấu ')'.");
            }

            return inner;
        }

        // Dấu trừ đứng đầu: -5 hoặc -A
        if (token == "-")
        {
            return new BinaryExpression(new NumberExpression(0), Operator.Subtract, ParseAtom());
        }

        if (double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out double number))
        {
            return new NumberExpression(number);
        }

        if (IsIdentifier(token))
        {
            return new VariableExpression(token);
        }

        throw new ParseException($"Không hiểu ký hiệu \"{token}\".");
    }

    private string? Peek() => _position < _tokens.Count ? _tokens[_position] : null;

    private string? NextToken() => _position < _tokens.Count ? _tokens[_position++] : null;

    private void RequireEnd(string source)
    {
        if (_position < _tokens.Count)
        {
            throw new ParseException($"Thừa ký hiệu \"{_tokens[_position]}\" trong \"{source}\".");
        }
    }
}
