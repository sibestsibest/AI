using System.Text;

namespace Ai.Phase2.LinAlg;

/// <summary>
/// Vector số thực, tự viết từ đầu — không dùng System.Numerics hay thư viện nào.
///
/// Vì sao AI cần vector? Vì một neuron nhận NHIỀU đầu vào cùng lúc. Thay vì
/// viết w1*x1 + w2*x2 + w3*x3 + ... ta gom lại thành một phép tích vô hướng
/// (dot product). Toàn bộ deep learning chỉ là vector và ma trận nhân nhau.
/// </summary>
public sealed class Vec
{
    private readonly double[] _values;

    public int Length => _values.Length;

    public double this[int i]
    {
        get => _values[i];
        set => _values[i] = value;
    }

    /// <summary>Vector toàn số 0 với độ dài cho trước.</summary>
    public Vec(int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        _values = new double[length];
    }

    public Vec(params double[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        _values = (double[])values.Clone();
    }

    public static Vec Of(params double[] values) => new(values);

    public double[] ToArray() => (double[])_values.Clone();

    /// <summary>
    /// Tích vô hướng:  a · b = Σ aᵢ·bᵢ
    /// Ví dụ: [1, 2, 3] · [4, 5, 6] = 1·4 + 2·5 + 3·6 = 4 + 10 + 18 = 32
    /// Đây là phép tính lõi của một neuron.
    /// </summary>
    public double Dot(Vec other)
    {
        RequireSameLength(other);

        double sum = 0;
        for (int i = 0; i < Length; i++)
        {
            sum += _values[i] * other._values[i];
        }

        return sum;
    }

    /// <summary>Cộng từng phần tử: [1,2] + [10,20] = [11,22]</summary>
    public Vec Add(Vec other)
    {
        RequireSameLength(other);

        var result = new Vec(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = _values[i] + other._values[i];
        }

        return result;
    }

    /// <summary>Trừ từng phần tử — dùng để tính sai số (ŷ − y).</summary>
    public Vec Subtract(Vec other)
    {
        RequireSameLength(other);

        var result = new Vec(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = _values[i] - other._values[i];
        }

        return result;
    }

    /// <summary>Nhân với một số: 2 · [1,2,3] = [2,4,6]</summary>
    public Vec Scale(double factor)
    {
        var result = new Vec(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = _values[i] * factor;
        }

        return result;
    }

    /// <summary>
    /// Nhân từng phần tử (Hadamard product): [1,2,3] ⊙ [10,20,30] = [10,40,90]
    /// Khác hẳn dot product — kết quả là vector, không phải một số.
    /// Phase 3 sẽ dùng phép này để nhân delta với đạo hàm của activation.
    /// </summary>
    public Vec Hadamard(Vec other)
    {
        RequireSameLength(other);

        var result = new Vec(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = _values[i] * other._values[i];
        }

        return result;
    }

    /// <summary>Áp một hàm lên từng phần tử — dùng để chạy activation cho cả layer.</summary>
    public Vec Map(Func<double, double> f)
    {
        ArgumentNullException.ThrowIfNull(f);

        var result = new Vec(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = f(_values[i]);
        }

        return result;
    }

    public double Sum()
    {
        double total = 0;
        foreach (double v in _values) total += v;
        return total;
    }

    /// <summary>Độ dài Euclid: ‖v‖ = √(Σ vᵢ²). Dùng cho cosine similarity ở Phase 5.</summary>
    public double Norm() => Math.Sqrt(Dot(this));

    public static Vec operator +(Vec a, Vec b) => a.Add(b);
    public static Vec operator -(Vec a, Vec b) => a.Subtract(b);
    public static Vec operator *(Vec a, double k) => a.Scale(k);
    public static Vec operator *(double k, Vec a) => a.Scale(k);

    private void RequireSameLength(Vec other)
    {
        ArgumentNullException.ThrowIfNull(other);

        if (other.Length != Length)
        {
            throw new ArgumentException(
                $"Kích thước vector không khớp: {Length} và {other.Length}.", nameof(other));
        }
    }

    public override string ToString()
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(_values[i].ToString("F4"));
        }

        return sb.Append(']').ToString();
    }
}
