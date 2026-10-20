using System.Text;

namespace Ai.Phase2.LinAlg;

/// <summary>
/// Ma trận số thực, tự viết từ đầu.
///
/// Vì sao AI cần ma trận? Một LAYER có nhiều neuron, mỗi neuron có một vector
/// trọng số riêng. Xếp chồng các vector đó lại thành ma trận W (mỗi HÀNG là
/// một neuron), khi đó toàn bộ layer chỉ là:
///
///     output = activation( W · x + b )
///
/// Đây là lý do GPU chạy AI nhanh: GPU sinh ra để nhân ma trận.
/// </summary>
public sealed class Mat
{
    private readonly double[,] _values;

    public int Rows { get; }

    public int Cols { get; }

    public double this[int row, int col]
    {
        get => _values[row, col];
        set => _values[row, col] = value;
    }

    public Mat(int rows, int cols)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));

        Rows = rows;
        Cols = cols;
        _values = new double[rows, cols];
    }

    /// <summary>Tạo ma trận từ các hàng: FromRows([1,2], [3,4]) -> ma trận 2x2.</summary>
    public static Mat FromRows(params double[][] rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (rows.Length == 0) return new Mat(0, 0);

        int cols = rows[0].Length;
        var m = new Mat(rows.Length, cols);

        for (int r = 0; r < rows.Length; r++)
        {
            if (rows[r].Length != cols)
            {
                throw new ArgumentException("Các hàng phải có cùng số cột.", nameof(rows));
            }

            for (int c = 0; c < cols; c++)
            {
                m[r, c] = rows[r][c];
            }
        }

        return m;
    }

    /// <summary>
    /// Nhân ma trận với vector:  (W · x)ᵣ = Σ_c W[r,c] · x[c]
    ///
    /// Ví dụ:  W = [[1, 2],      x = [10]
    ///              [3, 4]]           [20]
    ///     hàng 0: 1·10 + 2·20 = 50
    ///     hàng 1: 3·10 + 4·20 = 110   ->  W·x = [50, 110]
    ///
    /// Mỗi HÀNG của W là trọng số của một neuron, nên mỗi phần tử kết quả
    /// chính là tổng có trọng số của một neuron.
    /// </summary>
    public Vec Multiply(Vec vector)
    {
        ArgumentNullException.ThrowIfNull(vector);

        if (vector.Length != Cols)
        {
            throw new ArgumentException(
                $"Không nhân được ma trận {Rows}x{Cols} với vector độ dài {vector.Length}.",
                nameof(vector));
        }

        var result = new Vec(Rows);
        for (int r = 0; r < Rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < Cols; c++)
            {
                sum += _values[r, c] * vector[c];
            }

            result[r] = sum;
        }

        return result;
    }

    /// <summary>Nhân hai ma trận: (A·B)[r,c] = Σ_k A[r,k]·B[k,c]</summary>
    public Mat Multiply(Mat other)
    {
        ArgumentNullException.ThrowIfNull(other);

        if (other.Rows != Cols)
        {
            throw new ArgumentException(
                $"Không nhân được ma trận {Rows}x{Cols} với {other.Rows}x{other.Cols}.",
                nameof(other));
        }

        var result = new Mat(Rows, other.Cols);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < other.Cols; c++)
            {
                double sum = 0;
                for (int k = 0; k < Cols; k++)
                {
                    sum += _values[r, k] * other[k, c];
                }

                result[r, c] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Chuyển vị: đổi hàng thành cột. Wᵀ[c,r] = W[r,c]
    /// Phase 3 cần phép này để lan truyền lỗi ngược từ layer sau về layer trước.
    /// </summary>
    public Mat Transpose()
    {
        var result = new Mat(Cols, Rows);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                result[c, r] = _values[r, c];
            }
        }

        return result;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        for (int r = 0; r < Rows; r++)
        {
            sb.Append("  [");
            for (int c = 0; c < Cols; c++)
            {
                if (c > 0) sb.Append(", ");
                sb.Append(_values[r, c].ToString("F4"));
            }

            sb.AppendLine("]");
        }

        return sb.ToString().TrimEnd();
    }
}
