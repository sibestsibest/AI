using Ai.Phase8.LinAlg;

namespace Ai.Phase8.Transformer;

/// <summary>
/// Một khối tham số học được: giá trị và gradient tích luỹ của nó.
///
/// Phase 3 gắn gradient trực tiếp vào từng Neuron. Transformer không có
/// "neuron" theo nghĩa đó — nó là các ma trận nhân với nhau — nên gradient
/// được gom theo MA TRẬN. Cách tổ chức khác, chain rule thì y hệt.
/// </summary>
public sealed class Param
{
    public Mat Value { get; }

    public Mat Grad { get; }

    public string Name { get; }

    public int Count => Value.Rows * Value.Cols;

    public Param(string name, int rows, int cols)
    {
        Name = name;
        Value = new Mat(rows, cols);
        Grad = new Mat(rows, cols);
    }

    /// <summary>Khởi tạo Xavier — cùng lý do như Phase 2.</summary>
    public static Param Random(string name, int rows, int cols, Random rng)
    {
        var p = new Param(name, rows, cols);
        double limit = Math.Sqrt(6.0 / (rows + cols));

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                p.Value[r, c] = (rng.NextDouble() * 2 - 1) * limit;
            }
        }

        return p;
    }

    /// <summary>Khởi tạo bằng một hằng số (bias = 0, gamma của LayerNorm = 1).</summary>
    public static Param Constant(string name, int rows, int cols, double value)
    {
        var p = new Param(name, rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++) p.Value[r, c] = value;
        }

        return p;
    }

    public void ZeroGrad()
    {
        for (int r = 0; r < Grad.Rows; r++)
        {
            for (int c = 0; c < Grad.Cols; c++) Grad[r, c] = 0;
        }
    }

    public void AccumulateGrad(Mat delta)
    {
        ArgumentNullException.ThrowIfNull(delta);

        for (int r = 0; r < Grad.Rows; r++)
        {
            for (int c = 0; c < Grad.Cols; c++) Grad[r, c] += delta[r, c];
        }
    }

    public void AccumulateGrad(Vec rowVector)
    {
        ArgumentNullException.ThrowIfNull(rowVector);

        for (int c = 0; c < Grad.Cols; c++) Grad[0, c] += rowVector[c];
    }

    public override string ToString() => $"{Name} ({Value.Rows}×{Value.Cols})";
}
