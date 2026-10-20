using Ai.Phase8.LinAlg;
using Ai.Phase8.Transformer;

namespace Ai.Phase8.Tests;

/// <summary>
/// Công cụ kiểm chứng backward của TỪNG module riêng lẻ.
///
/// Cách làm: định nghĩa một hàm vô hướng giả
///
///     L = Σᵢⱼ output[i,j] · R[i,j]        với R là ma trận ngẫu nhiên cố định
///
/// Khi đó ∂L/∂output = R, nên gọi Backward(R) phải cho ra đúng gradient của L
/// theo tham số và theo đầu vào. So với đạo hàm số là biết ngay đúng hay sai.
///
/// Gradient check toàn transformer cũng bắt được lỗi, nhưng không cho biết
/// lỗi nằm ở module nào. Kiểm tra từng module thì định vị được ngay.
/// </summary>
public static class ModuleGradientCheck
{
    public static double MaxParameterError(
        Func<Mat, Mat> forward,
        Func<Mat, Mat> backward,
        Action zeroGrad,
        IReadOnlyList<Param> parameters,
        Mat input,
        Random rng,
        double h = 1e-6)
    {
        var output = forward(input);
        var r = RandomMat(output.Rows, output.Cols, rng);

        zeroGrad();
        forward(input);
        backward(r);

        double maxError = 0;

        foreach (var p in parameters)
        {
            for (int row = 0; row < p.Value.Rows; row++)
            {
                for (int col = 0; col < p.Value.Cols; col++)
                {
                    double original = p.Value[row, col];

                    p.Value[row, col] = original + h;
                    double plus = WeightedSum(forward(input), r);

                    p.Value[row, col] = original - h;
                    double minus = WeightedSum(forward(input), r);

                    p.Value[row, col] = original;

                    maxError = Math.Max(maxError, Math.Abs((plus - minus) / (2 * h) - p.Grad[row, col]));
                }
            }
        }

        return maxError;
    }

    public static double MaxInputError(
        Func<Mat, Mat> forward,
        Func<Mat, Mat> backward,
        Action zeroGrad,
        Mat input,
        Random rng,
        double h = 1e-6)
    {
        var output = forward(input);
        var r = RandomMat(output.Rows, output.Cols, rng);

        zeroGrad();
        forward(input);
        var analytic = backward(r);

        double maxError = 0;

        for (int row = 0; row < input.Rows; row++)
        {
            for (int col = 0; col < input.Cols; col++)
            {
                double original = input[row, col];

                input[row, col] = original + h;
                double plus = WeightedSum(forward(input), r);

                input[row, col] = original - h;
                double minus = WeightedSum(forward(input), r);

                input[row, col] = original;

                maxError = Math.Max(maxError, Math.Abs((plus - minus) / (2 * h) - analytic[row, col]));
            }
        }

        return maxError;
    }

    public static Mat RandomMat(int rows, int cols, Random rng)
    {
        var m = new Mat(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++) m[r, c] = rng.NextDouble() * 2 - 1;
        }

        return m;
    }

    private static double WeightedSum(Mat a, Mat weights)
    {
        double total = 0;
        for (int r = 0; r < a.Rows; r++)
        {
            for (int c = 0; c < a.Cols; c++) total += a[r, c] * weights[r, c];
        }

        return total;
    }
}
