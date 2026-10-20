namespace Ai.Phase1;

/// <summary>
/// Kiểm chứng công thức đạo hàm viết tay bằng đạo hàm số (numerical gradient).
///
/// Định nghĩa gốc của đạo hàm:  f'(x) ≈ ( f(x+h) - f(x-h) ) / (2h)
///
/// Nếu gradient giải tích (CalculateGradient) trùng với gradient số này thì
/// phần backpropagation chắc chắn đúng. Đây là công cụ bắt buộc phải có trước
/// khi sang mạng nhiều lớp ở phase sau, vì lúc đó sai một dấu là không thể
/// phát hiện bằng mắt.
/// </summary>
public static class GradientCheck
{
    /// <summary>Tính MSE cho một cặp (w, b) bất kỳ, không phụ thuộc model.</summary>
    public static double LossAt(double w, double b, IReadOnlyList<Sample> data)
    {
        double sum = 0;

        foreach (var sample in data)
        {
            double error = w * sample.X + b - sample.Y;
            sum += error * error;
        }

        return sum / data.Count;
    }

    /// <summary>Gradient xấp xỉ bằng sai phân trung tâm.</summary>
    public static Gradient Numerical(double w, double b, IReadOnlyList<Sample> data, double h = 1e-6)
    {
        double dw = (LossAt(w + h, b, data) - LossAt(w - h, b, data)) / (2 * h);
        double db = (LossAt(w, b + h, data) - LossAt(w, b - h, data)) / (2 * h);

        return new Gradient(dw, db);
    }
}
