namespace Ai.Phase2.Activations;

/// <summary>
/// f(z) = tanh(z) = (e^z − e^(−z)) / (e^z + e^(−z))  — bóp về khoảng (−1, 1).
///
/// Giống sigmoid nhưng đối xứng quanh 0, nên đầu ra có trung bình ≈ 0.
/// Điều đó làm gradient ổn định hơn, vì vậy tanh thường tốt hơn sigmoid
/// cho các layer ẩn.
///
/// Đạo hàm:  f'(z) = 1 − tanh²(z)
/// Kiểm chứng số: tanh(0) = 0 -> f'(0) = 1 − 0 = 1 (dốc nhất tại gốc).
/// </summary>
public sealed class Tanh : IActivation
{
    public string Name => "Tanh";

    public double Forward(double z) => Math.Tanh(z);

    public double Derivative(double z)
    {
        double t = Math.Tanh(z);
        return 1.0 - t * t;
    }
}
