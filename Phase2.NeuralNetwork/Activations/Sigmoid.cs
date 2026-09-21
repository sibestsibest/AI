namespace Ai.Phase2.Activations;

/// <summary>
/// f(z) = 1 / (1 + e^(−z))  — bóp mọi số thực về khoảng (0, 1).
///
///     z = −∞ -> 0      z = 0 -> 0.5      z = +∞ -> 1
///
/// Vì đầu ra nằm trong (0,1) nên đọc được như một XÁC SUẤT — dùng cho
/// bài toán phân loại nhị phân ở Phase 4.
///
/// Đạo hàm có dạng cực đẹp:  f'(z) = f(z)·(1 − f(z))
/// Chứng minh: đặt s = (1+e^(−z))⁻¹, thì
///     s' = −(1+e^(−z))⁻² · (−e^(−z)) = e^(−z) / (1+e^(−z))²
///        = s · e^(−z)/(1+e^(−z)) = s · (1 − s)
/// Kiểm chứng số: f(0) = 0.5 -> f'(0) = 0.5·0.5 = 0.25.
/// </summary>
public sealed class Sigmoid : IActivation
{
    public string Name => "Sigmoid";

    public double Forward(double z) => 1.0 / (1.0 + Math.Exp(-z));

    public double Derivative(double z)
    {
        double s = Forward(z);
        return s * (1.0 - s);
    }
}
