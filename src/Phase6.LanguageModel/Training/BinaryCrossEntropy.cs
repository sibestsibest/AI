using Ai.Phase6.LinAlg;

namespace Ai.Phase6.Training;

/// <summary>
/// Binary Cross Entropy — hàm loss dành riêng cho bài toán PHÂN LOẠI.
///
///     L = −[ y·log(ŷ) + (1−y)·log(1−ŷ) ]
///
/// Đọc công thức: chỉ một trong hai số hạng sống sót.
///     y = 1  ->  L = −log(ŷ)        đoán ŷ→1 thì L→0, đoán ŷ→0 thì L→∞
///     y = 0  ->  L = −log(1−ŷ)      ngược lại
///
/// Ví dụ số (y = 1):
///     ŷ = 0.9   ->  L = −log(0.9)   = 0.105     đoán tốt, phạt nhẹ
///     ŷ = 0.5   ->  L = −log(0.5)   = 0.693     đoán bừa
///     ŷ = 0.1   ->  L = −log(0.1)   = 2.303     đoán sai, phạt nặng
///     ŷ = 0.01  ->  L = −log(0.01)  = 4.605     rất sai, phạt rất nặng
///
/// VÌ SAO KHÔNG DÙNG MSE CHO PHÂN LOẠI?
/// Với MSE, sai hoàn toàn (ŷ=0 khi y=1) chỉ bị phạt 1.0 — hữu hạn và nhỏ.
/// Tệ hơn: khi ŷ tiến về 0 hoặc 1, sigmoid bão hoà nên f'(z) ≈ 0, gradient
/// của MSE gần như biến mất và model NGỪNG HỌC đúng lúc nó sai nhất.
/// BCE ghép với sigmoid thì gradient rút gọn thành đúng (ŷ − y) — không bao
/// giờ triệt tiêu. Đó là lý do mọi bài phân loại đều dùng cross entropy.
/// </summary>
public sealed class BinaryCrossEntropy : ILossFunction
{
    /// <summary>
    /// Chặn ŷ khỏi chạm đúng 0 hoặc 1, vì log(0) = −∞ sẽ làm hỏng mọi phép tính.
    /// </summary>
    private const double Epsilon = 1e-12;

    public string Name => "BinaryCrossEntropy";

    public double Compute(Vec predicted, Vec target)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(target);
        RequireSameLength(predicted, target);

        double sum = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double p = Math.Clamp(predicted[i], Epsilon, 1 - Epsilon);
            sum += -(target[i] * Math.Log(p) + (1 - target[i]) * Math.Log(1 - p));
        }

        return sum / predicted.Length;
    }

    /// <summary>
    ///     ∂L/∂ŷ = (ŷ − y) / (ŷ·(1 − ŷ))
    ///
    /// Mẫu số trông đáng sợ, nhưng chính nó là thứ triệt tiêu f'(z) = ŷ(1−ŷ)
    /// của sigmoid ở bước backward, để lại đúng δ = ŷ − y.
    /// </summary>
    public Vec Gradient(Vec predicted, Vec target)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(target);
        RequireSameLength(predicted, target);

        int m = predicted.Length;
        var gradient = new Vec(m);

        for (int i = 0; i < m; i++)
        {
            double p = Math.Clamp(predicted[i], Epsilon, 1 - Epsilon);
            gradient[i] = (p - target[i]) / (p * (1 - p)) / m;
        }

        return gradient;
    }

    private static void RequireSameLength(Vec predicted, Vec target)
    {
        if (predicted.Length != target.Length)
        {
            throw new ArgumentException(
                $"Dự đoán có {predicted.Length} phần tử nhưng target có {target.Length}.");
        }
    }
}
