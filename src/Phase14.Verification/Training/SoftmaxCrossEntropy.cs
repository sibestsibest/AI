using Ai.Phase14.LinAlg;

namespace Ai.Phase14.Training;

/// <summary>
/// Cross Entropy Loss ghép sẵn với Softmax.
///
/// CROSS ENTROPY:
///     L = − Σᵢ yᵢ · log(pᵢ)
///
/// Với target là one-hot (chỉ một từ đúng, ở vị trí k) thì mọi số hạng khác
/// biến mất và công thức thu về:
///     L = − log(p_k)
/// tức là: "model gán bao nhiêu xác suất cho đáp án đúng?"
///
/// VÍ DỤ SỐ (từ đúng là 'cats'):
///     p(cats) = 0.90  ->  L = −log(0.90) = 0.105    tốt
///     p(cats) = 0.50  ->  L = −log(0.50) = 0.693    tạm
///     p(cats) = 0.10  ->  L = −log(0.10) = 2.303    tệ
///     p(cats) = 0.01  ->  L = −log(0.01) = 4.605    rất tệ
///
/// Ý NGHĨA THÔNG TIN: L chính là số "nat" thông tin cần để mã hoá đáp án
/// đúng dưới phân bố mà model dự đoán. Model càng ngạc nhiên thì càng tốn
/// nhiều bit. Học = học cách bớt ngạc nhiên.
///
/// e^L được gọi là PERPLEXITY: model đang lưỡng lự giữa bao nhiêu phương án.
/// Perplexity 1.0 = chắc chắn tuyệt đối; perplexity 6.0 trên từ điển 6 từ
/// = đoán bừa hoàn toàn.
///
/// VÌ SAO GHÉP SOFTMAX VÀO LUÔN?
/// Tính riêng thì đạo hàm của softmax là một ma trận Jacobian V×V, rối và dễ
/// sai. Ghép lại thì mọi thứ rút gọn thành một dòng duy nhất:
///
///     ∂L/∂z = p − y
///
/// "Xác suất dự đoán trừ đi đáp án đúng". Giống hệt cặp Sigmoid + BCE ở
/// Phase 4 — không phải trùng hợp: sigmoid là softmax cho 2 lớp.
///
/// LƯU Ý: <see cref="Compute"/> và <see cref="Gradient"/> nhận LOGITS (đầu ra
/// thô của layer cuối, activation Identity), KHÔNG phải xác suất. Softmax
/// được áp bên trong.
/// </summary>
public sealed class SoftmaxCrossEntropy : ILossFunction
{
    private const double Epsilon = 1e-12;

    public string Name => "SoftmaxCrossEntropy";

    /// <param name="predicted">Logits thô từ layer cuối.</param>
    /// <param name="target">Vector one-hot của từ đúng.</param>
    public double Compute(Vec predicted, Vec target)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(target);
        RequireSameLength(predicted, target);

        var probabilities = Softmax.Apply(predicted);

        double loss = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            if (target[i] != 0)
            {
                loss -= target[i] * Math.Log(Math.Max(probabilities[i], Epsilon));
            }
        }

        return loss;
    }

    /// <summary>∂L/∂logits = p − y. Một dòng, không Jacobian.</summary>
    public Vec Gradient(Vec predicted, Vec target)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(target);
        RequireSameLength(predicted, target);

        return Softmax.Apply(predicted).Subtract(target);
    }

    /// <summary>Perplexity = e^L — "model đang lưỡng lự giữa bao nhiêu phương án".</summary>
    public static double Perplexity(double crossEntropyLoss) => Math.Exp(crossEntropyLoss);

    private static void RequireSameLength(Vec predicted, Vec target)
    {
        if (predicted.Length != target.Length)
        {
            throw new ArgumentException(
                $"Logits có {predicted.Length} phần tử nhưng target có {target.Length}.");
        }
    }
}
