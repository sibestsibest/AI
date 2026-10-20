using Ai.Phase5.LinAlg;
using Ai.Phase5.Network;

namespace Ai.Phase5.Training;

/// <summary>
/// MSE — chính hàm loss của Phase 1, nay mở rộng cho nhiều đầu ra:
///
///     L = (1/m) · Σⱼ (ŷⱼ − yⱼ)²          m = số đầu ra
///     ∂L/∂ŷⱼ = (2/m) · (ŷⱼ − yⱼ)
///
/// Với m = 1 công thức thu về đúng Phase 1, nên hai phase nhất quán với nhau.
/// </summary>
public sealed class MeanSquaredError : ILossFunction
{
    public string Name => "MeanSquaredError";

    public double Compute(Vec predicted, Vec target)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(target);
        RequireSameLength(predicted, target);

        double sum = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double error = predicted[i] - target[i];
            sum += error * error;
        }

        return sum / predicted.Length;
    }

    public Vec Gradient(Vec predicted, Vec target)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(target);
        RequireSameLength(predicted, target);

        int m = predicted.Length;
        var gradient = new Vec(m);

        for (int i = 0; i < m; i++)
        {
            gradient[i] = 2.0 / m * (predicted[i] - target[i]);
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

/// <summary>
/// Tính loss trung bình trên CẢ tập dữ liệu. Đây là đại lượng mà quá trình
/// học thực sự tìm cách cực tiểu hoá:
///
///     J = (1/n) · Σᵢ L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
/// </summary>
public static class LossEvaluator
{
    public static double AverageLoss(NeuralNetwork network, TrainingData data, ILossFunction loss)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(loss);

        double total = 0;
        foreach (var sample in data)
        {
            total += loss.Compute(network.Forward(sample.Input), sample.Target);
        }

        return total / data.Count;
    }
}
