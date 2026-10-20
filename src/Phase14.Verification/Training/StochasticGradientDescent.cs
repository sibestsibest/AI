namespace Ai.Phase14.Training;

/// <summary>
/// Gradient Descent — chính xác quy tắc của Phase 1, nay áp cho mọi tham số:
///
///     pᵢ := pᵢ − learningRate · ∂L/∂pᵢ
///
/// Phase 1 có 2 tham số nên viết tay hai dòng. Mạng ở Phase 2 có hàng chục,
/// GPT-4 có hàng nghìn tỉ — nhưng vẫn là đúng một dòng lệnh này lặp lại.
/// </summary>
public sealed class StochasticGradientDescent : IOptimizer
{
    public string Name => "SGD";

    public double LearningRate { get; }

    public StochasticGradientDescent(double learningRate)
    {
        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate phải dương.");
        }

        LearningRate = learningRate;
    }

    public void Step(double[] parameters, double[] gradients)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(gradients);

        if (parameters.Length != gradients.Length)
        {
            throw new ArgumentException(
                $"Có {parameters.Length} tham số nhưng {gradients.Length} gradient.");
        }

        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] -= LearningRate * gradients[i];
        }
    }
}
