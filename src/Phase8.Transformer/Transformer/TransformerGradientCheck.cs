using Ai.Phase8.Language;
using Ai.Phase8.Training;

namespace Ai.Phase8.Transformer;

public sealed record TransformerCheckResult(
    double[] Analytic,
    double[] Numerical,
    double MaxAbsoluteError,
    int WorstIndex)
{
    public bool Passed => MaxAbsoluteError < 1e-7;
}

/// <summary>
/// Gradient check cho toàn bộ transformer.
///
/// Backprop qua attention và LayerNorm là phần dễ sai nhất trong cả dự án:
/// Jacobian của softmax, ba nhánh gradient của X trong attention, hai số hạng
/// hiệu chỉnh qua μ và σ của LayerNorm. Không có công cụ này thì gần như
/// chắc chắn sẽ có một dấu sai ở đâu đó mà không ai phát hiện ra.
///
/// Nguyên lý vẫn y hệt Phase 1: so với định nghĩa gốc của đạo hàm.
/// </summary>
public static class TransformerGradientCheck
{
    public static TransformerCheckResult Check(
        MiniTransformer model,
        IReadOnlyList<NextTokenSample> samples,
        double h = 1e-5)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(samples);

        var analytic = ComputeByBackpropagation(model, samples);
        var numerical = ComputeNumerically(model, samples, h);

        double maxError = 0;
        int worst = 0;

        for (int i = 0; i < analytic.Length; i++)
        {
            double error = Math.Abs(analytic[i] - numerical[i]);
            if (error > maxError)
            {
                maxError = error;
                worst = i;
            }
        }

        return new TransformerCheckResult(analytic, numerical, maxError, worst);
    }

    public static double[] ComputeByBackpropagation(
        MiniTransformer model, IReadOnlyList<NextTokenSample> samples)
    {
        var loss = new SoftmaxCrossEntropy();
        model.ZeroGradients();

        foreach (var sample in samples)
        {
            var target = model.OneHot(sample.Target);
            var logits = model.Forward(sample.Context);
            model.Backward(loss.Gradient(logits, target));
        }

        var gradients = model.GetAllGradients();
        for (int i = 0; i < gradients.Length; i++) gradients[i] /= samples.Count;

        return gradients;
    }

    public static double[] ComputeNumerically(
        MiniTransformer model, IReadOnlyList<NextTokenSample> samples, double h = 1e-5)
    {
        var parameters = model.GetAllParameters();
        var gradients = new double[parameters.Length];

        for (int i = 0; i < parameters.Length; i++)
        {
            double original = parameters[i];

            parameters[i] = original + h;
            model.SetAllParameters(parameters);
            double lossPlus = model.AverageLoss(samples);

            parameters[i] = original - h;
            model.SetAllParameters(parameters);
            double lossMinus = model.AverageLoss(samples);

            gradients[i] = (lossPlus - lossMinus) / (2 * h);
            parameters[i] = original;
        }

        model.SetAllParameters(parameters);
        return gradients;
    }
}
