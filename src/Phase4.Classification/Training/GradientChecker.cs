using Ai.Phase4.Network;

namespace Ai.Phase4.Training;

/// <summary>Kết quả đối chiếu backpropagation với đạo hàm số.</summary>
public sealed record GradientCheckResult(
    double[] Analytic,
    double[] Numerical,
    double MaxAbsoluteError,
    double MaxRelativeError,
    int WorstIndex)
{
    /// <summary>
    /// Đạt khi lệch TƯƠNG ĐỐI dưới 1e-6, HOẶC lệch TUYỆT ĐỐI dưới 1e-8.
    ///
    /// Vì sao cần cả hai điều kiện? Sai phân trung tâm có một sàn nhiễu
    /// không thể vượt qua:
    ///
    ///     nhiễu ≈ ε_máy · |L| / h  ≈  10⁻¹⁶ / 10⁻⁵  =  10⁻¹¹
    ///
    /// Với tham số có gradient thật ≈ 0 (chuyện rất thường gặp ở mạng sâu),
    /// đạo hàm số trả về đúng cái nhiễu 10⁻¹¹ đó, còn backprop trả về 10⁻¹⁸.
    /// Lệch tuyệt đối là 10⁻¹¹ — không đáng kể — nhưng lệch TƯƠNG ĐỐI lại
    /// thành 10⁻³ và nếu chỉ xét tiêu chí tương đối thì sẽ báo sai oan.
    ///
    /// Ngưỡng 1e-8 nằm cao hơn sàn nhiễu khoảng 1000 lần, trong khi một bug
    /// backprop thật sự gây lệch cỡ 1e-1 — thừa sức phân biệt.
    /// </summary>
    public bool Passed => MaxRelativeError < 1e-6 || MaxAbsoluteError < 1e-8;
}

/// <summary>
/// Đối chiếu gradient của backpropagation với numerical gradient.
///
/// Đây là công cụ quan trọng nhất của Phase 3. Backpropagation sai vẫn chạy,
/// vẫn in ra số, loss thậm chí vẫn có thể giảm — chỉ là model học kém hơn mức
/// đáng lẽ đạt được, và không cách nào phát hiện bằng mắt. Chỉ có gradient
/// check mới bắt được.
///
/// Quy trình chuẩn: viết backprop -> gradient check -> nếu đạt mới train thật.
/// </summary>
public static class GradientChecker
{
    public static GradientCheckResult Check(
        NeuralNetwork network,
        TrainingData data,
        ILossFunction loss,
        double h = 1e-5)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(loss);

        var analytic = ComputeByBackpropagation(network, data, loss);
        var numerical = NumericalGradient.Compute(network, data, loss, h);

        double maxAbsolute = 0;
        double maxRelative = 0;
        int worstIndex = 0;

        for (int i = 0; i < analytic.Length; i++)
        {
            double absolute = Math.Abs(analytic[i] - numerical[i]);

            // Chia cho độ lớn để so công bằng: lệch 0.001 trên gradient 1000
            // là bình thường, nhưng trên gradient 0.001 là sai hoàn toàn.
            double scale = Math.Max(Math.Abs(analytic[i]) + Math.Abs(numerical[i]), 1e-8);
            double relative = absolute / scale;

            if (relative > maxRelative)
            {
                maxRelative = relative;
                worstIndex = i;
            }

            maxAbsolute = Math.Max(maxAbsolute, absolute);
        }

        return new GradientCheckResult(analytic, numerical, maxAbsolute, maxRelative, worstIndex);
    }

    /// <summary>
    /// Chạy forward + backward trên cả tập dữ liệu và trả về gradient trung bình.
    /// Không cập nhật tham số — chỉ đọc gradient ra.
    /// </summary>
    public static double[] ComputeByBackpropagation(
        NeuralNetwork network,
        TrainingData data,
        ILossFunction loss)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(loss);

        network.ZeroGradients();

        foreach (var sample in data)
        {
            var predicted = network.Forward(sample.Input);
            network.Backward(loss.Gradient(predicted, sample.Target));
        }

        var gradients = network.GetGradients();
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] /= data.Count;
        }

        return gradients;
    }
}
