using System.Diagnostics;
using Ai.Phase6.Network;

namespace Ai.Phase6.Training;

/// <summary>
/// Tính gradient của loss theo MỌI tham số bằng định nghĩa gốc của đạo hàm:
///
///     ∂J/∂pᵢ ≈ ( J(pᵢ + h) − J(pᵢ − h) ) / (2h)
///
/// Đây là bản tổng quát hoá của GradientCheck ở Phase 1, cho cả một mạng.
///
/// VÌ SAO PHASE 2 DÙNG NÓ: Phase 2 mới chỉ có Forward Pass. Với công cụ này,
/// mạng đã train được THẬT mà không cần một dòng backpropagation nào —
/// đủ để chứng minh Neuron/Layer/Network/Loss/Optimizer đều đúng.
///
/// VÌ SAO PHẢI BỎ NÓ Ở PHASE 3: mỗi bước học tốn 2·P lần forward pass
/// (P = số tham số). Mạng 25 tham số -> 50 lần forward cho MỘT bước.
/// Backpropagation cho cùng kết quả chỉ với 1 lần forward + 1 lần backward,
/// bất kể P lớn cỡ nào. Với GPT cỡ 10¹² tham số, cách này cần 2·10¹² lần
/// forward mỗi bước — bất khả thi tuyệt đối.
///
/// Nhưng nó vẫn còn giá trị vĩnh viễn: làm THƯỚC ĐO để kiểm tra
/// backpropagation viết tay có đúng không.
/// </summary>
public static class NumericalGradient
{
    /// <summary>Số lần forward pass đã thực hiện — để minh hoạ chi phí.</summary>
    public static long ForwardPassCount { get; private set; }

    public static void ResetCounter() => ForwardPassCount = 0;

    /// <summary>
    /// Pseudocode:
    ///   for mỗi tham số i:
    ///       lưu giá trị gốc
    ///       p[i] = gốc + h  ->  tính loss trung bình -> Jplus
    ///       p[i] = gốc − h  ->  tính loss trung bình -> Jminus
    ///       g[i] = (Jplus − Jminus) / (2h)
    ///       trả p[i] về giá trị gốc
    /// </summary>
    public static double[] Compute(
        NeuralNetwork network,
        TrainingData data,
        ILossFunction loss,
        double h = 1e-5)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(loss);

        var parameters = network.GetParameters();
        var gradients = new double[parameters.Length];

        for (int i = 0; i < parameters.Length; i++)
        {
            double original = parameters[i];

            parameters[i] = original + h;
            network.SetParameters(parameters);
            double lossPlus = LossEvaluator.AverageLoss(network, data, loss);

            parameters[i] = original - h;
            network.SetParameters(parameters);
            double lossMinus = LossEvaluator.AverageLoss(network, data, loss);

            gradients[i] = (lossPlus - lossMinus) / (2 * h);

            parameters[i] = original;
            ForwardPassCount += 2L * data.Count;
        }

        // Trả mạng về đúng trạng thái ban đầu — hàm này không được để lại dấu vết.
        network.SetParameters(parameters);

        Debug.Assert(gradients.Length == network.ParameterCount);
        return gradients;
    }
}
