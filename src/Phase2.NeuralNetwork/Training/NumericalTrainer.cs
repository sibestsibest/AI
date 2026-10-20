using Ai.Phase2.Network;

namespace Ai.Phase2.Training;

/// <summary>Trạng thái model tại một epoch.</summary>
public readonly record struct EpochSnapshot(int Epoch, double Loss);

/// <summary>
/// Vòng lặp huấn luyện của Phase 2. Giống hệt Phase 1 về cấu trúc, chỉ khác
/// chỗ lấy gradient: Phase 1 có công thức giải tích cho 2 tham số, ở đây
/// dùng numerical gradient cho toàn bộ mạng.
///
///     mỗi epoch:
///         đo loss                        (để quan sát)
///         tính gradient theo mọi tham số
///         optimizer.Step(params, grads)  (p -= lr·g)
///         ghi tham số mới vào mạng
///
/// Phase 3 sẽ thêm BackpropTrainer có đúng cấu trúc này, chỉ thay dòng
/// tính gradient. Khi đó hai trainer phải cho kết quả trùng nhau.
/// </summary>
public sealed class NumericalTrainer
{
    private readonly NeuralNetwork _network;
    private readonly ILossFunction _loss;
    private readonly IOptimizer _optimizer;

    public NumericalTrainer(NeuralNetwork network, ILossFunction loss, IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(optimizer);

        _network = network;
        _loss = loss;
        _optimizer = optimizer;
    }

    public void Train(TrainingData data, int epochs, Action<EpochSnapshot>? onEpoch = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (epochs < 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double loss = LossEvaluator.AverageLoss(_network, data, _loss);

            var gradients = NumericalGradient.Compute(_network, data, _loss);

            // Lưu gradient vào từng neuron: giữ đúng lời hứa "Neuron phải có
            // gradient", và giúp soi được từng neuron khi debug.
            _network.SetGradients(gradients);

            var parameters = _network.GetParameters();
            _optimizer.Step(parameters, gradients);
            _network.SetParameters(parameters);

            onEpoch?.Invoke(new EpochSnapshot(epoch, loss));
        }
    }
}
