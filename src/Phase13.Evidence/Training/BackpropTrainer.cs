using Ai.Phase13.Network;

namespace Ai.Phase13.Training;

/// <summary>
/// Vòng học đầy đủ của Phase 3 — đúng chu trình:
///
///     Forward -> Prediction -> Calculate Loss -> Backpropagation
///     -> Calculate gradients -> Update weights -> Forward again
///
/// Pseudocode:
///
///     mỗi epoch:
///         network.ZeroGradients()                  # xoá gradient cũ
///         cho từng mẫu (x, y):
///             ŷ = network.Forward(x)               # Forward -> Prediction
///             totalLoss += loss.Compute(ŷ, y)      # Calculate Loss
///             network.Backward(loss.Gradient(ŷ,y)) # Backprop -> gradients
///         g = network.GetGradients() / n           # trung bình trên n mẫu
///         optimizer.Step(params, g)                # Update weights
///         network.SetParameters(params)            # vòng sau Forward lại
///
/// So với NumericalTrainer của Phase 2, chỉ khác ĐÚNG cách lấy gradient.
/// Kết quả phải trùng nhau — có unit test khẳng định điều đó.
/// </summary>
public sealed class BackpropTrainer
{
    private readonly NeuralNetwork _network;
    private readonly ILossFunction _loss;
    private readonly IOptimizer _optimizer;

    /// <summary>Đếm số lần forward pass — để so chi phí với numerical gradient.</summary>
    public long ForwardPassCount { get; private set; }

    public BackpropTrainer(NeuralNetwork network, ILossFunction loss, IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(optimizer);

        _network = network;
        _loss = loss;
        _optimizer = optimizer;
    }

    /// <summary>
    /// Chạy một epoch và trả về loss trung bình ĐO TRƯỚC khi cập nhật.
    /// Tách riêng để demo có thể gọi từng bước một.
    /// </summary>
    public double RunEpoch(TrainingData data)
    {
        ArgumentNullException.ThrowIfNull(data);

        // Bước 0: xoá gradient của epoch trước. Quên bước này là model hỏng.
        _network.ZeroGradients();

        double totalLoss = 0;

        foreach (var sample in data)
        {
            // Forward -> Prediction
            var predicted = _network.Forward(sample.Input);
            ForwardPassCount++;

            // Calculate Loss
            totalLoss += _loss.Compute(predicted, sample.Target);

            // Backpropagation -> cộng dồn gradient vào từng neuron
            _network.Backward(_loss.Gradient(predicted, sample.Target));
        }

        // Loss của cả tập là TRUNG BÌNH, nên gradient cũng phải chia cho n.
        // Không chia thì gradient lớn gấp n lần và learning rate mất ý nghĩa.
        var gradients = _network.GetGradients();
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] /= data.Count;
        }

        // Update weights
        var parameters = _network.GetParameters();
        _optimizer.Step(parameters, gradients);
        _network.SetParameters(parameters);

        return totalLoss / data.Count;
    }

    public void Train(TrainingData data, int epochs, Action<EpochSnapshot>? onEpoch = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (epochs < 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double loss = RunEpoch(data);
            onEpoch?.Invoke(new EpochSnapshot(epoch, loss));
        }
    }
}
