using Ai.Phase2.Activations;
using Ai.Phase2.LinAlg;

namespace Ai.Phase2.Network;

/// <summary>
/// Một lớp (layer): nhiều neuron cùng nhận chung một vector đầu vào,
/// mỗi neuron sinh ra một số, gộp lại thành vector đầu ra.
///
///     input  (độ dài InputSize)
///       │  mọi neuron đều thấy TOÀN BỘ input  (fully connected)
///       ├──> Neuron 0 ──> a₀
///       ├──> Neuron 1 ──> a₁
///       └──> Neuron 2 ──> a₂
///     output = [a₀, a₁, a₂]
///
/// Dạng ma trận tương đương:  output = f(W · x + b)
/// với W là ma trận OutputSize × InputSize, mỗi HÀNG là trọng số một neuron.
/// </summary>
public sealed class Layer
{
    public Neuron[] Neurons { get; }

    public IActivation Activation { get; }

    public int InputSize { get; }

    public int OutputSize => Neurons.Length;

    /// <summary>Đầu ra của lần forward gần nhất.</summary>
    public Vec? LastOutput { get; private set; }

    /// <summary>
    /// Khởi tạo ngẫu nhiên theo Xavier/Glorot:
    ///
    ///     limit = √( 6 / (fanIn + fanOut) )      w ~ U(−limit, +limit)
    ///
    /// Vì sao không random bừa trong [−1, 1]? Vì z = Σ wᵢ·xᵢ cộng dồn fanIn số
    /// hạng. Layer càng rộng, z càng dễ văng ra xa, rơi vào vùng bão hoà của
    /// sigmoid/tanh nơi đạo hàm ≈ 0 và mạng ngừng học. Chia theo kích thước
    /// layer giữ cho z ở quanh gốc toạ độ.
    /// </summary>
    public Layer(int inputSize, int neuronCount, IActivation activation, Random rng)
    {
        ArgumentNullException.ThrowIfNull(activation);
        ArgumentNullException.ThrowIfNull(rng);
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (neuronCount <= 0) throw new ArgumentOutOfRangeException(nameof(neuronCount));

        InputSize = inputSize;
        Activation = activation;

        double limit = Math.Sqrt(6.0 / (inputSize + neuronCount));

        Neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
        {
            Neurons[i] = new Neuron(inputSize, activation, rng, limit);
        }
    }

    /// <summary>Tạo layer từ các neuron có trọng số xác định — dùng cho test/demo.</summary>
    public Layer(IActivation activation, params Neuron[] neurons)
    {
        ArgumentNullException.ThrowIfNull(activation);
        ArgumentNullException.ThrowIfNull(neurons);
        if (neurons.Length == 0) throw new ArgumentException("Layer phải có ít nhất 1 neuron.", nameof(neurons));

        int inputSize = neurons[0].InputSize;
        if (neurons.Any(n => n.InputSize != inputSize))
        {
            throw new ArgumentException("Mọi neuron trong layer phải có cùng số đầu vào.", nameof(neurons));
        }

        Activation = activation;
        Neurons = neurons;
        InputSize = inputSize;
    }

    /// <summary>
    /// Forward pass của cả layer: chạy từng neuron rồi gom kết quả.
    /// </summary>
    public Vec Forward(Vec input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.Length != InputSize)
        {
            throw new ArgumentException(
                $"Layer cần {InputSize} đầu vào nhưng nhận {input.Length}.", nameof(input));
        }

        var output = new Vec(OutputSize);
        for (int i = 0; i < OutputSize; i++)
        {
            output[i] = Neurons[i].Forward(input);
        }

        LastOutput = output;
        return output;
    }

    /// <summary>
    /// Gom trọng số của mọi neuron thành ma trận W (OutputSize × InputSize).
    /// Chỉ để minh hoạ và kiểm chứng: <c>Forward(x)</c> phải cho đúng
    /// cùng kết quả với <c>f(W·x + b)</c>. Có unit test khẳng định điều này.
    /// </summary>
    public Mat WeightMatrix()
    {
        var w = new Mat(OutputSize, InputSize);
        for (int r = 0; r < OutputSize; r++)
        {
            for (int c = 0; c < InputSize; c++)
            {
                w[r, c] = Neurons[r].Weights[c];
            }
        }

        return w;
    }

    public Vec BiasVector()
    {
        var b = new Vec(OutputSize);
        for (int i = 0; i < OutputSize; i++)
        {
            b[i] = Neurons[i].Bias;
        }

        return b;
    }

    /// <summary>Số tham số học được: OutputSize · InputSize trọng số + OutputSize bias.</summary>
    public int ParameterCount => OutputSize * InputSize + OutputSize;

    public override string ToString() =>
        $"Layer({InputSize} -> {OutputSize}, {Activation.Name}, {ParameterCount} tham số)";
}
