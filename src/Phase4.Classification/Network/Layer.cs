using Ai.Phase4.Activations;
using Ai.Phase4.LinAlg;

namespace Ai.Phase4.Network;

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

    // ------------------------------------------------------------------
    // BACKWARD PASS — phần mới của Phase 3
    // ------------------------------------------------------------------

    /// <summary>
    /// Lan truyền ngược qua một layer.
    ///
    /// Đầu vào : ∂L/∂a — loss thay đổi ra sao theo ĐẦU RA của layer này.
    /// Đầu ra  : ∂L/∂x — loss thay đổi ra sao theo ĐẦU VÀO của layer này,
    ///                   tức là chính ∂L/∂a của layer đứng trước.
    ///
    /// Ba công thức, tất cả đều là chain rule:
    ///
    ///   (1) δᵢ = ∂L/∂aᵢ · f'(zᵢ)                 đi qua activation
    ///   (2) ∂L/∂wᵢⱼ = δᵢ · xⱼ    ,  ∂L/∂bᵢ = δᵢ  gradient của tham số
    ///   (3) ∂L/∂xⱼ = Σᵢ wᵢⱼ · δᵢ                 lan ngược về layer trước
    ///
    /// Công thức (3) chính là Wᵀ · δ. Đó là lý do Phase 2 phải viết Transpose.
    ///
    /// Gradient được CỘNG DỒN (+=) để tổng hợp qua nhiều mẫu dữ liệu —
    /// nhớ gọi <see cref="ZeroGradients"/> trước mỗi epoch.
    /// </summary>
    public Vec Backward(Vec lossGradientWrtOutput)
    {
        ArgumentNullException.ThrowIfNull(lossGradientWrtOutput);

        if (lossGradientWrtOutput.Length != OutputSize)
        {
            throw new ArgumentException(
                $"Layer có {OutputSize} đầu ra nhưng nhận {lossGradientWrtOutput.Length} gradient.",
                nameof(lossGradientWrtOutput));
        }

        var lossGradientWrtInput = new Vec(InputSize);

        for (int i = 0; i < OutputSize; i++)
        {
            var neuron = Neurons[i];

            if (neuron.LastInput is null)
            {
                throw new InvalidOperationException("Phải gọi Forward trước khi gọi Backward.");
            }

            // (1) Đi ngược qua activation để ra δ = ∂L/∂z
            neuron.Delta = lossGradientWrtOutput[i] * neuron.Activation.Derivative(neuron.WeightedSum);

            var input = neuron.LastInput;
            for (int j = 0; j < InputSize; j++)
            {
                // (2) Gradient của trọng số: δ nhân với đầu vào tương ứng.
                neuron.WeightGradients[j] += neuron.Delta * input[j];

                // (3) Phần lỗi mà đầu vào j phải chịu, gộp từ mọi neuron.
                lossGradientWrtInput[j] += neuron.Weights[j] * neuron.Delta;
            }

            // (2) Gradient của bias: bias có "đầu vào" luôn bằng 1, nên bằng đúng δ.
            neuron.BiasGradient += neuron.Delta;
        }

        return lossGradientWrtInput;
    }

    /// <summary>Xoá gradient của mọi neuron trong layer.</summary>
    public void ZeroGradients()
    {
        foreach (var neuron in Neurons)
        {
            neuron.ZeroGradients();
        }
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
