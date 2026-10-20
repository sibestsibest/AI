using Ai.Phase2.Activations;
using Ai.Phase2.LinAlg;

namespace Ai.Phase2.Network;

/// <summary>
/// Một neuron: đơn vị tính toán nhỏ nhất của mạng.
///
///     z = w · x + b        (tổng có trọng số — giống hệt Phase 1)
///     a = f(z)             (activation — phần mới của Phase 2)
///
/// Phase 1 là neuron có đúng 1 đầu vào và f = Identity. Phase 2 chỉ mở rộng
/// hai điểm: nhiều đầu vào, và có activation.
/// </summary>
public sealed class Neuron
{
    /// <summary>Trọng số — mỗi đầu vào một trọng số. Đây là "tri thức" của neuron.</summary>
    public Vec Weights { get; }

    /// <summary>Độ lệch — cho phép neuron kích hoạt ngay cả khi mọi đầu vào bằng 0.</summary>
    public double Bias { get; set; }

    public IActivation Activation { get; }

    /// <summary>Số đầu vào mà neuron này nhận.</summary>
    public int InputSize => Weights.Length;

    // --- Trạng thái ghi lại từ lần Forward gần nhất ---------------------
    // Phải lưu lại vì Phase 3 (backpropagation) cần đúng những giá trị này
    // để tính đạo hàm. Không lưu thì phải chạy forward lại, rất phí.

    /// <summary>Đầu vào của lần forward gần nhất.</summary>
    public Vec? LastInput { get; private set; }

    /// <summary>z — tổng có trọng số trước khi qua activation.</summary>
    public double WeightedSum { get; private set; }

    /// <summary>a = f(z) — đầu ra của neuron.</summary>
    public double Output { get; private set; }

    // --- Chỗ chứa gradient ---------------------------------------------
    // Phase 2 điền các ô này bằng numerical gradient; Phase 3 sẽ điền bằng
    // backpropagation. Cấu trúc dữ liệu giống nhau, chỉ khác cách tính.

    /// <summary>∂L/∂w — mỗi trọng số một giá trị.</summary>
    public Vec WeightGradients { get; }

    /// <summary>∂L/∂b</summary>
    public double BiasGradient { get; set; }

    /// <summary>
    /// δ = ∂L/∂z — "lỗi" quy về tổng có trọng số của neuron này.
    /// Đây là đại lượng trung tâm của backpropagation ở Phase 3.
    /// </summary>
    public double Delta { get; set; }

    /// <summary>Khởi tạo ngẫu nhiên theo Xavier (xem <see cref="Layer"/>).</summary>
    public Neuron(int inputSize, IActivation activation, Random rng, double weightLimit)
    {
        ArgumentNullException.ThrowIfNull(activation);
        ArgumentNullException.ThrowIfNull(rng);
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));

        Activation = activation;
        Weights = new Vec(inputSize);
        WeightGradients = new Vec(inputSize);

        for (int i = 0; i < inputSize; i++)
        {
            Weights[i] = (rng.NextDouble() * 2 - 1) * weightLimit;
        }

        // Bias khởi tạo bằng 0: không có lý do gì để đoán lệch về một phía.
        Bias = 0;
    }

    /// <summary>Khởi tạo với trọng số xác định — dùng cho unit test và demo tính tay.</summary>
    public Neuron(Vec weights, double bias, IActivation activation)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(activation);

        Weights = weights;
        Bias = bias;
        Activation = activation;
        WeightGradients = new Vec(weights.Length);
    }

    /// <summary>
    /// Forward pass của một neuron:
    ///     z = Σ wᵢ·xᵢ + b
    ///     a = f(z)
    /// </summary>
    public double Forward(Vec input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.Length != InputSize)
        {
            throw new ArgumentException(
                $"Neuron cần {InputSize} đầu vào nhưng nhận {input.Length}.", nameof(input));
        }

        LastInput = input;
        WeightedSum = Weights.Dot(input) + Bias;
        Output = Activation.Forward(WeightedSum);

        return Output;
    }

    public override string ToString() =>
        $"Neuron(w={Weights}, b={Bias:F4}, f={Activation.Name})";
}
