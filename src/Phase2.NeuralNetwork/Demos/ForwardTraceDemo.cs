using Ai.Phase2.Activations;
using Ai.Phase2.LinAlg;
using Ai.Phase2.Network;

namespace Ai.Phase2.Demos;

/// <summary>
/// Demo 2 — mạng 1 → 3 → 1 với trọng số ĐẶT SẴN, để soi từng con số
/// của forward pass và đối chiếu với phép tính tay.
///
///     Input x
///        ↓
///     Neuron 1   Neuron 2   Neuron 3      (hidden, ReLU)
///        ↓          ↓          ↓
///          Output neuron (Identity)
/// </summary>
public static class ForwardTraceDemo
{
    /// <summary>Dựng đúng mạng dùng trong phần giải thích và trong unit test.</summary>
    public static NeuralNetwork BuildHandCraftedNetwork()
    {
        var relu = new ReLU();

        var hidden = new Layer(relu,
            new Neuron(Vec.Of(0.5), bias: 0.0, relu),    // n1
            new Neuron(Vec.Of(-1.0), bias: 0.5, relu),   // n2
            new Neuron(Vec.Of(2.0), bias: -1.0, relu));  // n3

        var identity = new Identity();
        var output = new Layer(identity,
            new Neuron(Vec.Of(1.0, 2.0, -0.5), bias: 0.25, identity));

        return new NeuralNetwork(hidden, output);
    }

    public static void Run()
    {
        Console.WriteLine("\n\n=== DEMO 2: FORWARD PASS TÍNH TAY (mạng 1 → 3 → 1) ===\n");

        var network = BuildHandCraftedNetwork();
        Console.WriteLine(network.Describe());

        const double x = 2.0;
        Console.WriteLine($"\n  Đầu vào: x = {x}\n");

        var hidden = network.Layers[0];
        var outputLayer = network.Layers[1];

        var result = network.Forward(Vec.Of(x));

        Console.WriteLine("  --- Layer ẩn (3 neuron, ReLU) ---");
        for (int i = 0; i < hidden.OutputSize; i++)
        {
            var n = hidden.Neurons[i];
            Console.WriteLine(
                $"    Neuron {i + 1}: z = {n.Weights[0]:F2}·{x} + ({n.Bias:F2}) = {n.WeightedSum,6:F2}" +
                $"   ->  ReLU(z) = {n.Output,5:F2}");
        }

        Console.WriteLine($"\n    Vector đầu ra layer ẩn: {hidden.LastOutput}");

        var outNeuron = outputLayer.Neurons[0];
        Console.WriteLine("\n  --- Layer đầu ra (1 neuron, Identity) ---");
        Console.WriteLine($"    w = {outNeuron.Weights} , b = {outNeuron.Bias:F2}");
        Console.WriteLine(
            $"    z = ({outNeuron.Weights[0]:F2}·{hidden.LastOutput![0]:F2}) + " +
            $"({outNeuron.Weights[1]:F2}·{hidden.LastOutput[1]:F2}) + " +
            $"({outNeuron.Weights[2]:F2}·{hidden.LastOutput[2]:F2}) + ({outNeuron.Bias:F2})" +
            $" = {outNeuron.WeightedSum:F4}");
        Console.WriteLine($"    Identity(z) = {outNeuron.Output:F4}");

        Console.WriteLine($"\n  ĐẦU RA MẠNG: {result}    (tính tay: −0.25)");

        // ------------------------------------------------------------------
        // Chứng minh dạng neuron-by-neuron và dạng ma trận là MỘT
        // ------------------------------------------------------------------
        Console.WriteLine("\n  --- Cùng phép tính, viết dưới dạng ma trận ---");
        var w = hidden.WeightMatrix();
        var bias = hidden.BiasVector();

        Console.WriteLine("    W (3x1), mỗi hàng là trọng số của một neuron:");
        Console.WriteLine($"{w}");
        Console.WriteLine($"    b = {bias}");

        var z = w.Multiply(Vec.Of(x)).Add(bias);
        var activated = z.Map(new ReLU().Forward);

        Console.WriteLine($"    z = W·x + b      = {z}");
        Console.WriteLine($"    a = ReLU(z)      = {activated}");
        Console.WriteLine($"    Layer.Forward(x) = {hidden.LastOutput}   <- trùng khớp");

        // ------------------------------------------------------------------
        // Vì sao cần activation
        // ------------------------------------------------------------------
        Console.WriteLine("\n  --- Vì sao phải có activation? ---");
        Console.WriteLine("    Neuron 2 có z = −1.5 < 0, ReLU cắt thành 0: neuron này 'tắt' với x = 2.");
        Console.WriteLine("    Nếu bỏ activation, W₂·(W₁·x + b₁) + b₂ rút gọn lại thành W'·x + b'");
        Console.WriteLine("    — tức là vẫn chỉ là MỘT đường thẳng, xếp bao nhiêu layer cũng vô nghĩa.");
    }
}
