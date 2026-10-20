using Ai.Phase3.Activations;
using Ai.Phase3.Network;
using Ai.Phase3.Training;

namespace Ai.Phase3.Demos;

/// <summary>
/// Demo 2 — kiểm chứng backpropagation trên mạng lớn hơn, ngẫu nhiên,
/// nhiều layer và nhiều loại activation khác nhau.
///
/// Ví dụ 4 tham số ở Demo 1 tính tay được nên dễ tin. Nhưng mạng thật có
/// hàng chục tham số đan chéo nhau — chỉ gradient check mới khẳng định
/// được công thức đúng ở mọi vị trí.
/// </summary>
public static class GradientCheckDemo
{
    public static void Run()
    {
        Console.WriteLine("\n\n=== DEMO 2: GRADIENT CHECK TRÊN MẠNG NGẪU NHIÊN ===\n");

        var loss = new MeanSquaredError();
        var data = TrainingData.Square(pointCount: 9);

        (string Name, NeuralNetwork Network)[] architectures =
        [
            ("[1 → 4 (Tanh) → 1]", Build(new Random(1), (1, 4, new Tanh()), (4, 1, new Identity()))),
            ("[1 → 5 (Sigmoid) → 1]", Build(new Random(2), (1, 5, new Sigmoid()), (5, 1, new Identity()))),
            ("[1 → 6 (ReLU) → 1]", Build(new Random(3), (1, 6, new ReLU()), (6, 1, new Identity()))),
            ("[1 → 4 (Tanh) → 3 (Sigmoid) → 1]",
                Build(new Random(4), (1, 4, new Tanh()), (4, 3, new Sigmoid()), (3, 1, new Identity()))),
        ];

        Console.WriteLine($"  {"Kiến trúc",-34} {"tham số",8} {"lệch tuyệt đối",15} {"lệch tương đối",15}  kết quả");
        Console.WriteLine("  " + new string('-', 95));

        foreach (var (name, network) in architectures)
        {
            var check = GradientChecker.Check(network, data, loss);

            Console.WriteLine(
                $"  {name,-34} {network.ParameterCount,8} " +
                $"{check.MaxAbsoluteError,15:E3} {check.MaxRelativeError,15:E3}  " +
                $"{(check.Passed ? "ĐẠT" : "KHÔNG ĐẠT")}");
        }

        // ------------------------------------------------------------------
        // Cho thấy gradient check bắt lỗi thật sự
        // ------------------------------------------------------------------
        Console.WriteLine("\n  --- Gradient check có thực sự bắt được lỗi không? ---");
        Console.WriteLine("  Giả lập một bug kinh điển: quên nhân với đạo hàm của activation.\n");

        var buggy = Build(new Random(1), (1, 4, new Tanh()), (4, 1, new Identity()));
        var correct = GradientChecker.ComputeByBackpropagation(buggy, data, loss);
        var numerical = NumericalGradient.Compute(buggy, data, loss);

        // Bug giả lập: bỏ qua f'(z) ở layer ẩn, tức là coi tanh như Identity.
        var wrong = SimulateMissingActivationDerivative(buggy, data, loss);

        Console.WriteLine($"  {"tham số",10} {"backprop đúng",16} {"backprop lỗi",16} {"đạo hàm số",16}");
        for (int i = 0; i < Math.Min(5, correct.Length); i++)
        {
            Console.WriteLine($"  {i,10} {correct[i],16:F8} {wrong[i],16:F8} {numerical[i],16:F8}");
        }

        double worstWrong = 0;
        for (int i = 0; i < correct.Length; i++)
        {
            worstWrong = Math.Max(worstWrong, Math.Abs(wrong[i] - numerical[i]));
        }

        Console.WriteLine($"\n  Bản đúng lệch  : {Max(correct, numerical):E3}  -> ĐẠT");
        Console.WriteLine($"  Bản có bug lệch: {worstWrong:E3}  -> bị phát hiện ngay");
        Console.WriteLine("\n  Không có gradient check, bug này vẫn làm loss giảm — chỉ là");
        Console.WriteLine("  chậm hơn và kém hơn, và bạn sẽ không bao giờ biết vì sao.");
    }

    private static NeuralNetwork Build(Random rng, params (int In, int Out, IActivation Act)[] spec)
    {
        var layers = spec.Select(s => new Layer(s.In, s.Out, s.Act, rng)).ToArray();
        return new NeuralNetwork(layers);
    }

    private static double Max(double[] a, double[] b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    /// <summary>
    /// Tính lại gradient nhưng cố tình BỎ QUA f'(z) ở các layer ẩn —
    /// đúng lỗi mà người mới viết backprop hay mắc.
    /// </summary>
    private static double[] SimulateMissingActivationDerivative(
        NeuralNetwork network, TrainingData data, ILossFunction loss)
    {
        network.ZeroGradients();

        foreach (var sample in data)
        {
            var predicted = network.Forward(sample.Input);
            var gradient = loss.Gradient(predicted, sample.Target);

            for (int l = network.Layers.Length - 1; l >= 0; l--)
            {
                var layer = network.Layers[l];
                var next = new LinAlg.Vec(layer.InputSize);

                for (int i = 0; i < layer.OutputSize; i++)
                {
                    var neuron = layer.Neurons[i];

                    // BUG: đáng lẽ phải nhân với neuron.Activation.Derivative(neuron.WeightedSum)
                    neuron.Delta = gradient[i];

                    for (int j = 0; j < layer.InputSize; j++)
                    {
                        neuron.WeightGradients[j] += neuron.Delta * neuron.LastInput![j];
                        next[j] += neuron.Weights[j] * neuron.Delta;
                    }

                    neuron.BiasGradient += neuron.Delta;
                }

                gradient = next;
            }
        }

        var gradients = network.GetGradients();
        for (int i = 0; i < gradients.Length; i++) gradients[i] /= data.Count;
        return gradients;
    }
}
