using Ai.Phase4.Activations;
using Ai.Phase4.LinAlg;
using Ai.Phase4.Network;
using Ai.Phase4.Training;

namespace Ai.Phase4.Demos;

/// <summary>
/// Phase 4 — mạng tự học XOR. Không hard-code logic XOR ở bất kỳ đâu:
/// mạng chỉ thấy 4 cặp (đầu vào, đáp án) và phải tự tìm ra quy tắc.
/// </summary>
public static class XorDemo
{
    public static NeuralNetwork BuildXorNetwork(int seed)
    {
        var rng = new Random(seed);
        return new NeuralNetwork(
            new Layer(2, 4, new Tanh(), rng),        // layer ẩn: tạo đặc trưng mới
            new Layer(4, 1, new Sigmoid(), rng));    // layer ra: xác suất thuộc lớp 1
    }

    public static void Run()
    {
        Console.WriteLine("=== DEMO 1: VÌ SAO XOR KHÓ ===\n");
        ShowLinearFailure();

        Console.WriteLine("\n\n=== DEMO 2: MẠNG CÓ LAYER ẨN HỌC ĐƯỢC XOR ===\n");
        var network = TrainXor();

        Console.WriteLine("\n\n=== DEMO 3: LAYER ẨN ĐÃ HỌC RA ĐẶC TRƯNG GÌ? ===\n");
        InspectHiddenLayer(network);

        Console.WriteLine("\n\n=== DEMO 4: AND / OR / XOR — CÙNG MỘT MẠNG ===\n");
        CompareLogicGates();
    }

    /// <summary>Mạng không có layer ẩn (perceptron) không thể học XOR.</summary>
    private static void ShowLinearFailure()
    {
        var loss = new BinaryCrossEntropy();

        foreach (var (name, data) in new[]
                 {
                     ("AND", LogicDatasets.And()),
                     ("OR", LogicDatasets.Or()),
                     ("XOR", LogicDatasets.Xor()),
                 })
        {
            var perceptron = new NeuralNetwork(new Layer(2, 1, new Sigmoid(), new Random(1)));
            new BackpropTrainer(perceptron, loss, new StochasticGradientDescent(0.5))
                .Train(data, epochs: 20_000);

            Console.WriteLine($"  Perceptron (không layer ẩn) học {name,-3}: " +
                              $"accuracy = {Classifier.Accuracy(perceptron, data):P0}, " +
                              $"loss = {LossEvaluator.AverageLoss(perceptron, data, loss):F4}");
        }

        Console.WriteLine("\n  AND và OR: tách được bằng MỘT đường thẳng -> perceptron làm được.");
        Console.WriteLine("  XOR: hai lớp nằm chéo nhau -> không đường thẳng nào tách nổi.");
        Console.WriteLine("  Đây là giới hạn Minsky & Papert chỉ ra năm 1969.");
    }

    private static NeuralNetwork TrainXor()
    {
        var data = LogicDatasets.Xor();
        var loss = new BinaryCrossEntropy();
        var network = BuildXorNetwork(seed: 42);

        Console.WriteLine(network.Describe());

        var check = GradientChecker.Check(network, data, loss);
        Console.WriteLine($"\n  Gradient check (sigmoid + BCE): lệch {check.MaxAbsoluteError:E3} -> " +
                          $"{(check.Passed ? "ĐẠT" : "KHÔNG ĐẠT")}\n");

        var trainer = new BackpropTrainer(network, loss, new StochasticGradientDescent(0.5));

        Console.WriteLine($"{"Epoch",7} | {"Loss",12} | {"(0,0)",8} | {"(0,1)",8} | {"(1,0)",8} | {"(1,1)",8} | acc");
        Console.WriteLine(new string('-', 76));

        trainer.Train(data, epochs: 10_000, onEpoch: s =>
        {
            if (s.Epoch != 1 && s.Epoch % 1000 != 0) return;

            var p = LogicDatasets.Inputs
                .Select(i => network.Forward(Vec.Of(i.A, i.B))[0])
                .ToArray();

            Console.WriteLine($"{s.Epoch,7} | {s.Loss,12:F8} | {p[0],8:F4} | {p[1],8:F4} | " +
                              $"{p[2],8:F4} | {p[3],8:F4} | {Classifier.Accuracy(network, data):P0}");
        });

        Console.WriteLine($"\n  {"A",3} {"B",3} | {"A XOR B",8} | {"xác suất",10} | {"quyết định",11} | kết quả");
        Console.WriteLine("  " + new string('-', 60));

        double[] expected = [0, 1, 1, 0];
        for (int i = 0; i < 4; i++)
        {
            var (a, b) = LogicDatasets.Inputs[i];
            double probability = network.Forward(Vec.Of(a, b))[0];
            int decision = Classifier.Classify(probability);

            Console.WriteLine($"  {a,3} {b,3} | {expected[i],8} | {probability,10:F6} | {decision,11} | " +
                              $"{(decision == expected[i] ? "đúng" : "SAI")}");
        }

        Console.WriteLine($"\n  Accuracy: {Classifier.Accuracy(network, data):P0}");
        Console.WriteLine("  Mạng chưa bao giờ được cho biết quy tắc XOR — nó tự suy ra từ 4 ví dụ.");

        return network;
    }

    /// <summary>
    /// Layer ẩn học được gì? Nó biến không gian đầu vào thành một không gian
    /// mới, trong đó XOR TRỞ THÀNH khả phân tuyến tính.
    /// </summary>
    private static void InspectHiddenLayer(NeuralNetwork network)
    {
        var hidden = network.Layers[0];

        Console.WriteLine("  Giá trị 4 neuron ẩn ứng với từng đầu vào:\n");
        Console.WriteLine($"  {"đầu vào",10} | {"h₀",8} {"h₁",8} {"h₂",8} {"h₃",8} | {"đầu ra",8}");
        Console.WriteLine("  " + new string('-', 58));

        foreach (var (a, b) in LogicDatasets.Inputs)
        {
            double output = network.Forward(Vec.Of(a, b))[0];
            var h = hidden.LastOutput!;

            Console.WriteLine($"  {$"({a},{b})",10} | {h[0],8:F4} {h[1],8:F4} {h[2],8:F4} {h[3],8:F4} | {output,8:F4}");
        }

        Console.WriteLine("\n  Mỗi neuron ẩn là một đường thẳng cắt mặt phẳng đầu vào.");
        Console.WriteLine("  Layer ra tổ hợp các đường đó lại — tổ hợp của nhiều đường thẳng");
        Console.WriteLine("  thì không còn là đường thẳng nữa, và XOR giải được.");
        Console.WriteLine("\n  Trọng số layer ẩn (mỗi hàng một neuron, 2 cột = 2 đầu vào):");
        Console.WriteLine(hidden.WeightMatrix());
        Console.WriteLine($"  bias = {hidden.BiasVector()}");
    }

    private static void CompareLogicGates()
    {
        var loss = new BinaryCrossEntropy();

        Console.WriteLine($"  {"Cổng",6} | {"loss cuối",12} | accuracy | {"(0,0)",7} {"(0,1)",7} {"(1,0)",7} {"(1,1)",7}");
        Console.WriteLine("  " + new string('-', 70));

        foreach (var (name, data) in new[]
                 {
                     ("AND", LogicDatasets.And()),
                     ("OR", LogicDatasets.Or()),
                     ("XOR", LogicDatasets.Xor()),
                 })
        {
            var network = BuildXorNetwork(seed: 42);
            new BackpropTrainer(network, loss, new StochasticGradientDescent(0.5))
                .Train(data, epochs: 10_000);

            var p = LogicDatasets.Inputs.Select(i => network.Forward(Vec.Of(i.A, i.B))[0]).ToArray();

            Console.WriteLine($"  {name,6} | {LossEvaluator.AverageLoss(network, data, loss),12:E3} | " +
                              $"{Classifier.Accuracy(network, data),8:P0} | " +
                              $"{p[0],7:F4} {p[1],7:F4} {p[2],7:F4} {p[3],7:F4}");
        }

        Console.WriteLine("\n  Cùng một kiến trúc [2 → 4 → 1] học được cả ba cổng logic.");
        Console.WriteLine("  Không có dòng code nào nói cho mạng biết AND, OR hay XOR là gì.");
    }
}
