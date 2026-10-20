using System.Diagnostics;
using Ai.Phase3.Activations;
using Ai.Phase3.Network;
using Ai.Phase3.Training;

namespace Ai.Phase3.Demos;

/// <summary>
/// Demo 3 — hai cách lấy gradient, CÙNG kết quả, khác nhau về chi phí.
///
/// Đây là lý do duy nhất backpropagation tồn tại. Nó không chính xác hơn
/// numerical gradient (thực ra numerical gradient còn là thước đo để kiểm tra
/// nó). Nó chỉ NHANH HƠN — nhưng nhanh hơn đủ nhiều để biến AI từ bất khả thi
/// thành khả thi.
/// </summary>
public static class SpeedComparisonDemo
{
    public static void Run()
    {
        Console.WriteLine("\n\n=== DEMO 3: BACKPROP vs NUMERICAL GRADIENT ===\n");

        var data = TrainingData.Square(pointCount: 21);
        var loss = new MeanSquaredError();
        const int epochs = 3000;
        const double learningRate = 0.3;

        // Hai mạng khởi tạo GIỐNG HỆT nhau để so sánh công bằng.
        var numericalNet = BuildNetwork();
        var backpropNet = BuildNetwork();
        backpropNet.SetParameters(numericalNet.GetParameters());

        Console.WriteLine(backpropNet.Describe());
        Console.WriteLine($"\n  Train {epochs} epoch, learning rate {learningRate}, {data.Count} mẫu.\n");

        // --- Cách 1: numerical gradient (Phase 2) ---
        NumericalGradient.ResetCounter();
        var sw1 = Stopwatch.StartNew();
        new NumericalTrainer(numericalNet, loss, new StochasticGradientDescent(learningRate))
            .Train(data, epochs);
        sw1.Stop();
        long numericalForwards = NumericalGradient.ForwardPassCount;

        // --- Cách 2: backpropagation (Phase 3) ---
        var backpropTrainer = new BackpropTrainer(backpropNet, loss, new StochasticGradientDescent(learningRate));
        var sw2 = Stopwatch.StartNew();
        backpropTrainer.Train(data, epochs);
        sw2.Stop();

        double numericalLoss = LossEvaluator.AverageLoss(numericalNet, data, loss);
        double backpropLoss = LossEvaluator.AverageLoss(backpropNet, data, loss);

        Console.WriteLine($"  {"",-22} {"Numerical",16} {"Backpropagation",18}");
        Console.WriteLine("  " + new string('-', 58));
        Console.WriteLine($"  {"Loss cuối",-22} {numericalLoss,16:E4} {backpropLoss,18:E4}");
        Console.WriteLine($"  {"Forward pass",-22} {numericalForwards,16:N0} {backpropTrainer.ForwardPassCount,18:N0}");
        Console.WriteLine($"  {"Thời gian (giây)",-22} {sw1.Elapsed.TotalSeconds,16:F2} {sw2.Elapsed.TotalSeconds,18:F2}");
        Console.WriteLine($"  {"Nhanh hơn",-22} {"—",16} {sw1.Elapsed.TotalSeconds / sw2.Elapsed.TotalSeconds,17:F1}x");

        // --- Hai cách có hội tụ về cùng một chỗ không? ---
        var a = numericalNet.GetParameters();
        var b = backpropNet.GetParameters();
        double maxDiff = a.Zip(b, (x, y) => Math.Abs(x - y)).Max();

        Console.WriteLine($"\n  Lệch tham số lớn nhất giữa hai mạng: {maxDiff:E3}");
        Console.WriteLine("  Cùng xuất phát, cùng learning rate, cùng dữ liệu -> cùng đích đến.");
        Console.WriteLine("  Backprop không cho kết quả TỐT HƠN, chỉ cho kết quả NHANH HƠN.");

        Console.WriteLine($"\n  Với mạng {backpropNet.ParameterCount} tham số, numerical cần 2 × {backpropNet.ParameterCount} " +
                          $"= {2 * backpropNet.ParameterCount} lần forward mỗi bước.");
        Console.WriteLine("  Khoảng cách này giãn tuyến tính theo số tham số: mạng 1 triệu tham số");
        Console.WriteLine("  thì numerical chậm hơn khoảng 1 triệu lần. Backprop thì vẫn thế.");
    }

    private static NeuralNetwork BuildNetwork()
    {
        var rng = new Random(42);
        return new NeuralNetwork(
            new Layer(1, 8, new Tanh(), rng),
            new Layer(8, 1, new Identity(), rng));
    }
}
