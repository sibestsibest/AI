using System.Diagnostics;
using Ai.Phase2.Activations;
using Ai.Phase2.Network;
using Ai.Phase2.Training;

namespace Ai.Phase2.Demos;

/// <summary>
/// Demo 4 — layer ẩn đem lại năng lực mà Phase 1 không thể có.
///
/// Dữ liệu: y = x² trên [−1, 1]. Model đường thẳng của Phase 1 chắc chắn
/// thất bại (parabol không phải đường thẳng). Mạng [1 → 8 (tanh) → 1] thì học được.
/// </summary>
public static class NonlinearFitDemo
{
    public static void Run()
    {
        Console.WriteLine("\n\n=== DEMO 4: HỌC QUAN HỆ PHI TUYẾN y = x² ===\n");

        var data = TrainingData.Square(pointCount: 21);
        var loss = new MeanSquaredError();

        // --- Đối chứng: model tuyến tính kiểu Phase 1 ---
        var linear = new NeuralNetwork(new Layer(1, 1, new Identity(), new Random(7)));
        new NumericalTrainer(linear, loss, new StochasticGradientDescent(0.1))
            .Train(data, epochs: 1500);

        double linearLoss = LossEvaluator.AverageLoss(linear, data, loss);
        Console.WriteLine($"  [Đối chứng] Mạng tuyến tính [1 -> 1]   : Loss = {linearLoss:F6}");
        Console.WriteLine("              Đã train hết sức nhưng vẫn kẹt — đường thẳng không cong được.\n");

        // --- Mạng có layer ẩn ---
        var rng = new Random(42);
        var network = new NeuralNetwork(
            new Layer(1, 8, new Tanh(), rng),
            new Layer(8, 1, new Identity(), rng));

        Console.WriteLine(network.Describe());

        var trainer = new NumericalTrainer(network, loss, new StochasticGradientDescent(0.3));

        NumericalGradient.ResetCounter();
        var stopwatch = Stopwatch.StartNew();

        Console.WriteLine($"\n{"Epoch",6} | {"Loss",14}");
        Console.WriteLine(new string('-', 25));

        trainer.Train(data, epochs: 3000, onEpoch: s =>
        {
            if (s.Epoch == 1 || s.Epoch % 500 == 0)
            {
                Console.WriteLine($"{s.Epoch,6} | {s.Loss,14:F8}");
            }
        });

        stopwatch.Stop();

        double finalLoss = LossEvaluator.AverageLoss(network, data, loss);
        Console.WriteLine($"\n  Mạng có layer ẩn [1 -> 8 -> 1]        : Loss = {finalLoss:E3}");
        Console.WriteLine($"  Tốt hơn model tuyến tính khoảng {linearLoss / finalLoss:F0} lần.\n");

        Console.WriteLine($"  {"x",6} | {"dự đoán",10} | {"đúng (x²)",10} | {"lệch",9}");
        Console.WriteLine(new string('-', 45));
        foreach (double x in new[] { -1.0, -0.6, -0.25, 0.0, 0.25, 0.6, 1.0 })
        {
            double predicted = network.PredictScalar(x);
            Console.WriteLine($"  {x,6:F2} | {predicted,10:F6} | {x * x,10:F6} | {Math.Abs(predicted - x * x),9:F6}");
        }

        // --- Vì sao Phase 3 phải tồn tại ---
        Console.WriteLine("\n  --- Chi phí của numerical gradient ---");
        Console.WriteLine($"    Tham số của mạng      : {network.ParameterCount}");
        Console.WriteLine($"    Forward pass đã chạy  : {NumericalGradient.ForwardPassCount:N0}");
        Console.WriteLine($"    Thời gian             : {stopwatch.Elapsed.TotalSeconds:F2} giây");
        Console.WriteLine($"    Mỗi bước học tốn 2 × {network.ParameterCount} = {2 * network.ParameterCount} lần forward.");
        Console.WriteLine("    Backpropagation (Phase 3) cho ĐÚNG kết quả này với 1 forward + 1 backward,");
        Console.WriteLine("    không phụ thuộc số tham số. Đó là lý do nó tồn tại.");
    }
}
