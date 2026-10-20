using System.Diagnostics;
using Ai.Phase3.Activations;
using Ai.Phase3.Network;
using Ai.Phase3.Training;

namespace Ai.Phase3.Demos;

/// <summary>
/// Demo 4 — backprop nhanh nên giờ train được mạng sâu hơn, lâu hơn, tốt hơn.
///
/// Dữ liệu: y = sin(π·x) trên [−1, 1] — cong hơn hẳn x², cần nhiều neuron
/// và nhiều epoch hơn. Với numerical gradient thì bài này đã quá chậm để chạy.
/// </summary>
public static class DeepTrainingDemo
{
    public static void Run()
    {
        Console.WriteLine("\n\n=== DEMO 4: TRAIN MẠNG SÂU BẰNG BACKPROP ===\n");

        // y = sin(π·x)
        int pointCount = 41;
        var pairs = new (double, double)[pointCount];
        for (int i = 0; i < pointCount; i++)
        {
            double x = -1.0 + 2.0 * i / (pointCount - 1);
            pairs[i] = (x, Math.Sin(Math.PI * x));
        }

        var data = TrainingData.FromScalars(pairs);
        var loss = new MeanSquaredError();

        var rng = new Random(42);
        var network = new NeuralNetwork(
            new Layer(1, 16, new Tanh(), rng),
            new Layer(16, 16, new Tanh(), rng),
            new Layer(16, 1, new Identity(), rng));

        Console.WriteLine(network.Describe());

        // Gradient check TRƯỚC khi train — quy trình chuẩn.
        var check = GradientChecker.Check(network, data, loss);
        Console.WriteLine($"\n  Gradient check trước khi train: lệch {check.MaxRelativeError:E3} -> " +
                          $"{(check.Passed ? "ĐẠT, có thể train" : "KHÔNG ĐẠT, dừng lại")}");

        Console.WriteLine($"    lệch tuyệt đối lớn nhất = {check.MaxAbsoluteError:E3}");
        Console.WriteLine($"    tham số tệ nhất #{check.WorstIndex}: backprop = {check.Analytic[check.WorstIndex]:E6}, " +
                          $"đạo hàm số = {check.Numerical[check.WorstIndex]:E6}");

        if (!check.Passed)
        {
            Console.WriteLine("  Backprop sai — không train nữa.");
            return;
        }

        var trainer = new BackpropTrainer(network, loss, new StochasticGradientDescent(0.1));
        const int epochs = 20_000;

        var sw = Stopwatch.StartNew();
        Console.WriteLine($"\n{"Epoch",8} | {"Loss",14}");
        Console.WriteLine(new string('-', 27));

        trainer.Train(data, epochs, s =>
        {
            if (s.Epoch == 1 || s.Epoch % 5000 == 0)
            {
                Console.WriteLine($"{s.Epoch,8} | {s.Loss,14:E6}");
            }
        });

        sw.Stop();

        Console.WriteLine($"\n  {epochs:N0} epoch trong {sw.Elapsed.TotalSeconds:F2} giây " +
                          $"({network.ParameterCount} tham số, {trainer.ForwardPassCount:N0} forward pass).");
        Console.WriteLine($"  Cùng khối lượng này bằng numerical gradient sẽ tốn khoảng " +
                          $"{2L * network.ParameterCount * data.Count * epochs:N0} forward pass.\n");

        Console.WriteLine($"  {"x",7} | {"dự đoán",11} | {"sin(πx)",11} | {"lệch",9}");
        Console.WriteLine("  " + new string('-', 45));
        foreach (double x in new[] { -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0 })
        {
            double predicted = network.PredictScalar(x);
            double actual = Math.Sin(Math.PI * x);
            Console.WriteLine($"  {x,7:F2} | {predicted,11:F6} | {actual,11:F6} | {Math.Abs(predicted - actual),9:E2}");
        }

        // Dữ liệu chưa từng thấy — nằm GIỮA các điểm train.
        Console.WriteLine("\n  Dự đoán tại các điểm KHÔNG có trong tập train:");
        foreach (double x in new[] { -0.6125, 0.1875, 0.6375 })
        {
            double predicted = network.PredictScalar(x);
            double actual = Math.Sin(Math.PI * x);
            Console.WriteLine($"  {x,7:F4} | {predicted,11:F6} | {actual,11:F6} | {Math.Abs(predicted - actual),9:E2}");
        }
    }
}
