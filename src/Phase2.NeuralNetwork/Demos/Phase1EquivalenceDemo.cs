using Ai.Phase2.Activations;
using Ai.Phase2.LinAlg;
using Ai.Phase2.Network;
using Ai.Phase2.Training;

namespace Ai.Phase2.Demos;

/// <summary>
/// Demo 3 — chứng minh Phase 2 TỔNG QUÁT HOÁ Phase 1 chứ không mâu thuẫn.
///
/// Một mạng gồm đúng 1 layer, 1 neuron, 1 đầu vào, activation Identity
/// chính là model y = w·x + b của Phase 1. Train nó bằng bộ máy mới
/// (Loss + Optimizer + numerical gradient) phải ra lại w ≈ 2, b ≈ 1.
/// </summary>
public static class Phase1EquivalenceDemo
{
    public static void Run()
    {
        Console.WriteLine("\n\n=== DEMO 3: MẠNG NEURAL TỔNG QUÁT HOÁ PHASE 1 ===\n");

        var data = TrainingData.Phase1Linear();
        var loss = new MeanSquaredError();

        // --- Bước A: gradient của mạng có trùng công thức giải tích Phase 1? ---
        var identity = new Identity();
        var probe = new NeuralNetwork(
            new Layer(identity, new Neuron(Vec.Of(1.0), bias: 0.0, identity)));

        var gradients = NumericalGradient.Compute(probe, data, loss);

        Console.WriteLine("  Mạng [1 -> 1, Identity] tại w = 1, b = 0:");
        Console.WriteLine($"    ∂L/∂w = {gradients[0],10:F6}   Phase 1 tính tay: −28");
        Console.WriteLine($"    ∂L/∂b = {gradients[1],10:F6}   Phase 1 tính tay: −8");
        Console.WriteLine("    -> Hai phase hoàn toàn nhất quán.\n");

        // --- Bước B: train từ khởi tạo ngẫu nhiên ---
        var rng = new Random(42);
        var network = new NeuralNetwork(new Layer(1, 1, identity, rng));
        var optimizer = new StochasticGradientDescent(learningRate: 0.05);
        var trainer = new NumericalTrainer(network, loss, optimizer);

        var start = network.GetParameters();
        Console.WriteLine($"  Khởi tạo ngẫu nhiên: w = {start[0]:F6}, b = {start[1]:F6}");
        Console.WriteLine($"  Loss ban đầu       : {LossEvaluator.AverageLoss(network, data, loss):F6}\n");

        Console.WriteLine($"{"Epoch",6} | {"Loss",14} | {"Weight",10} | {"Bias",10}");
        Console.WriteLine(new string('-', 50));

        trainer.Train(data, epochs: 1000, onEpoch: snapshot =>
        {
            if (snapshot.Epoch != 1 && snapshot.Epoch % 200 != 0) return;

            var p = network.GetParameters();
            Console.WriteLine($"{snapshot.Epoch,6} | {snapshot.Loss,14:F8} | {p[0],10:F6} | {p[1],10:F6}");
        });

        var final = network.GetParameters();
        Console.WriteLine($"\n  Model học được: y = {final[0]:F6}·x + {final[1]:F6}");
        Console.WriteLine($"  Loss cuối     : {LossEvaluator.AverageLoss(network, data, loss):E3}");
        Console.WriteLine($"  Dự đoán x = 10: {network.PredictScalar(10):F6}   (kỳ vọng 21)");
        Console.WriteLine("\n  Lưu ý: KHÔNG có dòng backpropagation nào. Gradient lấy từ");
        Console.WriteLine("  numerical differentiation — Phase 3 sẽ thay bằng backprop.");
    }
}
