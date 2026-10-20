using Ai.Phase3.Activations;
using Ai.Phase3.LinAlg;
using Ai.Phase3.Network;
using Ai.Phase3.Training;

namespace Ai.Phase3.Demos;

/// <summary>
/// Demo 1 — VÍ DỤ SỐ NHỎ NHẤT CÓ THỂ, tính tay được toàn bộ.
///
/// Mạng: 1 đầu vào -> 1 neuron ẩn (Sigmoid) -> 1 neuron ra (Identity)
/// Chỉ 4 tham số: w₁, b₁, w₂, b₂
///
/// Chọn số cố tình để mọi phép tính ra số tròn:
///     x = 1 , y = 0
///     w₁ = 0 , b₁ = 0 , w₂ = 2 , b₂ = 0
/// </summary>
public static class HandCalculationDemo
{
    public static NeuralNetwork BuildTinyNetwork()
    {
        var sigmoid = new Sigmoid();
        var identity = new Identity();

        return new NeuralNetwork(
            new Layer(sigmoid, new Neuron(Vec.Of(0.0), bias: 0.0, sigmoid)),
            new Layer(identity, new Neuron(Vec.Of(2.0), bias: 0.0, identity)));
    }

    public static TrainingData TinyDataset() =>
        new(new TrainingSample(Vec.Of(1.0), Vec.Of(0.0)));

    public static void Run()
    {
        Console.WriteLine("=== DEMO 1: BACKPROPAGATION TÍNH TAY (4 THAM SỐ) ===\n");

        var network = BuildTinyNetwork();
        var data = TinyDataset();
        var loss = new MeanSquaredError();

        var hidden = network.Layers[0].Neurons[0];
        var output = network.Layers[1].Neurons[0];

        Console.WriteLine("  Mạng:  x ──[w₁,b₁ + Sigmoid]──> a₁ ──[w₂,b₂ + Identity]──> ŷ");
        Console.WriteLine($"  Tham số ban đầu: w₁ = {hidden.Weights[0]}, b₁ = {hidden.Bias}, " +
                          $"w₂ = {output.Weights[0]}, b₂ = {output.Bias}");
        Console.WriteLine("  Dữ liệu: x = 1, y = 0\n");

        // ==================================================================
        // BƯỚC 1 — FORWARD
        // ==================================================================
        Console.WriteLine("  ┌─ BƯỚC 1: FORWARD ─────────────────────────────────────");
        var predicted = network.Forward(Vec.Of(1.0));

        Console.WriteLine($"  │  z₁ = w₁·x + b₁ = 0·1 + 0        = {hidden.WeightedSum}");
        Console.WriteLine($"  │  a₁ = sigmoid(z₁) = 1/(1+e⁰)     = {hidden.Output}");
        Console.WriteLine($"  │  z₂ = w₂·a₁ + b₂ = 2·0.5 + 0     = {output.WeightedSum}");
        Console.WriteLine($"  │  ŷ  = Identity(z₂)               = {output.Output}");
        Console.WriteLine("  └────────────────────────────────────────────────────────\n");

        // ==================================================================
        // BƯỚC 2 — LOSS
        // ==================================================================
        Console.WriteLine("  ┌─ BƯỚC 2: TÍNH LOSS ───────────────────────────────────");
        double lossValue = loss.Compute(predicted, Vec.Of(0.0));
        var lossGradient = loss.Gradient(predicted, Vec.Of(0.0));

        Console.WriteLine($"  │  L = (ŷ − y)² = (1 − 0)²         = {lossValue}");
        Console.WriteLine($"  │  ∂L/∂ŷ = 2·(ŷ − y) = 2·(1 − 0)   = {lossGradient[0]}");
        Console.WriteLine("  │  <- đây là mắt xích đầu tiên của chuỗi chain rule");
        Console.WriteLine("  └────────────────────────────────────────────────────────\n");

        // ==================================================================
        // BƯỚC 3 — BACKPROPAGATION
        // ==================================================================
        Console.WriteLine("  ┌─ BƯỚC 3: BACKPROPAGATION (đi ngược) ──────────────────");
        network.ZeroGradients();
        network.Backward(lossGradient);

        Console.WriteLine("  │");
        Console.WriteLine("  │  LAYER RA (Identity, f'(z) = 1):");
        Console.WriteLine($"  │    δ₂ = ∂L/∂ŷ · f'(z₂) = 2 · 1        = {output.Delta}");
        Console.WriteLine($"  │    ∂L/∂w₂ = δ₂ · a₁    = 2 · 0.5      = {output.WeightGradients[0]}");
        Console.WriteLine($"  │    ∂L/∂b₂ = δ₂                        = {output.BiasGradient}");
        Console.WriteLine($"  │    ∂L/∂a₁ = w₂ · δ₂    = 2 · 2        = 4   <- chuyển cho layer trước");
        Console.WriteLine("  │");
        Console.WriteLine("  │  LAYER ẨN (Sigmoid, f'(z) = a·(1−a) = 0.5·0.5 = 0.25):");
        Console.WriteLine($"  │    δ₁ = ∂L/∂a₁ · f'(z₁) = 4 · 0.25    = {hidden.Delta}");
        Console.WriteLine($"  │    ∂L/∂w₁ = δ₁ · x      = 1 · 1       = {hidden.WeightGradients[0]}");
        Console.WriteLine($"  │    ∂L/∂b₁ = δ₁                        = {hidden.BiasGradient}");
        Console.WriteLine("  └────────────────────────────────────────────────────────\n");

        // ==================================================================
        // BƯỚC 4 — ĐỐI CHIẾU VỚI ĐẠO HÀM SỐ
        // ==================================================================
        Console.WriteLine("  ┌─ BƯỚC 4: KIỂM CHỨNG BẰNG ĐẠO HÀM SỐ ──────────────────");
        var check = GradientChecker.Check(network, data, loss);
        string[] names = ["∂L/∂w₁", "∂L/∂b₁", "∂L/∂w₂", "∂L/∂b₂"];
        double[] byHand = [1, 1, 1, 2];

        Console.WriteLine($"  │  {"",-8} {"backprop",12} {"đạo hàm số",14} {"tính tay",10}");
        for (int i = 0; i < names.Length; i++)
        {
            Console.WriteLine($"  │  {names[i],-8} {check.Analytic[i],12:F8} {check.Numerical[i],14:F8} {byHand[i],10}");
        }

        Console.WriteLine($"  │");
        Console.WriteLine($"  │  Lệch tương đối lớn nhất: {check.MaxRelativeError:E3}  ->  {(check.Passed ? "ĐẠT" : "KHÔNG ĐẠT")}");
        Console.WriteLine("  └────────────────────────────────────────────────────────\n");

        // ==================================================================
        // BƯỚC 5 — CẬP NHẬT TRỌNG SỐ
        // ==================================================================
        const double learningRate = 0.1;
        Console.WriteLine($"  ┌─ BƯỚC 5: CẬP NHẬT TRỌNG SỐ (lr = {learningRate}) ──────────────");

        var parameters = network.GetParameters();
        var gradients = check.Analytic;
        new StochasticGradientDescent(learningRate).Step(parameters, gradients);
        network.SetParameters(parameters);

        Console.WriteLine($"  │  w₁ = 0 − 0.1·1 = {parameters[0]:F4}");
        Console.WriteLine($"  │  b₁ = 0 − 0.1·1 = {parameters[1]:F4}");
        Console.WriteLine($"  │  w₂ = 2 − 0.1·1 = {parameters[2]:F4}");
        Console.WriteLine($"  │  b₂ = 0 − 0.1·2 = {parameters[3]:F4}");
        Console.WriteLine("  └────────────────────────────────────────────────────────\n");

        // ==================================================================
        // BƯỚC 6 — FORWARD LẠI
        // ==================================================================
        Console.WriteLine("  ┌─ BƯỚC 6: FORWARD LẠI ─────────────────────────────────");
        var again = network.Forward(Vec.Of(1.0));
        double newLoss = loss.Compute(again, Vec.Of(0.0));

        Console.WriteLine($"  │  z₁ = {hidden.WeightedSum:F6}   a₁ = {hidden.Output:F6}");
        Console.WriteLine($"  │  z₂ = {output.WeightedSum:F6}   ŷ  = {again[0]:F6}");
        Console.WriteLine("  │");
        Console.WriteLine($"  │  Loss:  {lossValue:F6}  ->  {newLoss:F6}   (giảm {(1 - newLoss / lossValue) * 100:F1}%)");
        Console.WriteLine("  │  ŷ đang đi từ 1 về phía target 0 — model đã học được một bước.");
        Console.WriteLine("  └────────────────────────────────────────────────────────");
    }
}
