using Ai.Phase3.Activations;
using Ai.Phase3.LinAlg;
using Ai.Phase3.Network;
using Ai.Phase3.Training;

namespace Ai.Phase3.Tests;

/// <summary>
/// Test quan trọng nhất của Phase 3: backpropagation phải cho ra ĐÚNG
/// cùng gradient với đạo hàm số, trên mọi kiến trúc và mọi activation.
/// </summary>
public class GradientCheckerTests
{
    public static TheoryData<string, int> Architectures => new()
    {
        { "tanh", 1 },
        { "sigmoid", 2 },
        { "relu", 3 },
        { "deep", 4 },
        { "wide", 5 },
        { "multi-output", 6 },
    };

    private static NeuralNetwork BuildArchitecture(string kind, int seed)
    {
        var rng = new Random(seed);

        return kind switch
        {
            "tanh" => new NeuralNetwork(
                new Layer(1, 4, new Tanh(), rng),
                new Layer(4, 1, new Identity(), rng)),

            "sigmoid" => new NeuralNetwork(
                new Layer(1, 5, new Sigmoid(), rng),
                new Layer(5, 1, new Identity(), rng)),

            "relu" => new NeuralNetwork(
                new Layer(1, 6, new ReLU(), rng),
                new Layer(6, 1, new Identity(), rng)),

            "deep" => new NeuralNetwork(
                new Layer(1, 4, new Tanh(), rng),
                new Layer(4, 3, new Sigmoid(), rng),
                new Layer(3, 2, new Tanh(), rng),
                new Layer(2, 1, new Identity(), rng)),

            "wide" => new NeuralNetwork(
                new Layer(1, 20, new Tanh(), rng),
                new Layer(20, 1, new Identity(), rng)),

            "multi-output" => new NeuralNetwork(
                new Layer(1, 4, new Tanh(), rng),
                new Layer(4, 3, new Identity(), rng)),

            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
    }

    private static TrainingData DataFor(NeuralNetwork network)
    {
        if (network.OutputSize == 1) return TrainingData.Square(9);

        var samples = new[] { -0.8, -0.2, 0.4, 0.9 }
            .Select(x => new TrainingSample(Vec.Of(x), Vec.Of(x * x, x, 1 - x)))
            .ToArray();

        return new TrainingData(samples);
    }

    [Theory]
    [MemberData(nameof(Architectures))]
    public void Backprop_TrungVoiDaoHamSo(string kind, int seed)
    {
        var network = BuildArchitecture(kind, seed);
        var data = DataFor(network);

        var check = GradientChecker.Check(network, data, new MeanSquaredError());

        Assert.True(
            check.Passed,
            $"{kind}: lệch tương đối {check.MaxRelativeError:E3}, tuyệt đối {check.MaxAbsoluteError:E3} " +
            $"tại tham số #{check.WorstIndex} " +
            $"(backprop {check.Analytic[check.WorstIndex]:E6}, số {check.Numerical[check.WorstIndex]:E6})");
    }

    [Theory]
    [MemberData(nameof(Architectures))]
    public void Backprop_TrungVoiDaoHamSo_ChoTungThamSoMot(string kind, int seed)
    {
        var network = BuildArchitecture(kind, seed);
        var data = DataFor(network);

        var check = GradientChecker.Check(network, data, new MeanSquaredError());

        for (int i = 0; i < check.Analytic.Length; i++)
        {
            double absolute = Math.Abs(check.Analytic[i] - check.Numerical[i]);
            Assert.True(absolute < 1e-6, $"{kind} tham số #{i}: lệch {absolute:E3}");
        }
    }

    [Fact]
    public void Check_KhongDeLaiDauVetTrenThamSo()
    {
        var network = BuildArchitecture("deep", 4);
        var before = network.GetParameters();

        GradientChecker.Check(network, DataFor(network), new MeanSquaredError());

        Assert.Equal(before, network.GetParameters());
    }

    /// <summary>
    /// Gradient check phải BẮT được bug. Ở đây cố tình bỏ f'(z) ở layer ẩn —
    /// đúng lỗi mà người mới viết backprop hay mắc nhất.
    /// </summary>
    [Fact]
    public void PhatHienDuocBackpropSai()
    {
        var network = BuildArchitecture("tanh", 1);
        var data = DataFor(network);
        var loss = new MeanSquaredError();

        var correct = GradientChecker.ComputeByBackpropagation(network, data, loss);
        var buggy = BackpropWithoutActivationDerivative(network, data, loss);
        var numerical = NumericalGradient.Compute(network, data, loss);

        double correctError = correct.Zip(numerical, (a, n) => Math.Abs(a - n)).Max();
        double buggyError = buggy.Zip(numerical, (a, n) => Math.Abs(a - n)).Max();

        Assert.True(correctError < 1e-8, $"Bản đúng phải khớp, nhưng lệch {correctError:E3}");
        Assert.True(buggyError > 1e-3, $"Bản có bug phải bị phát hiện, nhưng chỉ lệch {buggyError:E3}");
    }

    [Fact]
    public void TieuChiDat_ChapNhanGradientGanKhong()
    {
        // Gradient thật ≈ 0: đạo hàm số trả về nhiễu ~1e-11, lệch tương đối
        // lên tới 1e-3 nhưng lệch tuyệt đối không đáng kể -> vẫn phải ĐẠT.
        var noisy = new GradientCheckResult(
            Analytic: [1e-18],
            Numerical: [1.1e-11],
            MaxAbsoluteError: 1.1e-11,
            MaxRelativeError: 1.1e-3,
            WorstIndex: 0);

        Assert.True(noisy.Passed);
    }

    [Fact]
    public void TieuChiDat_TuChoiSaiSoThatSu()
    {
        var wrong = new GradientCheckResult(
            Analytic: [0.5],
            Numerical: [0.3],
            MaxAbsoluteError: 0.2,
            MaxRelativeError: 0.25,
            WorstIndex: 0);

        Assert.False(wrong.Passed);
    }

    private static double[] BackpropWithoutActivationDerivative(
        NeuralNetwork network, TrainingData data, ILossFunction loss)
    {
        network.ZeroGradients();

        foreach (var sample in data)
        {
            var gradient = loss.Gradient(network.Forward(sample.Input), sample.Target);

            for (int l = network.Layers.Length - 1; l >= 0; l--)
            {
                var layer = network.Layers[l];
                var next = new Vec(layer.InputSize);

                for (int i = 0; i < layer.OutputSize; i++)
                {
                    var neuron = layer.Neurons[i];
                    neuron.Delta = gradient[i];   // BUG: thiếu · f'(z)

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
