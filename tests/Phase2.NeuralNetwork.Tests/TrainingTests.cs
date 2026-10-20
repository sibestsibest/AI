using Ai.Phase2.Activations;
using Ai.Phase2.LinAlg;
using Ai.Phase2.Network;
using Ai.Phase2.Training;

namespace Ai.Phase2.Tests;

public class LossFunctionTests
{
    [Fact]
    public void Mse_MotDauRa_TrungVoiKetQuaTinhTay()
    {
        // (3 − 5)² / 1 = 4
        Assert.Equal(4.0, new MeanSquaredError().Compute(Vec.Of(3), Vec.Of(5)), precision: 10);
    }

    [Fact]
    public void Mse_NhieuDauRa_LayTrungBinh()
    {
        // ((1−3)² + (2−4)²) / 2 = (4 + 4) / 2 = 4
        Assert.Equal(4.0, new MeanSquaredError().Compute(Vec.Of(1, 2), Vec.Of(3, 4)), precision: 10);
    }

    [Fact]
    public void Mse_BangKhongKhiDuDoanDung()
    {
        Assert.Equal(0.0, new MeanSquaredError().Compute(Vec.Of(1, 2, 3), Vec.Of(1, 2, 3)), precision: 12);
    }

    [Fact]
    public void MseGradient_TrungVoiKetQuaTinhTay()
    {
        // ∂L/∂ŷ = (2/m)·(ŷ − y) ; m = 2 -> [1·(1−3), 1·(2−4)] = [−2, −2]
        var g = new MeanSquaredError().Gradient(Vec.Of(1, 2), Vec.Of(3, 4));

        Assert.Equal([-2.0, -2.0], g.ToArray());
    }

    [Fact]
    public void MseGradient_TrungVoiDaoHamSo()
    {
        var mse = new MeanSquaredError();
        var target = Vec.Of(3, -1);
        var predicted = Vec.Of(0.7, 2.2);
        const double h = 1e-6;

        var analytic = mse.Gradient(predicted, target);

        for (int i = 0; i < predicted.Length; i++)
        {
            var plus = predicted.ToArray();
            var minus = predicted.ToArray();
            plus[i] += h;
            minus[i] -= h;

            double numeric = (mse.Compute(new Vec(plus), target) - mse.Compute(new Vec(minus), target)) / (2 * h);

            Assert.True(Math.Abs(numeric - analytic[i]) < 1e-6);
        }
    }

    [Fact]
    public void KhacKichThuoc_ThiNemLoi()
    {
        var mse = new MeanSquaredError();

        Assert.Throws<ArgumentException>(() => mse.Compute(Vec.Of(1), Vec.Of(1, 2)));
        Assert.Throws<ArgumentException>(() => mse.Gradient(Vec.Of(1), Vec.Of(1, 2)));
    }

    [Fact]
    public void AverageLoss_LayTrungBinhTrenCaTapDuLieu()
    {
        // Mạng hằng số ŷ = 0 trên dataset Phase 1:
        // ((0−3)² + (0−5)² + (0−7)² + (0−9)² + (0−11)²)/5 = 285/5 = 57
        var identity = new Identity();
        var network = new NeuralNetwork(
            new Layer(identity, new Neuron(Vec.Of(0.0), bias: 0, identity)));

        double loss = LossEvaluator.AverageLoss(network, TrainingData.Phase1Linear(), new MeanSquaredError());

        Assert.Equal(57.0, loss, precision: 10);
    }
}

public class OptimizerTests
{
    [Fact]
    public void Sgd_TruDiLearningRateNhanGradient()
    {
        var parameters = new double[] { 1.0, 2.0 };
        var gradients = new double[] { 10.0, -10.0 };

        new StochasticGradientDescent(0.1).Step(parameters, gradients);

        Assert.Equal(0.0, parameters[0], precision: 10);   // 1 − 0.1·10
        Assert.Equal(3.0, parameters[1], precision: 10);   // 2 − 0.1·(−10)
    }

    [Fact]
    public void Sgd_GradientBangKhong_ThiKhongDoiGi()
    {
        var parameters = new double[] { 1.5, -2.5 };

        new StochasticGradientDescent(0.9).Step(parameters, new double[2]);

        Assert.Equal([1.5, -2.5], parameters);
    }

    [Fact]
    public void Sgd_LearningRateKhongDuong_ThiNemLoi()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StochasticGradientDescent(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new StochasticGradientDescent(-0.1));
    }

    [Fact]
    public void Sgd_SoLuongKhongKhop_ThiNemLoi()
    {
        Assert.Throws<ArgumentException>(() =>
            new StochasticGradientDescent(0.1).Step(new double[2], new double[3]));
    }
}

public class NumericalGradientTests
{
    /// <summary>
    /// Cầu nối với Phase 1: mạng [1 → 1, Identity] CHÍNH LÀ y = w·x + b.
    /// Tại w = 1, b = 0 trên dataset Phase 1, gradient phải ra đúng
    /// (−28, −8) — con số đã tính tay và test ở Phase 1.
    /// </summary>
    [Fact]
    public void TrungVoiGradientGiaiTichCuaPhase1()
    {
        var identity = new Identity();
        var network = new NeuralNetwork(
            new Layer(identity, new Neuron(Vec.Of(1.0), bias: 0.0, identity)));

        var g = NumericalGradient.Compute(network, TrainingData.Phase1Linear(), new MeanSquaredError());

        Assert.Equal(-28.0, g[0], precision: 5);
        Assert.Equal(-8.0, g[1], precision: 5);
    }

    [Fact]
    public void KhongDeLaiDauVetTrenThamSoCuaMang()
    {
        var rng = new Random(9);
        var network = new NeuralNetwork(
            new Layer(1, 4, new Tanh(), rng),
            new Layer(4, 1, new Identity(), rng));

        var before = network.GetParameters();

        NumericalGradient.Compute(network, TrainingData.Square(7), new MeanSquaredError());

        Assert.Equal(before, network.GetParameters());
    }

    [Fact]
    public void GradientBangKhongTaiDiemToiUu()
    {
        // Mạng đã ở đúng nghiệm w = 2, b = 1 -> không còn gì để học.
        var identity = new Identity();
        var network = new NeuralNetwork(
            new Layer(identity, new Neuron(Vec.Of(2.0), bias: 1.0, identity)));

        var g = NumericalGradient.Compute(network, TrainingData.Phase1Linear(), new MeanSquaredError());

        Assert.All(g, value => Assert.True(Math.Abs(value) < 1e-6));
    }

    [Fact]
    public void TraVeDungMotGradientChoMoiThamSo()
    {
        var rng = new Random(2);
        var network = new NeuralNetwork(
            new Layer(2, 3, new Tanh(), rng),
            new Layer(3, 2, new Identity(), rng));

        var data = new TrainingData(new TrainingSample(Vec.Of(0.5, -0.5), Vec.Of(1, 0)));

        Assert.Equal(network.ParameterCount, NumericalGradient.Compute(network, data, new MeanSquaredError()).Length);
    }
}

public class NumericalTrainerTests
{
    [Fact]
    public void MangTuyenTinh_HocLaiDuocKetQuaCuaPhase1()
    {
        var data = TrainingData.Phase1Linear();
        var loss = new MeanSquaredError();
        var network = new NeuralNetwork(new Layer(1, 1, new Identity(), new Random(42)));

        new NumericalTrainer(network, loss, new StochasticGradientDescent(0.05))
            .Train(data, epochs: 1000);

        var p = network.GetParameters();
        Assert.Equal(2.0, p[0], precision: 4);
        Assert.Equal(1.0, p[1], precision: 4);
    }

    [Fact]
    public void LayerAn_HocDuocQuanHePhiTuyenMaMangTuyenTinhKhongTheHoc()
    {
        var data = TrainingData.Square(pointCount: 11);
        var loss = new MeanSquaredError();
        var sgd = new StochasticGradientDescent(0.3);

        var linear = new NeuralNetwork(new Layer(1, 1, new Identity(), new Random(7)));
        new NumericalTrainer(linear, loss, sgd).Train(data, epochs: 2000);

        // Dùng chung một Random để hai layer không nhận cùng dãy số.
        var rng = new Random(42);
        var deep = new NeuralNetwork(
            new Layer(1, 6, new Tanh(), rng),
            new Layer(6, 1, new Identity(), rng));
        new NumericalTrainer(deep, loss, sgd).Train(data, epochs: 2000);

        double linearLoss = LossEvaluator.AverageLoss(linear, data, loss);
        double deepLoss = LossEvaluator.AverageLoss(deep, data, loss);

        Assert.True(linearLoss > 0.05, $"Mạng tuyến tính phải kẹt, nhưng Loss = {linearLoss}");
        Assert.True(deepLoss < linearLoss / 10, $"Mạng có layer ẩn phải tốt hơn hẳn: {deepLoss} vs {linearLoss}");
    }

    [Fact]
    public void LossGiamDeuQuaTungEpoch()
    {
        var data = TrainingData.Phase1Linear();
        var network = new NeuralNetwork(new Layer(1, 1, new Identity(), new Random(42)));
        var losses = new List<double>();

        new NumericalTrainer(network, new MeanSquaredError(), new StochasticGradientDescent(0.05))
            .Train(data, epochs: 200, onEpoch: s => losses.Add(s.Loss));

        for (int i = 1; i < losses.Count; i++)
        {
            Assert.True(losses[i] <= losses[i - 1], $"Loss tăng tại epoch {i + 1}");
        }
    }

    [Fact]
    public void GoiCallbackDungSoLan_VaDanhSoTu1()
    {
        var network = new NeuralNetwork(new Layer(1, 1, new Identity(), new Random(1)));
        var epochs = new List<int>();

        new NumericalTrainer(network, new MeanSquaredError(), new StochasticGradientDescent(0.01))
            .Train(TrainingData.Phase1Linear(), epochs: 5, onEpoch: s => epochs.Add(s.Epoch));

        Assert.Equal([1, 2, 3, 4, 5], epochs);
    }

    [Fact]
    public void LuuGradientVaoTungNeuronSauKhiTrain()
    {
        var network = new NeuralNetwork(new Layer(1, 1, new Identity(), new Random(1)));

        new NumericalTrainer(network, new MeanSquaredError(), new StochasticGradientDescent(0.01))
            .Train(TrainingData.Phase1Linear(), epochs: 1);

        var neuron = network.Layers[0].Neurons[0];
        Assert.NotEqual(0.0, neuron.WeightGradients[0]);
        Assert.NotEqual(0.0, neuron.BiasGradient);
    }

    [Fact]
    public void KhongEpochNao_ThiMangGiuNguyen()
    {
        var network = new NeuralNetwork(new Layer(1, 1, new Identity(), new Random(1)));
        var before = network.GetParameters();

        new NumericalTrainer(network, new MeanSquaredError(), new StochasticGradientDescent(0.01))
            .Train(TrainingData.Phase1Linear(), epochs: 0);

        Assert.Equal(before, network.GetParameters());
    }
}
