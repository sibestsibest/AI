using Ai.Phase3.Activations;
using Ai.Phase3.LinAlg;
using Ai.Phase3.Network;
using Ai.Phase3.Training;

namespace Ai.Phase3.Tests;

public class BackpropTrainerTests
{
    private static NeuralNetwork LinearNetwork(int seed) =>
        new(new Layer(1, 1, new Identity(), new Random(seed)));

    private static NeuralNetwork DeepNetwork(int seed)
    {
        var rng = new Random(seed);
        return new NeuralNetwork(
            new Layer(1, 8, new Tanh(), rng),
            new Layer(8, 1, new Identity(), rng));
    }

    /// <summary>Liên tục với Phase 1: cùng dataset, cùng kết quả w ≈ 2, b ≈ 1.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(42)]
    [InlineData(2024)]
    public void MangTuyenTinh_HocLaiKetQuaCuaPhase1(int seed)
    {
        var network = LinearNetwork(seed);

        new BackpropTrainer(network, new MeanSquaredError(), new StochasticGradientDescent(0.05))
            .Train(TrainingData.Phase1Linear(), epochs: 1000);

        var p = network.GetParameters();
        Assert.Equal(2.0, p[0], precision: 4);
        Assert.Equal(1.0, p[1], precision: 4);
    }

    /// <summary>
    /// Backprop và numerical gradient chỉ khác cách TÍNH gradient. Sau MỘT
    /// epoch, hai bên phải cho ra tham số gần như trùng khít.
    /// </summary>
    [Fact]
    public void MotEpoch_ChoKetQuaGiongHetNumericalTrainer()
    {
        var data = TrainingData.Square(11);
        var loss = new MeanSquaredError();

        var viaNumerical = DeepNetwork(42);
        var viaBackprop = DeepNetwork(42);
        viaBackprop.SetParameters(viaNumerical.GetParameters());

        new NumericalTrainer(viaNumerical, loss, new StochasticGradientDescent(0.3)).Train(data, 1);
        new BackpropTrainer(viaBackprop, loss, new StochasticGradientDescent(0.3)).Train(data, 1);

        var a = viaNumerical.GetParameters();
        var b = viaBackprop.GetParameters();

        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(Math.Abs(a[i] - b[i]) < 1e-9, $"Tham số #{i}: {a[i]} vs {b[i]}");
        }
    }

    /// <summary>
    /// Qua nhiều epoch, hai bên vẫn bám nhau nhưng KHÔNG còn trùng khít:
    /// mỗi bước numerical gradient lệch khoảng 1e-11 so với đạo hàm thật, và
    /// sai số đó tích luỹ dần. Sau 300 epoch lệch cỡ 1e-6 — vẫn là cùng một
    /// quỹ đạo học, chỉ khác ở chữ số cuối. Đây là hạn chế của đạo hàm số,
    /// không phải của backpropagation.
    /// </summary>
    [Fact]
    public void NhieuEpoch_VanBamSatNumericalTrainer()
    {
        var data = TrainingData.Square(11);
        var loss = new MeanSquaredError();
        const int epochs = 300;

        var viaNumerical = DeepNetwork(42);
        var viaBackprop = DeepNetwork(42);
        viaBackprop.SetParameters(viaNumerical.GetParameters());

        new NumericalTrainer(viaNumerical, loss, new StochasticGradientDescent(0.3)).Train(data, epochs);
        new BackpropTrainer(viaBackprop, loss, new StochasticGradientDescent(0.3)).Train(data, epochs);

        var a = viaNumerical.GetParameters();
        var b = viaBackprop.GetParameters();

        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(Math.Abs(a[i] - b[i]) < 1e-4, $"Tham số #{i}: {a[i]} vs {b[i]}");
        }

        // Và quan trọng hơn: cùng chất lượng model.
        Assert.Equal(
            LossEvaluator.AverageLoss(viaNumerical, data, loss),
            LossEvaluator.AverageLoss(viaBackprop, data, loss),
            precision: 8);
    }

    [Fact]
    public void RunEpoch_TraVeLossDoTruocKhiCapNhat()
    {
        var data = TrainingData.Phase1Linear();
        var loss = new MeanSquaredError();
        var network = LinearNetwork(42);

        double expected = LossEvaluator.AverageLoss(network, data, loss);
        double reported = new BackpropTrainer(network, loss, new StochasticGradientDescent(0.05))
            .RunEpoch(data);

        Assert.Equal(expected, reported, precision: 12);
    }

    [Fact]
    public void LossGiamDeuQuaTungEpoch()
    {
        var network = DeepNetwork(42);
        var losses = new List<double>();

        new BackpropTrainer(network, new MeanSquaredError(), new StochasticGradientDescent(0.1))
            .Train(TrainingData.Square(11), epochs: 500, onEpoch: s => losses.Add(s.Loss));

        for (int i = 1; i < losses.Count; i++)
        {
            Assert.True(losses[i] <= losses[i - 1], $"Loss tăng tại epoch {i + 1}");
        }
    }

    [Fact]
    public void HocDuocQuanHePhiTuyen()
    {
        var data = TrainingData.Square(21);
        var loss = new MeanSquaredError();
        var network = DeepNetwork(42);

        new BackpropTrainer(network, loss, new StochasticGradientDescent(0.3))
            .Train(data, epochs: 3000);

        Assert.True(LossEvaluator.AverageLoss(network, data, loss) < 1e-3);
    }

    [Fact]
    public void GradientDuocChiaTrungBinhTheoSoMau()
    {
        // Dataset gồm cùng một mẫu lặp lại 4 lần phải cho gradient bằng
        // hệt dataset chỉ có 1 mẫu — nếu quên chia cho n thì sẽ gấp 4.
        var sample = new TrainingSample(Vec.Of(1.0), Vec.Of(0.0));
        var loss = new MeanSquaredError();

        var one = GradientChecker.ComputeByBackpropagation(
            Demos.HandCalculationDemo.BuildTinyNetwork(), new TrainingData(sample), loss);

        var four = GradientChecker.ComputeByBackpropagation(
            Demos.HandCalculationDemo.BuildTinyNetwork(),
            new TrainingData(sample, sample, sample, sample), loss);

        Assert.Equal(one, four);
    }

    [Fact]
    public void DemDungSoForwardPass()
    {
        var data = TrainingData.Phase1Linear();   // 5 mẫu
        var trainer = new BackpropTrainer(LinearNetwork(1), new MeanSquaredError(),
            new StochasticGradientDescent(0.01));

        trainer.Train(data, epochs: 10);

        // Backprop: đúng 1 forward mỗi mẫu mỗi epoch, không phụ thuộc số tham số.
        Assert.Equal(50, trainer.ForwardPassCount);
    }

    [Fact]
    public void GoiCallbackDungSoLan_VaDanhSoTu1()
    {
        var epochs = new List<int>();

        new BackpropTrainer(LinearNetwork(1), new MeanSquaredError(), new StochasticGradientDescent(0.01))
            .Train(TrainingData.Phase1Linear(), epochs: 5, onEpoch: s => epochs.Add(s.Epoch));

        Assert.Equal([1, 2, 3, 4, 5], epochs);
    }

    [Fact]
    public void KhongEpochNao_ThiMangGiuNguyen()
    {
        var network = LinearNetwork(1);
        var before = network.GetParameters();

        new BackpropTrainer(network, new MeanSquaredError(), new StochasticGradientDescent(0.01))
            .Train(TrainingData.Phase1Linear(), epochs: 0);

        Assert.Equal(before, network.GetParameters());
    }

    /// <summary>
    /// Mỗi epoch phải tự xoá gradient. Nếu không, train 2 lần liên tiếp 100
    /// epoch sẽ khác train 1 lần 200 epoch.
    /// </summary>
    [Fact]
    public void TrainerTuXoaGradientMoiEpoch()
    {
        var data = TrainingData.Square(11);
        var loss = new MeanSquaredError();

        var inOneGo = DeepNetwork(42);
        new BackpropTrainer(inOneGo, loss, new StochasticGradientDescent(0.1)).Train(data, 200);

        var inTwoParts = DeepNetwork(42);
        var trainer = new BackpropTrainer(inTwoParts, loss, new StochasticGradientDescent(0.1));
        trainer.Train(data, 100);
        trainer.Train(data, 100);

        Assert.Equal(inOneGo.GetParameters(), inTwoParts.GetParameters());
    }

    [Fact]
    public void ModelDaTrain_TongQuatHoaSangDiemChuaTungThay()
    {
        var network = LinearNetwork(42);

        new BackpropTrainer(network, new MeanSquaredError(), new StochasticGradientDescent(0.05))
            .Train(TrainingData.Phase1Linear(), epochs: 1000);

        foreach (double x in new double[] { 6, 10, 100, -3 })
        {
            Assert.Equal(2 * x + 1, network.PredictScalar(x), precision: 3);
        }
    }
}
