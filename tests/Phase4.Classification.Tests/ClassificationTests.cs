using Ai.Phase4.Activations;
using Ai.Phase4.Demos;
using Ai.Phase4.LinAlg;
using Ai.Phase4.Network;
using Ai.Phase4.Training;

namespace Ai.Phase4.Tests;

public class BinaryCrossEntropyTests
{
    private static readonly BinaryCrossEntropy Bce = new();

    [Theory]
    [InlineData(0.9, 1, 0.10536051565782628)]   // −log(0.9)
    [InlineData(0.5, 1, 0.6931471805599453)]    // −log(0.5)
    [InlineData(0.1, 1, 2.302585092994046)]     // −log(0.1)
    [InlineData(0.1, 0, 0.10536051565782628)]   // −log(1−0.1)
    [InlineData(0.5, 0, 0.6931471805599453)]
    public void Compute_TrungVoiCongThucLogarit(double predicted, double target, double expected)
    {
        Assert.Equal(expected, Bce.Compute(Vec.Of(predicted), Vec.Of(target)), precision: 10);
    }

    [Fact]
    public void PhatNangHonKhiDoanCangSai()
    {
        double[] predictions = [0.9, 0.5, 0.1, 0.01];
        var losses = predictions.Select(p => Bce.Compute(Vec.Of(p), Vec.Of(1))).ToArray();

        for (int i = 1; i < losses.Length; i++)
        {
            Assert.True(losses[i] > losses[i - 1]);
        }

        // Sai hoàn toàn bị phạt gấp hơn 40 lần so với đoán tốt.
        Assert.True(losses[^1] / losses[0] > 40);
    }

    [Fact]
    public void GanKhongKhiDuDoanGanHoanHao()
    {
        Assert.True(Bce.Compute(Vec.Of(0.999999), Vec.Of(1)) < 1e-5);
        Assert.True(Bce.Compute(Vec.Of(0.000001), Vec.Of(0)) < 1e-5);
    }

    [Fact]
    public void KhongTraVeVoCucKhiDuDoanBangKhongHoacMot()
    {
        // log(0) = −∞ sẽ làm hỏng mọi phép tính sau đó -> phải bị chặn.
        Assert.True(double.IsFinite(Bce.Compute(Vec.Of(0.0), Vec.Of(1))));
        Assert.True(double.IsFinite(Bce.Compute(Vec.Of(1.0), Vec.Of(0))));
        Assert.True(double.IsFinite(Bce.Gradient(Vec.Of(0.0), Vec.Of(1))[0]));
    }

    [Fact]
    public void Gradient_TrungVoiDaoHamSo()
    {
        const double h = 1e-7;

        foreach (double p in new[] { 0.1, 0.3, 0.5, 0.7, 0.9 })
        {
            foreach (double y in new double[] { 0, 1 })
            {
                double numeric = (Bce.Compute(Vec.Of(p + h), Vec.Of(y)) -
                                  Bce.Compute(Vec.Of(p - h), Vec.Of(y))) / (2 * h);
                double analytic = Bce.Gradient(Vec.Of(p), Vec.Of(y))[0];

                Assert.True(Math.Abs(numeric - analytic) < 1e-4,
                    $"p={p}, y={y}: giải tích {analytic}, số {numeric}");
            }
        }
    }

    /// <summary>
    /// Điểm đẹp nhất của cặp Sigmoid + BCE: mẫu số ŷ(1−ŷ) của gradient loss
    /// triệt tiêu đúng f'(z) = ŷ(1−ŷ) của sigmoid, để lại δ = ŷ − y.
    /// Nhờ vậy gradient KHÔNG BAO GIỜ biến mất, kể cả khi sigmoid bão hoà.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void SigmoidCongBce_ChoDeltaBangDuDoanTruTarget(double target)
    {
        var sigmoid = new Sigmoid();
        var network = new NeuralNetwork(
            new Layer(sigmoid, new Neuron(Vec.Of(1.3), bias: -0.4, sigmoid)));

        var predicted = network.Forward(Vec.Of(2.0));
        network.ZeroGradients();
        network.Backward(Bce.Gradient(predicted, Vec.Of(target)));

        double delta = network.Layers[0].Neurons[0].Delta;

        Assert.Equal(predicted[0] - target, delta, precision: 10);
    }

    /// <summary>
    /// Đối chứng: cùng tình huống nhưng dùng MSE thì delta bị nhân thêm
    /// ŷ(1−ŷ), và khi sigmoid bão hoà thì gradient gần như biến mất.
    /// </summary>
    [Fact]
    public void Mse_ThiGradientBienMatKhiSigmoidBaoHoa()
    {
        var sigmoid = new Sigmoid();

        // z = 10 -> ŷ ≈ 0.99995, nhưng target là 0: sai hoàn toàn.
        var network = new NeuralNetwork(
            new Layer(sigmoid, new Neuron(Vec.Of(10.0), bias: 0, sigmoid)));

        var predicted = network.Forward(Vec.Of(1.0));

        network.ZeroGradients();
        network.Backward(new MeanSquaredError().Gradient(predicted, Vec.Of(0.0)));
        double mseDelta = Math.Abs(network.Layers[0].Neurons[0].Delta);

        network.Forward(Vec.Of(1.0));
        network.ZeroGradients();
        network.Backward(Bce.Gradient(predicted, Vec.Of(0.0)));
        double bceDelta = Math.Abs(network.Layers[0].Neurons[0].Delta);

        Assert.True(mseDelta < 1e-4, $"MSE delta phải teo lại, nhưng = {mseDelta}");
        Assert.True(bceDelta > 0.99, $"BCE delta phải giữ nguyên độ lớn, nhưng = {bceDelta}");
    }
}

public class ClassifierTests
{
    [Theory]
    [InlineData(0.0, 0)]
    [InlineData(0.49999, 0)]
    [InlineData(0.5, 1)]
    [InlineData(0.9, 1)]
    public void Classify_SoVoiNguong(double probability, int expected)
    {
        Assert.Equal(expected, Classifier.Classify(probability));
    }

    [Fact]
    public void Accuracy_DemDungTiLeMauPhanLoaiDung()
    {
        // Mạng hằng số luôn trả về sigmoid(bias lớn) ≈ 1 -> đúng 2/4 mẫu XOR.
        var sigmoid = new Sigmoid();
        var alwaysOne = new NeuralNetwork(
            new Layer(sigmoid, new Neuron(Vec.Of(0.0, 0.0), bias: 10, sigmoid)));

        Assert.Equal(0.5, Classifier.Accuracy(alwaysOne, LogicDatasets.Xor()), precision: 10);
        Assert.Equal(0.25, Classifier.Accuracy(alwaysOne, LogicDatasets.And()), precision: 10);
        Assert.Equal(0.75, Classifier.Accuracy(alwaysOne, LogicDatasets.Or()), precision: 10);
    }
}

public class LogicDatasetTests
{
    [Fact]
    public void Xor_DungBangChanLy()
    {
        var data = LogicDatasets.Xor();
        double[] expected = [0, 1, 1, 0];

        Assert.Equal(4, data.Count);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(expected[i], data[i].Target[0]);
        }
    }

    [Fact]
    public void And_Or_DungBangChanLy()
    {
        Assert.Equal([0.0, 0, 0, 1], LogicDatasets.And().Select(s => s.Target[0]).ToArray());
        Assert.Equal([0.0, 1, 1, 1], LogicDatasets.Or().Select(s => s.Target[0]).ToArray());
    }
}

public class XorLearningTests
{
    private const int Epochs = 10_000;
    private const double LearningRate = 0.5;

    /// <summary>Mạng có layer ẩn phải học được XOR từ nhiều khởi tạo khác nhau.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(42)]
    [InlineData(2024)]
    public void MangCoLayerAn_HocDuocXor(int seed)
    {
        var data = LogicDatasets.Xor();
        var network = XorDemo.BuildXorNetwork(seed);

        new BackpropTrainer(network, new BinaryCrossEntropy(), new StochasticGradientDescent(LearningRate))
            .Train(data, Epochs);

        Assert.Equal(1.0, Classifier.Accuracy(network, data));

        // Không chỉ đúng phía ngưỡng — phải TỰ TIN đúng.
        Assert.True(network.Forward(Vec.Of(0, 0))[0] < 0.05);
        Assert.True(network.Forward(Vec.Of(0, 1))[0] > 0.95);
        Assert.True(network.Forward(Vec.Of(1, 0))[0] > 0.95);
        Assert.True(network.Forward(Vec.Of(1, 1))[0] < 0.05);
    }

    /// <summary>
    /// Perceptron (không layer ẩn) KHÔNG thể học XOR dù train bao lâu —
    /// giới hạn Minsky &amp; Papert 1969. Nhưng nó học được AND và OR.
    /// </summary>
    [Fact]
    public void Perceptron_KhongTheHocXor()
    {
        var data = LogicDatasets.Xor();
        var perceptron = new NeuralNetwork(new Layer(2, 1, new Sigmoid(), new Random(1)));

        new BackpropTrainer(perceptron, new BinaryCrossEntropy(), new StochasticGradientDescent(LearningRate))
            .Train(data, 20_000);

        Assert.True(Classifier.Accuracy(perceptron, data) <= 0.5);
    }

    [Theory]
    [InlineData("and")]
    [InlineData("or")]
    public void Perceptron_HocDuocCongKhaPhanTuyenTinh(string gate)
    {
        var data = gate == "and" ? LogicDatasets.And() : LogicDatasets.Or();
        var perceptron = new NeuralNetwork(new Layer(2, 1, new Sigmoid(), new Random(1)));

        new BackpropTrainer(perceptron, new BinaryCrossEntropy(), new StochasticGradientDescent(LearningRate))
            .Train(data, 20_000);

        Assert.Equal(1.0, Classifier.Accuracy(perceptron, data));
    }

    [Theory]
    [InlineData("and")]
    [InlineData("or")]
    [InlineData("xor")]
    public void MangCoLayerAn_HocDuocCaBaCongLogic(string gate)
    {
        var data = gate switch
        {
            "and" => LogicDatasets.And(),
            "or" => LogicDatasets.Or(),
            _ => LogicDatasets.Xor(),
        };

        var network = XorDemo.BuildXorNetwork(42);
        new BackpropTrainer(network, new BinaryCrossEntropy(), new StochasticGradientDescent(LearningRate))
            .Train(data, Epochs);

        Assert.Equal(1.0, Classifier.Accuracy(network, data));
    }

    [Fact]
    public void GradientCheck_DatVoiSigmoidVaBce()
    {
        var network = XorDemo.BuildXorNetwork(42);

        var check = GradientChecker.Check(network, LogicDatasets.Xor(), new BinaryCrossEntropy());

        Assert.True(check.Passed, $"lệch {check.MaxAbsoluteError:E3}");
    }

    [Fact]
    public void LayerAn_TaoRaBieuDienKhacNhauChoTungDauVao()
    {
        var network = XorDemo.BuildXorNetwork(42);
        new BackpropTrainer(network, new BinaryCrossEntropy(), new StochasticGradientDescent(LearningRate))
            .Train(LogicDatasets.Xor(), Epochs);

        var representations = LogicDatasets.Inputs
            .Select(i =>
            {
                network.Forward(Vec.Of(i.A, i.B));
                return network.Layers[0].LastOutput!.ToArray();
            })
            .ToArray();

        // Bốn đầu vào phải cho bốn biểu diễn ẩn phân biệt được.
        for (int i = 0; i < 4; i++)
        {
            for (int j = i + 1; j < 4; j++)
            {
                double distance = representations[i]
                    .Zip(representations[j], (a, b) => (a - b) * (a - b)).Sum();

                Assert.True(distance > 0.01, $"Biểu diễn ẩn của mẫu {i} và {j} trùng nhau.");
            }
        }
    }
}
