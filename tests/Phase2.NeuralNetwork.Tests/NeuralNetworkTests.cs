using Ai.Phase2.Activations;
using Ai.Phase2.Demos;
using Ai.Phase2.LinAlg;
using Ai.Phase2.Network;

namespace Ai.Phase2.Tests;

public class NeuralNetworkTests
{
    /// <summary>
    /// Mạng 1 → 3 → 1 tính tay trong phần giải thích:
    ///   hidden (ReLU):  z = [0.5·2+0, −1·2+0.5, 2·2−1] = [1, −1.5, 3]
    ///                   a = [1, 0, 3]
    ///   output:         z = 1·1 + 2·0 + (−0.5)·3 + 0.25 = −0.25
    /// </summary>
    [Fact]
    public void Forward_TrungVoiKetQuaTinhTay()
    {
        var network = ForwardTraceDemo.BuildHandCraftedNetwork();

        var output = network.Forward(Vec.Of(2));

        Assert.Equal(-0.25, output[0], precision: 10);
    }

    [Fact]
    public void Forward_GiaTriTrungGianCuaLayerAn_DungNhuTinhTay()
    {
        var network = ForwardTraceDemo.BuildHandCraftedNetwork();

        network.Forward(Vec.Of(2));

        Assert.Equal([1.0, 0.0, 3.0], network.Layers[0].LastOutput!.ToArray());
    }

    [Fact]
    public void Forward_DayDuLieuQuaTungLayerTheoThuTu()
    {
        // Hai layer identity, mỗi layer nhân đôi -> tổng cộng nhân 4.
        var identity = new Identity();
        var network = new NeuralNetwork(
            new Layer(identity, new Neuron(Vec.Of(2.0), bias: 0, identity)),
            new Layer(identity, new Neuron(Vec.Of(2.0), bias: 0, identity)));

        Assert.Equal(12.0, network.PredictScalar(3), precision: 10);
    }

    [Fact]
    public void Constructor_LayerKhongKhopKichThuoc_ThiNemLoi()
    {
        var rng = new Random(1);
        var first = new Layer(1, 3, new Tanh(), rng);
        var mismatched = new Layer(5, 1, new Identity(), rng);   // cần 5, chỉ nhận được 3

        var ex = Assert.Throws<ArgumentException>(() => new NeuralNetwork(first, mismatched));
        Assert.Contains("layer 0", ex.Message);
    }

    [Fact]
    public void Constructor_KhongCoLayer_ThiNemLoi()
    {
        Assert.Throws<ArgumentException>(() => new NeuralNetwork());
    }

    [Fact]
    public void ParameterCount_DemDungSoThamSo()
    {
        var rng = new Random(1);
        var network = new NeuralNetwork(
            new Layer(1, 8, new Tanh(), rng),        // 8·1 + 8 = 16
            new Layer(8, 1, new Identity(), rng));   // 1·8 + 1 = 9

        Assert.Equal(25, network.ParameterCount);
        Assert.Equal(25, network.GetParameters().Length);
    }

    [Fact]
    public void GetSetParameters_KhepKin()
    {
        var rng = new Random(11);
        var network = new NeuralNetwork(
            new Layer(2, 3, new Tanh(), rng),
            new Layer(3, 2, new Identity(), rng));

        var original = network.GetParameters();
        var input = Vec.Of(0.4, -0.9);
        var before = network.Forward(input).ToArray();

        // Phá tham số rồi khôi phục — đầu ra phải quay về y hệt.
        network.SetParameters(new double[network.ParameterCount]);
        Assert.NotEqual(before, network.Forward(input).ToArray());

        network.SetParameters(original);
        Assert.Equal(before, network.Forward(input).ToArray());
    }

    [Fact]
    public void SetParameters_ThayDoiThatSuDauRa()
    {
        var identity = new Identity();
        var network = new NeuralNetwork(
            new Layer(identity, new Neuron(Vec.Of(1.0), bias: 0, identity)));

        // Thứ tự tham số: [trọng số..., bias] cho từng neuron.
        network.SetParameters([3.0, 10.0]);

        Assert.Equal(16.0, network.PredictScalar(2), precision: 10);   // 3·2 + 10
    }

    [Fact]
    public void SetParameters_SaiSoLuong_ThiNemLoi()
    {
        var network = new NeuralNetwork(new Layer(1, 2, new Identity(), new Random(1)));

        Assert.Throws<ArgumentException>(() => network.SetParameters([1.0]));
    }

    [Fact]
    public void GetSetGradients_KhepKin_VaLuuVaoTungNeuron()
    {
        var rng = new Random(5);
        var network = new NeuralNetwork(
            new Layer(2, 2, new Tanh(), rng),
            new Layer(2, 1, new Identity(), rng));

        var gradients = Enumerable.Range(1, network.ParameterCount)
            .Select(i => (double)i)
            .ToArray();

        network.SetGradients(gradients);

        Assert.Equal(gradients, network.GetGradients());

        // Neuron đầu tiên có 2 đầu vào -> nhận gradient 1, 2 và bias 3.
        var firstNeuron = network.Layers[0].Neurons[0];
        Assert.Equal(1.0, firstNeuron.WeightGradients[0]);
        Assert.Equal(2.0, firstNeuron.WeightGradients[1]);
        Assert.Equal(3.0, firstNeuron.BiasGradient);
    }

    [Fact]
    public void Predict_GiongForward()
    {
        var network = ForwardTraceDemo.BuildHandCraftedNetwork();
        var input = Vec.Of(1.7);

        Assert.Equal(network.Forward(input).ToArray(), network.Predict(input).ToArray());
    }

    [Fact]
    public void Describe_NeuKienTrucVaSoThamSo()
    {
        var rng = new Random(1);
        var network = new NeuralNetwork(
            new Layer(1, 3, new ReLU(), rng),
            new Layer(3, 1, new Identity(), rng));

        string text = network.Describe();

        Assert.Contains("10 tham số", text);
        Assert.Contains("ReLU", text);
        Assert.Contains("Identity", text);
    }
}
