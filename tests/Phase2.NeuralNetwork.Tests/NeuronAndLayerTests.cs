using Ai.Phase2.Activations;
using Ai.Phase2.LinAlg;
using Ai.Phase2.Network;

namespace Ai.Phase2.Tests;

public class NeuronTests
{
    /// <summary>
    /// w = [0.5, −1] , b = 0.25 , x = [2, 3]
    /// z = 0.5·2 + (−1)·3 + 0.25 = 1 − 3 + 0.25 = −1.75
    /// </summary>
    [Fact]
    public void Forward_TinhDungTongCoTrongSo()
    {
        var neuron = new Neuron(Vec.Of(0.5, -1.0), bias: 0.25, new Identity());

        double output = neuron.Forward(Vec.Of(2, 3));

        Assert.Equal(-1.75, neuron.WeightedSum, precision: 10);
        Assert.Equal(-1.75, output, precision: 10);
    }

    [Fact]
    public void Forward_ApActivationLenTongCoTrongSo()
    {
        // Cùng z = −1.75 nhưng qua ReLU thì output bị cắt thành 0.
        var neuron = new Neuron(Vec.Of(0.5, -1.0), bias: 0.25, new ReLU());

        double output = neuron.Forward(Vec.Of(2, 3));

        Assert.Equal(-1.75, neuron.WeightedSum, precision: 10);
        Assert.Equal(0.0, output, precision: 10);
    }

    [Fact]
    public void Forward_LuuLaiTrangThaiChoPhase3()
    {
        var neuron = new Neuron(Vec.Of(1.0, 1.0), bias: 0, new Sigmoid());
        var input = Vec.Of(0.3, -0.3);

        neuron.Forward(input);

        Assert.Same(input, neuron.LastInput);
        Assert.Equal(0.0, neuron.WeightedSum, precision: 10);
        Assert.Equal(0.5, neuron.Output, precision: 10);
    }

    [Fact]
    public void Forward_SaiSoDauVao_ThiNemLoi()
    {
        var neuron = new Neuron(Vec.Of(1.0, 2.0), bias: 0, new Identity());

        Assert.Throws<ArgumentException>(() => neuron.Forward(Vec.Of(1)));
    }

    [Fact]
    public void KhoiTao_CoChoChuaGradientDungKichThuoc()
    {
        var neuron = new Neuron(Vec.Of(1.0, 2.0, 3.0), bias: 0, new Identity());

        Assert.Equal(3, neuron.WeightGradients.Length);
        Assert.Equal(0.0, neuron.BiasGradient);
        Assert.Equal(0.0, neuron.Delta);
    }

    [Fact]
    public void KhoiTaoNgauNhien_BiasBangKhong_TrongSoTrongGioiHan()
    {
        var rng = new Random(1);

        for (int i = 0; i < 20; i++)
        {
            var neuron = new Neuron(4, new Tanh(), rng, weightLimit: 0.5);

            Assert.Equal(0.0, neuron.Bias);
            for (int w = 0; w < neuron.InputSize; w++)
            {
                Assert.InRange(neuron.Weights[w], -0.5, 0.5);
            }
        }
    }
}

public class LayerTests
{
    [Fact]
    public void Forward_ChayTatCaNeuronTrenCungMotDauVao()
    {
        var identity = new Identity();
        var layer = new Layer(identity,
            new Neuron(Vec.Of(1.0, 0.0), bias: 0, identity),   // lấy x₀
            new Neuron(Vec.Of(0.0, 1.0), bias: 0, identity),   // lấy x₁
            new Neuron(Vec.Of(1.0, 1.0), bias: 5, identity));  // tổng + 5

        var output = layer.Forward(Vec.Of(3, 7));

        Assert.Equal([3.0, 7.0, 15.0], output.ToArray());
    }

    /// <summary>
    /// Điểm mấu chốt: cách viết "từng neuron một" và cách viết ma trận
    /// f(W·x + b) phải cho CÙNG một kết quả. Nếu không thì một trong hai sai.
    /// </summary>
    [Fact]
    public void Forward_TuongDuongVoiDangMaTran()
    {
        var rng = new Random(7);
        var layer = new Layer(inputSize: 4, neuronCount: 3, new Tanh(), rng);
        var input = Vec.Of(0.5, -1.2, 0.3, 2.0);

        var viaNeurons = layer.Forward(input);

        var viaMatrix = layer.WeightMatrix()
            .Multiply(input)
            .Add(layer.BiasVector())
            .Map(layer.Activation.Forward);

        for (int i = 0; i < viaNeurons.Length; i++)
        {
            Assert.Equal(viaNeurons[i], viaMatrix[i], precision: 12);
        }
    }

    [Fact]
    public void WeightMatrix_MoiHangLaMotNeuron()
    {
        var identity = new Identity();
        var layer = new Layer(identity,
            new Neuron(Vec.Of(1.0, 2.0), bias: 0, identity),
            new Neuron(Vec.Of(3.0, 4.0), bias: 0, identity));

        var w = layer.WeightMatrix();

        Assert.Equal(2, w.Rows);
        Assert.Equal(2, w.Cols);
        Assert.Equal(1.0, w[0, 0]);
        Assert.Equal(2.0, w[0, 1]);
        Assert.Equal(3.0, w[1, 0]);
        Assert.Equal(4.0, w[1, 1]);
    }

    [Fact]
    public void ParameterCount_BangSoTrongSoCongSoBias()
    {
        var layer = new Layer(inputSize: 4, neuronCount: 3, new Tanh(), new Random(1));

        // 3 neuron × 4 trọng số + 3 bias = 15
        Assert.Equal(15, layer.ParameterCount);
    }

    [Fact]
    public void XavierInit_GioiHanTheoKichThuocLayer()
    {
        // limit = √(6 / (fanIn + fanOut)) = √(6/100) ≈ 0.2449
        var layer = new Layer(inputSize: 50, neuronCount: 50, new Tanh(), new Random(3));
        double limit = Math.Sqrt(6.0 / 100);

        foreach (var neuron in layer.Neurons)
        {
            for (int i = 0; i < neuron.InputSize; i++)
            {
                Assert.InRange(neuron.Weights[i], -limit, limit);
            }
        }
    }

    [Fact]
    public void Forward_SaiSoDauVao_ThiNemLoi()
    {
        var layer = new Layer(inputSize: 3, neuronCount: 2, new ReLU(), new Random(1));

        Assert.Throws<ArgumentException>(() => layer.Forward(Vec.Of(1, 2)));
    }

    [Fact]
    public void NeuronKhacSoDauVao_ThiNemLoi()
    {
        var identity = new Identity();

        Assert.Throws<ArgumentException>(() => new Layer(identity,
            new Neuron(Vec.Of(1.0), bias: 0, identity),
            new Neuron(Vec.Of(1.0, 2.0), bias: 0, identity)));
    }
}
