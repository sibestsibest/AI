using Ai.Phase3.Activations;
using Ai.Phase3.Demos;
using Ai.Phase3.LinAlg;
using Ai.Phase3.Network;
using Ai.Phase3.Training;

namespace Ai.Phase3.Tests;

/// <summary>
/// Kiểm tra backpropagation trên ví dụ 4 tham số tính tay được hoàn toàn.
///
///   Mạng:  x ──[w₁=0, b₁=0, Sigmoid]──> a₁ ──[w₂=2, b₂=0, Identity]──> ŷ
///   Dữ liệu: x = 1, y = 0
///
///   Forward :  z₁ = 0, a₁ = 0.5, z₂ = 1, ŷ = 1
///   Loss    :  L = 1, ∂L/∂ŷ = 2
///   Backward:  δ₂ = 2·1 = 2      ∂L/∂w₂ = 2·0.5 = 1   ∂L/∂b₂ = 2
///              ∂L/∂a₁ = 2·2 = 4
///              δ₁ = 4·0.25 = 1   ∂L/∂w₁ = 1·1 = 1     ∂L/∂b₁ = 1
/// </summary>
public class HandCalculatedBackwardTests
{
    private static (NeuralNetwork Net, Neuron Hidden, Neuron Output) Build()
    {
        var net = HandCalculationDemo.BuildTinyNetwork();
        return (net, net.Layers[0].Neurons[0], net.Layers[1].Neurons[0]);
    }

    [Fact]
    public void Forward_KhopVoiTinhTay()
    {
        var (net, hidden, output) = Build();

        var predicted = net.Forward(Vec.Of(1.0));

        Assert.Equal(0.0, hidden.WeightedSum, precision: 12);
        Assert.Equal(0.5, hidden.Output, precision: 12);
        Assert.Equal(1.0, output.WeightedSum, precision: 12);
        Assert.Equal(1.0, predicted[0], precision: 12);
    }

    [Fact]
    public void Loss_VaDaoHamCuaLoss_KhopVoiTinhTay()
    {
        var (net, _, _) = Build();
        var mse = new MeanSquaredError();
        var predicted = net.Forward(Vec.Of(1.0));
        var target = Vec.Of(0.0);

        Assert.Equal(1.0, mse.Compute(predicted, target), precision: 12);
        Assert.Equal(2.0, mse.Gradient(predicted, target)[0], precision: 12);
    }

    [Fact]
    public void Backward_DeltaKhopVoiTinhTay()
    {
        var (net, hidden, output) = Build();
        var mse = new MeanSquaredError();
        var predicted = net.Forward(Vec.Of(1.0));

        net.ZeroGradients();
        net.Backward(mse.Gradient(predicted, Vec.Of(0.0)));

        Assert.Equal(2.0, output.Delta, precision: 12);   // 2 · f'(z₂) = 2 · 1
        Assert.Equal(1.0, hidden.Delta, precision: 12);   // 4 · f'(z₁) = 4 · 0.25
    }

    [Fact]
    public void Backward_GradientKhopVoiTinhTay()
    {
        var (net, hidden, output) = Build();
        var mse = new MeanSquaredError();
        var predicted = net.Forward(Vec.Of(1.0));

        net.ZeroGradients();
        net.Backward(mse.Gradient(predicted, Vec.Of(0.0)));

        Assert.Equal(1.0, hidden.WeightGradients[0], precision: 12);   // ∂L/∂w₁
        Assert.Equal(1.0, hidden.BiasGradient, precision: 12);         // ∂L/∂b₁
        Assert.Equal(1.0, output.WeightGradients[0], precision: 12);   // ∂L/∂w₂
        Assert.Equal(2.0, output.BiasGradient, precision: 12);         // ∂L/∂b₂

        // Thứ tự mảng phẳng: [w₁, b₁, w₂, b₂]
        Assert.Equal([1.0, 1.0, 1.0, 2.0], net.GetGradients());
    }

    [Fact]
    public void LayerBackward_TraVeDaoHamTheoDauVao()
    {
        var (net, _, _) = Build();
        var mse = new MeanSquaredError();
        var predicted = net.Forward(Vec.Of(1.0));

        net.ZeroGradients();

        // ∂L/∂a₁ = w₂ · δ₂ = 2 · 2 = 4
        var gradientWrtHiddenOutput = net.Layers[1].Backward(mse.Gradient(predicted, Vec.Of(0.0)));

        Assert.Equal(4.0, gradientWrtHiddenOutput[0], precision: 12);
    }

    /// <summary>Chu trình đầy đủ: Forward -> Loss -> Backward -> Update -> Forward again.</summary>
    [Fact]
    public void MotBuocHoc_LamLossGiamDungNhuTinhTay()
    {
        var (net, _, _) = Build();
        var mse = new MeanSquaredError();
        var target = Vec.Of(0.0);

        double lossBefore = mse.Compute(net.Forward(Vec.Of(1.0)), target);

        net.ZeroGradients();
        net.Backward(mse.Gradient(net.Forward(Vec.Of(1.0)), target));

        var parameters = net.GetParameters();
        new StochasticGradientDescent(0.1).Step(parameters, net.GetGradients());
        net.SetParameters(parameters);

        // w₁ = 0−0.1·1 = −0.1 ; b₁ = −0.1 ; w₂ = 2−0.1·1 = 1.9 ; b₂ = 0−0.1·2 = −0.2
        Assert.Equal([-0.1, -0.1, 1.9, -0.2], parameters.Select(p => Math.Round(p, 10)).ToArray());

        double lossAfter = mse.Compute(net.Forward(Vec.Of(1.0)), target);

        Assert.Equal(1.0, lossBefore, precision: 12);
        Assert.True(Math.Abs(lossAfter - 0.4294383) < 1e-6, $"Loss sau bước học = {lossAfter}");
        Assert.True(lossAfter < lossBefore);
    }
}

public class BackwardMechanicsTests
{
    [Fact]
    public void Backward_CongDonGradientQuaNhieuMau()
    {
        var net = HandCalculationDemo.BuildTinyNetwork();
        var mse = new MeanSquaredError();

        net.ZeroGradients();

        // Chạy đúng một mẫu hai lần -> gradient phải gấp đôi.
        for (int i = 0; i < 2; i++)
        {
            var predicted = net.Forward(Vec.Of(1.0));
            net.Backward(mse.Gradient(predicted, Vec.Of(0.0)));
        }

        Assert.Equal([2.0, 2.0, 2.0, 4.0], net.GetGradients());
    }

    [Fact]
    public void ZeroGradients_XoaSachGradientVaDelta()
    {
        var net = HandCalculationDemo.BuildTinyNetwork();
        var mse = new MeanSquaredError();

        net.Backward(mse.Gradient(net.Forward(Vec.Of(1.0)), Vec.Of(0.0)));
        Assert.NotEqual(new double[4], net.GetGradients());

        net.ZeroGradients();

        Assert.Equal(new double[4], net.GetGradients());
        Assert.All(net.Layers.SelectMany(l => l.Neurons), n => Assert.Equal(0.0, n.Delta));
    }

    /// <summary>
    /// Quên ZeroGradients là bug kinh điển: gradient của epoch trước cộng dồn
    /// vào epoch sau. Test này khoá lại hành vi cộng dồn để nó luôn tường minh.
    /// </summary>
    [Fact]
    public void QuenZeroGradients_ThiGradientBiCongDon()
    {
        var net = HandCalculationDemo.BuildTinyNetwork();
        var mse = new MeanSquaredError();

        net.ZeroGradients();
        net.Backward(mse.Gradient(net.Forward(Vec.Of(1.0)), Vec.Of(0.0)));
        var first = net.GetGradients();

        net.Backward(mse.Gradient(net.Forward(Vec.Of(1.0)), Vec.Of(0.0)));
        var second = net.GetGradients();

        Assert.Equal(first.Select(g => g * 2).ToArray(), second);
    }

    [Fact]
    public void Backward_TruocKhiForward_ThiNemLoi()
    {
        var net = new NeuralNetwork(new Layer(1, 2, new Tanh(), new Random(1)));

        Assert.Throws<InvalidOperationException>(() => net.Backward(Vec.Of(1.0, 1.0)));
    }

    [Fact]
    public void Backward_SaiKichThuocGradient_ThiNemLoi()
    {
        var net = new NeuralNetwork(new Layer(1, 2, new Tanh(), new Random(1)));
        net.Forward(Vec.Of(0.5));

        Assert.Throws<ArgumentException>(() => net.Backward(Vec.Of(1.0)));
    }

    [Fact]
    public void Backward_KhongLamThayDoiThamSo()
    {
        var net = HandCalculationDemo.BuildTinyNetwork();
        var mse = new MeanSquaredError();
        var before = net.GetParameters();

        net.ZeroGradients();
        net.Backward(mse.Gradient(net.Forward(Vec.Of(1.0)), Vec.Of(0.0)));

        Assert.Equal(before, net.GetParameters());
    }

    /// <summary>
    /// Với mạng Identity thuần, δ ở mọi layer bằng nhau nhân theo trọng số —
    /// kiểm tra chuỗi lan ngược nối đúng qua nhiều layer.
    /// </summary>
    [Fact]
    public void Backward_LanNguocQuaNhieuLayer()
    {
        var identity = new Identity();
        var net = new NeuralNetwork(
            new Layer(identity, new Neuron(Vec.Of(3.0), bias: 0, identity)),
            new Layer(identity, new Neuron(Vec.Of(5.0), bias: 0, identity)));

        net.Forward(Vec.Of(2.0));           // a₁ = 6, ŷ = 30
        net.ZeroGradients();
        net.Backward(Vec.Of(1.0));          // đặt ∂L/∂ŷ = 1 cho dễ theo dõi

        var layer2 = net.Layers[1].Neurons[0];
        var layer1 = net.Layers[0].Neurons[0];

        Assert.Equal(1.0, layer2.Delta, precision: 12);
        Assert.Equal(6.0, layer2.WeightGradients[0], precision: 12);   // δ · a₁ = 1 · 6
        Assert.Equal(5.0, layer1.Delta, precision: 12);                // w₂ · δ₂ = 5 · 1
        Assert.Equal(10.0, layer1.WeightGradients[0], precision: 12);  // δ₁ · x = 5 · 2
    }
}
