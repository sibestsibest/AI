using Ai.Phase2.Activations;

namespace Ai.Phase2.Tests;

public class ActivationTests
{
    public static TheoryData<IActivation> AllActivations =>
    [
        new Identity(),
        new Sigmoid(),
        new Tanh(),
        new ReLU(),
    ];

    [Fact]
    public void Identity_TraVeChinhDauVao()
    {
        var f = new Identity();

        Assert.Equal(-3.7, f.Forward(-3.7), precision: 10);
        Assert.Equal(1.0, f.Derivative(-3.7), precision: 10);
    }

    [Theory]
    [InlineData(0, 0.5)]                    // 1/(1+e⁰) = 1/2
    [InlineData(2, 0.8807970779778823)]
    [InlineData(-2, 0.11920292202211755)]
    public void Sigmoid_TinhDungGiaTri(double z, double expected)
    {
        Assert.Equal(expected, new Sigmoid().Forward(z), precision: 10);
    }

    [Fact]
    public void Sigmoid_LuonNamTrongKhoangKhongMot()
    {
        var f = new Sigmoid();

        foreach (double z in new double[] { -1000, -5, 0, 5, 1000 })
        {
            Assert.InRange(f.Forward(z), 0.0, 1.0);
        }
    }

    [Fact]
    public void Sigmoid_DaoHamTaiKhong_Bang025()
    {
        // f'(0) = f(0)·(1 − f(0)) = 0.5 · 0.5 = 0.25 — đây cũng là giá trị lớn nhất.
        Assert.Equal(0.25, new Sigmoid().Derivative(0), precision: 10);
    }

    [Fact]
    public void Tanh_DoiXungQuanhGoc()
    {
        var f = new Tanh();

        Assert.Equal(0.0, f.Forward(0), precision: 10);
        Assert.Equal(-f.Forward(1.3), f.Forward(-1.3), precision: 10);
        Assert.Equal(1.0, f.Derivative(0), precision: 10);
    }

    [Theory]
    [InlineData(-3, 0)]
    [InlineData(0, 0)]
    [InlineData(2.5, 2.5)]
    public void ReLU_CatSoAmThanhKhong(double z, double expected)
    {
        Assert.Equal(expected, new ReLU().Forward(z), precision: 10);
    }

    [Fact]
    public void ReLU_DaoHamLaBacThang()
    {
        var f = new ReLU();

        Assert.Equal(0.0, f.Derivative(-1));
        Assert.Equal(0.0, f.Derivative(0));   // quy ước tại điểm gãy
        Assert.Equal(1.0, f.Derivative(1));
    }

    /// <summary>
    /// Đối chiếu mọi công thức đạo hàm với định nghĩa gốc:
    ///     f'(z) ≈ ( f(z+h) − f(z−h) ) / (2h)
    /// Bỏ qua z quanh 0 với ReLU vì tại đó hàm không khả vi.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllActivations))]
    public void DaoHam_TrungVoiDaoHamSo(IActivation activation)
    {
        const double h = 1e-6;

        foreach (double z in new double[] { -3.1, -1.4, -0.3, 0.7, 2.2, 4.5 })
        {
            double numeric = (activation.Forward(z + h) - activation.Forward(z - h)) / (2 * h);
            double analytic = activation.Derivative(z);

            Assert.True(
                Math.Abs(numeric - analytic) < 1e-5,
                $"{activation.Name} tại z={z}: giải tích {analytic}, số {numeric}");
        }
    }

    [Theory]
    [MemberData(nameof(AllActivations))]
    public void MoiActivation_CoTen(IActivation activation)
    {
        Assert.False(string.IsNullOrWhiteSpace(activation.Name));
    }
}
