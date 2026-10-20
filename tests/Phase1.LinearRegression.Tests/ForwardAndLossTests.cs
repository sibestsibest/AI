using Ai.Phase1;

namespace Ai.Phase1.Tests;

/// <summary>
/// Kiểm tra hai bước đầu của vòng học: Forward và Loss.
/// Mọi con số kỳ vọng đều được tính bằng tay, không lấy từ output của code.
/// </summary>
public class ForwardAndLossTests
{
    [Theory]
    [InlineData(3, 2, 4, 14)]   // 3*4 + 2  = 14
    [InlineData(2, 1, 5, 11)]   // 2*5 + 1  = 11
    [InlineData(0, 7, 99, 7)]   // w = 0 -> output luôn bằng bias
    [InlineData(-1.5, 0.5, 2, -2.5)]
    public void Forward_TinhDungCongThuc_WxCongB(double w, double b, double x, double expected)
    {
        var model = new LinearRegressionModel(w, b);

        Assert.Equal(expected, model.Forward(x), precision: 10);
    }

    [Fact]
    public void Predict_TraVeGiongHetForward()
    {
        var model = new LinearRegressionModel(1.234, -5.678);

        foreach (double x in new double[] { -10, 0, 0.5, 42 })
        {
            Assert.Equal(model.Forward(x), model.Predict(x));
        }
    }

    /// <summary>
    /// w = 1, b = 0 -> ŷ = 1, 2, 3, 4, 5
    /// lỗi           = -2, -3, -4, -5, -6
    /// bình phương   = 4 + 9 + 16 + 25 + 36 = 90
    /// MSE           = 90 / 5 = 18
    /// </summary>
    [Fact]
    public void CalculateLoss_TrungVoiKetQuaTinhTay()
    {
        var model = new LinearRegressionModel(w: 1, b: 0);

        Assert.Equal(18.0, model.CalculateLoss(Dataset.Training), precision: 10);
    }

    /// <summary>
    /// w = 0, b = 0 -> lỗi = -3, -5, -7, -9, -11
    /// bình phương = 9 + 25 + 49 + 81 + 121 = 285 -> MSE = 57
    /// </summary>
    [Fact]
    public void CalculateLoss_ModelRong_TrungVoiKetQuaTinhTay()
    {
        var model = new LinearRegressionModel(w: 0, b: 0);

        Assert.Equal(57.0, model.CalculateLoss(Dataset.Training), precision: 10);
    }

    [Fact]
    public void CalculateLoss_BangKhongKhiModelDuDoanHoanHao()
    {
        var model = new LinearRegressionModel(w: 2, b: 1);

        Assert.Equal(0.0, model.CalculateLoss(Dataset.Training), precision: 12);
    }

    [Fact]
    public void CalculateLoss_LuonKhongAm_VoiSaiSoAmHayDuong()
    {
        // Lệch lên và lệch xuống cùng một lượng phải cho cùng một Loss.
        var tooHigh = new LinearRegressionModel(w: 2, b: 1.5);
        var tooLow = new LinearRegressionModel(w: 2, b: 0.5);

        double lossHigh = tooHigh.CalculateLoss(Dataset.Training);
        double lossLow = tooLow.CalculateLoss(Dataset.Training);

        Assert.True(lossHigh > 0);
        Assert.Equal(lossHigh, lossLow, precision: 10);
    }

    [Fact]
    public void Dataset_DungNhuDeBai()
    {
        Assert.Equal(5, Dataset.Training.Length);
        Assert.Equal(new Sample(1, 3), Dataset.Training[0]);
        Assert.Equal(new Sample(5, 11), Dataset.Training[4]);
    }
}
