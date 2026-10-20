using Ai.Phase1;

namespace Ai.Phase1.Tests;

/// <summary>
/// Kiểm tra bước quan trọng nhất: đạo hàm. Sai ở đây thì model vẫn chạy,
/// vẫn in ra số, chỉ là không bao giờ học được — nên phải test kỹ nhất.
/// </summary>
public class GradientTests
{
    /// <summary>
    /// w = 1, b = 0 -> e = -2, -3, -4, -5, -6
    /// Σ e·x = (-2·1) + (-3·2) + (-4·3) + (-5·4) + (-6·5) = -70
    /// Σ e   = -20
    /// dL/dw = (2/5)·(-70) = -28
    /// dL/db = (2/5)·(-20) = -8
    /// </summary>
    [Fact]
    public void CalculateGradient_TrungVoiKetQuaTinhTay()
    {
        var model = new LinearRegressionModel(w: 1, b: 0);

        var g = model.CalculateGradient(Dataset.Training);

        Assert.Equal(-28.0, g.DwLoss, precision: 10);
        Assert.Equal(-8.0, g.DbLoss, precision: 10);
    }

    /// <summary>
    /// w = 0, b = 0 -> Σ e·x = -125, Σ e = -35
    /// dL/dw = (2/5)·(-125) = -50 ; dL/db = (2/5)·(-35) = -14
    /// </summary>
    [Fact]
    public void CalculateGradient_ModelRong_TrungVoiKetQuaTinhTay()
    {
        var model = new LinearRegressionModel(w: 0, b: 0);

        var g = model.CalculateGradient(Dataset.Training);

        Assert.Equal(-50.0, g.DwLoss, precision: 10);
        Assert.Equal(-14.0, g.DbLoss, precision: 10);
    }

    [Fact]
    public void CalculateGradient_BangKhongTaiDiemToiUu()
    {
        // Tại đáy của hàm Loss, mặt phẳng nằm ngang -> gradient = 0 -> ngừng học.
        var model = new LinearRegressionModel(w: 2, b: 1);

        var g = model.CalculateGradient(Dataset.Training);

        Assert.Equal(0.0, g.DwLoss, precision: 10);
        Assert.Equal(0.0, g.DbLoss, precision: 10);
    }

    [Fact]
    public void CalculateGradient_DuongKhiModelDuDoanQuaCao()
    {
        // Dự đoán quá cao -> muốn giảm Loss thì phải GIẢM w và b
        // -> gradient dương, và phép trừ trong update sẽ kéo tham số xuống.
        var model = new LinearRegressionModel(w: 5, b: 5);

        var g = model.CalculateGradient(Dataset.Training);

        Assert.True(g.DwLoss > 0);
        Assert.True(g.DbLoss > 0);
    }

    /// <summary>
    /// Gradient check: so công thức giải tích với đạo hàm số
    /// f'(x) ≈ ( f(x+h) − f(x−h) ) / 2h tại nhiều điểm khác nhau.
    /// Đây là bằng chứng công thức backpropagation viết tay là đúng.
    /// </summary>
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 0)]
    [InlineData(-3.7, 2.4)]
    [InlineData(2, 1)]
    [InlineData(10, -10)]
    public void CalculateGradient_TrungVoiDaoHamSo(double w, double b)
    {
        var model = new LinearRegressionModel(w, b);

        var analytic = model.CalculateGradient(Dataset.Training);
        var numeric = GradientCheck.Numerical(w, b, Dataset.Training);

        // Sai phân trung tâm có sai số bậc h², chấp nhận lệch tương đối nhỏ.
        AssertRelativelyClose(numeric.DwLoss, analytic.DwLoss);
        AssertRelativelyClose(numeric.DbLoss, analytic.DbLoss);
    }

    [Fact]
    public void GradientCheck_LossAt_KhopVoiCalculateLossCuaModel()
    {
        var model = new LinearRegressionModel(w: 1.7, b: -0.3);

        Assert.Equal(
            model.CalculateLoss(Dataset.Training),
            GradientCheck.LossAt(1.7, -0.3, Dataset.Training),
            precision: 10);
    }

    /// <summary>
    /// w := w − lr · dL/dw   với w = 1, lr = 0.1, dL/dw = −28  ->  1 + 2.8 = 3.8
    /// b := b − lr · dL/db   với b = 0, lr = 0.1, dL/db = −8   ->  0 + 0.8 = 0.8
    /// </summary>
    [Fact]
    public void UpdateWeights_DiNguocHuongGradient()
    {
        var model = new LinearRegressionModel(w: 1, b: 0);

        model.UpdateWeights(new Gradient(DwLoss: -28, DbLoss: -8), learningRate: 0.1);

        Assert.Equal(3.8, model.W, precision: 10);
        Assert.Equal(0.8, model.B, precision: 10);
    }

    [Fact]
    public void UpdateWeights_LearningRateBangKhong_ThiKhongHocGiCa()
    {
        var model = new LinearRegressionModel(w: 1, b: 0);

        model.UpdateWeights(model.CalculateGradient(Dataset.Training), learningRate: 0);

        Assert.Equal(1.0, model.W);
        Assert.Equal(0.0, model.B);
    }

    [Fact]
    public void MotBuocGradientDescent_LamLossGiam()
    {
        var model = new LinearRegressionModel(w: 1, b: 0);
        double lossBefore = model.CalculateLoss(Dataset.Training);

        model.UpdateWeights(model.CalculateGradient(Dataset.Training), learningRate: 0.01);

        Assert.True(model.CalculateLoss(Dataset.Training) < lossBefore);
    }

    private static void AssertRelativelyClose(double expected, double actual)
    {
        double tolerance = 1e-5 * Math.Max(1.0, Math.Abs(expected));
        Assert.True(
            Math.Abs(expected - actual) <= tolerance,
            $"Lệch quá lớn: kỳ vọng {expected}, nhận {actual}");
    }
}
