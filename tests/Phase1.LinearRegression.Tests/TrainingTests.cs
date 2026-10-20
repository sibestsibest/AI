using Ai.Phase1;

namespace Ai.Phase1.Tests;

/// <summary>
/// Kiểm tra toàn bộ vòng học: chạy đủ số epoch thì model có thật sự
/// tự tìm ra w ≈ 2, b ≈ 1 mà không ai nói cho nó biết hay không.
/// </summary>
public class TrainingTests
{
    private const double LearningRate = 0.05;
    private const int Epochs = 1000;

    [Theory]
    [InlineData(1)]
    [InlineData(42)]
    [InlineData(2024)]
    [InlineData(-7)]
    public void Train_TuHocDuocQuyLuat_DuKhoiTaoNgauNhienKhacNhau(int seed)
    {
        var model = new LinearRegressionModel(seed);

        model.Train(Dataset.Training, Epochs, LearningRate);

        Assert.Equal(2.0, model.W, precision: 4);
        Assert.Equal(1.0, model.B, precision: 4);
    }

    [Fact]
    public void Train_LamLossGiamVeGanKhong()
    {
        var model = new LinearRegressionModel(seed: 42);
        double lossBefore = model.CalculateLoss(Dataset.Training);

        model.Train(Dataset.Training, Epochs, LearningRate);

        double lossAfter = model.CalculateLoss(Dataset.Training);
        Assert.True(lossBefore > 1.0, $"Model chưa train phải sai nhiều, nhưng Loss = {lossBefore}");
        Assert.True(lossAfter < 1e-10, $"Loss sau train phải gần 0, nhưng = {lossAfter}");
    }

    [Fact]
    public void Train_LossGiamDeuQuaTungEpoch_KhongDaoDong()
    {
        var model = new LinearRegressionModel(seed: 42);
        var losses = new List<double>();

        model.Train(Dataset.Training, 300, LearningRate, s => losses.Add(s.Loss));

        // Với learning rate ổn định, mỗi epoch phải tốt hơn epoch trước.
        for (int i = 1; i < losses.Count; i++)
        {
            Assert.True(
                losses[i] <= losses[i - 1],
                $"Loss tăng tại epoch {i + 1}: {losses[i - 1]} -> {losses[i]}");
        }
    }

    [Fact]
    public void Train_GoiCallbackDungSoLan_VaDanhSoEpochTu1()
    {
        var model = new LinearRegressionModel(seed: 42);
        var epochs = new List<int>();

        model.Train(Dataset.Training, 10, LearningRate, s => epochs.Add(s.Epoch));

        Assert.Equal(10, epochs.Count);
        Assert.Equal(Enumerable.Range(1, 10), epochs);
    }

    [Fact]
    public void Train_KhongEpochNao_ThiModelGiuNguyen()
    {
        var model = new LinearRegressionModel(w: 0.5, b: -0.5);

        model.Train(Dataset.Training, epochs: 0, learningRate: LearningRate);

        Assert.Equal(0.5, model.W);
        Assert.Equal(-0.5, model.B);
    }

    [Fact]
    public void Train_LearningRateQuaLon_ThiPhanKy()
    {
        // Điều kiện hội tụ: lr < 2 / λ_max ≈ 0.0845.
        // Vượt ngưỡng thì mỗi bước vọt qua đáy và ngày càng xa hơn.
        var model = new LinearRegressionModel(seed: 42);
        double lossBefore = model.CalculateLoss(Dataset.Training);

        model.Train(Dataset.Training, epochs: 100, learningRate: 0.5);

        Assert.True(
            model.CalculateLoss(Dataset.Training) > lossBefore,
            "Learning rate vượt ngưỡng ổn định phải làm Loss tăng, không giảm.");
    }

    [Fact]
    public void ModelDaTrain_TongQuatHoaSangDuLieuChuaTungThay()
    {
        var model = new LinearRegressionModel(seed: 42);
        model.Train(Dataset.Training, Epochs, LearningRate);

        // x = 6, 10, 100, -3 đều không có trong tập train.
        foreach (double x in new double[] { 6, 10, 100, -3, 0.5 })
        {
            Assert.Equal(2 * x + 1, model.Predict(x), precision: 3);
        }
    }

    [Fact]
    public void KhoiTaoNgauNhien_CungSeed_ChoCungKetQua()
    {
        var a = new LinearRegressionModel(seed: 123);
        var b = new LinearRegressionModel(seed: 123);

        Assert.Equal(a.W, b.W);
        Assert.Equal(a.B, b.B);
    }

    [Fact]
    public void KhoiTaoNgauNhien_NamTrongKhoang_TruMotDenMot()
    {
        for (int seed = 0; seed < 50; seed++)
        {
            var model = new LinearRegressionModel(seed);

            Assert.InRange(model.W, -1.0, 1.0);
            Assert.InRange(model.B, -1.0, 1.0);
        }
    }

    [Fact]
    public void KhoiTaoNgauNhien_KhongPhaiLucNaoCungRaSoGiongNhau()
    {
        var weights = Enumerable.Range(0, 20)
            .Select(seed => new LinearRegressionModel(seed).W)
            .Distinct()
            .Count();

        Assert.True(weights > 1, "Khởi tạo phải thực sự ngẫu nhiên theo seed.");
    }
}
