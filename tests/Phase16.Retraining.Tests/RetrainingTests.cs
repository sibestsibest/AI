using Ai.Phase16.Agent;
using Ai.Phase16.Feedback;
using Ai.Phase16.Learning;
using Ai.Phase16.Memory;
using Ai.Phase16.Retraining;
using Ai.Phase16.Verification;

namespace Ai.Phase16.Tests;

/// <summary>Dựng sẵn số liệu đánh giá để test khâu so sánh và cửa triển khai tách khỏi khâu train.</summary>
internal static class Met
{
    public static EvaluationMetrics Make(int total, int correct, double macroF1Target = 1.0,
        IEnumerable<string>? wrong = null, IEnumerable<string>? predictionKeys = null,
        Intent predicted = Intent.Calculate, double meanLoss = 0.1,
        IEnumerable<string>? withoutTrainedWords = null)
    {
        var wrongList = (wrong ?? []).ToList();
        var keys = (predictionKeys ?? Enumerable.Range(1, total).Select(i => $"k{i}")).ToList();

        // Dựng PerLabel sao cho MacroF1 đúng bằng mức muốn test: một nhãn duy
        // nhất, với số liệu cho ra F1 = macroF1Target khi precision = recall.
        int support = Math.Max(1, total);
        int truePositive = (int)Math.Round(macroF1Target * support);

        var perLabel = new List<LabelMetrics>
        {
            new(predicted, support, truePositive, support - truePositive, support - truePositive),
        };

        return new EvaluationMetrics
        {
            Total = total,
            Correct = correct,
            PerLabel = perLabel,
            MeanLoss = meanLoss,
            Predictions = keys.ToDictionary(k => k, _ => predicted, StringComparer.Ordinal),
            Wrong = wrongList,
            WithoutTrainedWords = [.. withoutTrainedWords ?? []],
        };
    }

    public static DataSplit Split(params Intent[] missingLabels) => new()
    {
        Training = [],
        Validation = [],
        CorrectionsRoutedToTraining = [],
        LabelsMissingFromTraining = missingLabels,
    };
}

public class TrainingConfigTests
{
    [Fact]
    public void CauHinhHopLeThiKhongNem()
    {
        new TrainingConfig().Validate();
    }

    [Theory]
    [InlineData(0, 0.3, 16, 0.2)]
    [InlineData(100, 0, 16, 0.2)]
    [InlineData(100, 0.3, 0, 0.2)]
    [InlineData(100, 0.3, 16, 0.6)]
    [InlineData(100, 0.3, 16, -0.1)]
    public void CauHinhSaiThiNemNgayChuKhongDeTrainXongMoiHong(
        int epochs, double learningRate, int hiddenSize, double validationShare)
    {
        var config = new TrainingConfig
        {
            Epochs = epochs,
            LearningRate = learningRate,
            HiddenSize = hiddenSize,
            ValidationShare = validationShare,
        };

        Assert.Throws<ArgumentOutOfRangeException>(config.Validate);
    }

    [Fact]
    public void CungCauHinhThiCungVanTay()
    {
        Assert.Equal(new TrainingConfig().Fingerprint, new TrainingConfig().Fingerprint);
    }

    /// <summary>Seed là một phần của danh tính model — đổi seed là một model khác.</summary>
    [Fact]
    public void DoiSeedLaDoiVanTay()
    {
        Assert.NotEqual(
            new TrainingConfig { Seed = 1 }.Fingerprint,
            new TrainingConfig { Seed = 2 }.Fingerprint);
    }

    [Fact]
    public void DoiSieuThamSoNaoCungDoiVanTay()
    {
        var baseline = new TrainingConfig().Fingerprint;

        Assert.NotEqual(baseline, new TrainingConfig { Epochs = 2999 }.Fingerprint);
        Assert.NotEqual(baseline, new TrainingConfig { LearningRate = 0.31 }.Fingerprint);
        Assert.NotEqual(baseline, new TrainingConfig { HiddenSize = 17 }.Fingerprint);
        Assert.NotEqual(baseline, new TrainingConfig { ValidationShare = 0.25 }.Fingerprint);
    }

    /// <summary>
    /// Vân tay chốt cứng: nó phải giống nhau qua mọi lần chạy và mọi máy, nếu
    /// không thì hai lần train không so được với nhau.
    /// </summary>
    [Fact]
    public void VanTayTatDinhQuaCacLanChay()
    {
        Assert.Equal("e29c6055", new TrainingConfig().Fingerprint);
    }
}

public class DataSplitterTests
{
    private static DatasetVersion Dataset(int count)
    {
        var repository = new DatasetRepository(new ManualClock(Ans.Now));
        repository.Seed(Enumerable.Range(1, count)
            .Select(i => ($"câu hỏi thử số {NumberWord(i)}", (Intent)(i % 7))));

        return repository.Current;
    }

    /// <summary>
    /// Danh tính chuẩn hoá bỏ token một ký tự, nên phải dùng CHỮ để các ví dụ
    /// thử thật sự khác nhau — dùng "1", "2" thì chúng thành cùng một ví dụ.
    /// </summary>
    private static string NumberWord(int i) => i switch
    {
        _ when i % 7 == 0 => $"alpha{i}",
        _ when i % 7 == 1 => $"beta{i}",
        _ when i % 7 == 2 => $"gamma{i}",
        _ when i % 7 == 3 => $"delta{i}",
        _ when i % 7 == 4 => $"epsilon{i}",
        _ when i % 7 == 5 => $"zeta{i}",
        _ => $"eta{i}",
    };

    [Fact]
    public void ChiaHaiLanChoKetQuaGiongNhau()
    {
        var dataset = Dataset(40);

        var first = DataSplitter.Split(dataset, 0.2);
        var second = DataSplitter.Split(dataset, 0.2);

        Assert.Equal(first.Validation.Select(e => e.Key), second.Validation.Select(e => e.Key));
    }

    /// <summary>
    /// TÍNH CHẤT QUAN TRỌNG NHẤT: thêm ví dụ mới KHÔNG làm ví dụ cũ xê dịch
    /// giữa train và đánh giá.
    ///
    /// Không có nó thì một ví dụ hôm qua ở tập đánh giá hôm nay rơi vào tập
    /// train, model mới đã HỌC chính câu nó sắp bị kiểm, và điểm cao lên mà
    /// chẳng giỏi thêm.
    /// </summary>
    [Fact]
    public void ThemViDuKhongLamXeDichViDuCu()
    {
        var small = Dataset(20);
        var large = Dataset(40);

        var before = DataSplitter.Split(small, 0.2);
        var after = DataSplitter.Split(large, 0.2);

        var validationBefore = before.Validation.Select(e => e.Key).ToHashSet(StringComparer.Ordinal);
        var validationAfter = after.Validation.Select(e => e.Key).ToHashSet(StringComparer.Ordinal);

        foreach (var example in small.Examples)
        {
            Assert.Equal(validationBefore.Contains(example.Key), validationAfter.Contains(example.Key));
        }
    }

    [Fact]
    public void PhanDanhGiaBangKhongThiMoiViDuDeuDeTrain()
    {
        var split = DataSplitter.Split(Dataset(20), 0);

        Assert.Empty(split.Validation);
        Assert.Equal(20, split.Training.Count);
    }

    [Fact]
    public void PhanDanhGiaNgoaiKhoangThiNem()
    {
        var dataset = Dataset(10);

        Assert.Throws<ArgumentOutOfRangeException>(() => DataSplitter.Split(dataset, 0.7));
        Assert.Throws<ArgumentOutOfRangeException>(() => DataSplitter.Split(dataset, -0.1));
    }

    /// <summary>
    /// Nhãn chỉ có một ví dụ mà ví dụ đó rơi vào tập đánh giá thì model không
    /// học được nhãn đó — và accuracy tổng vẫn có thể đẹp nếu nhãn đó hiếm.
    /// </summary>
    [Fact]
    public void PhatHienNhanKhongCoViDuNaoDeTrain()
    {
        var repository = new DatasetRepository(new ManualClock(Ans.Now));

        // Tìm một câu rơi vào tập đánh giá (bucket < 200 với share 0.2).
        string onlyExample = Enumerable.Range(1, 500)
            .Select(i => $"câu riêng biệt số alpha{i}")
            .First(text => DataSplitter.Bucket($"{Intent.SearchInternet}|{Keywords.Canonical(text)}") < 200);

        repository.Seed(
            [
                ("tính tổng của A và B", Intent.Calculate),
                ("chào bạn nhé", Intent.SmallTalk),
                (onlyExample, Intent.SearchInternet),
            ]);

        var split = DataSplitter.Split(repository.Current, 0.2);

        Assert.Contains(Intent.SearchInternet, split.LabelsMissingFromTraining);
    }

    /// <summary>
    /// Ví dụ đến từ lời sửa của người dùng KHÔNG bốc thăm — nó luôn vào tập
    /// train, kể cả khi băm xếp nó vào tập đánh giá.
    ///
    /// Đây là bản sửa cho một lỗi thật: câu "overfitting là gì" có bucket 70,
    /// tức là dưới ngưỡng 200 nên rơi vào tập đánh giá. Model mới không học nó,
    /// nên vẫn đoán sai y như cũ — và vì cách chia là tất định, nó sẽ ở đó mãi,
    /// người dùng sửa bao nhiêu lần cũng vô ích.
    /// </summary>
    [Fact]
    public void ViDuSuaLoiLuonVaoTapTrainDuBamXepNoVaoTapDanhGia()
    {
        const string text = "overfitting là gì";
        string key = $"{Intent.SearchKnowledge}|{Keywords.Canonical(text)}";

        Assert.True(DataSplitter.Bucket(key) < 200, "câu thử phải là câu vốn bị xếp vào tập đánh giá");

        var repository = new DatasetRepository(new ManualClock(Ans.Now));
        repository.Seed([("tính tổng của A và B", Intent.Calculate), ("chào bạn nhé", Intent.SmallTalk)]);

        var asCurated = DataSplitter.Split(repository.Current, 0.2);
        Assert.Empty(asCurated.CorrectionsRoutedToTraining);

        repository.Commit(
            [
                new TrainingCandidate
                {
                    Id = "c1",
                    Input = text,
                    Label = Intent.SearchKnowledge,
                    Provenance = CandidateProvenance.UserCorrection,
                    FeedbackId = "f1",
                    CreatedAt = Ans.Now,
                },
            ]);

        var split = DataSplitter.Split(repository.Current, 0.2);

        Assert.Contains(split.Training, e => e.Key == key);
        Assert.DoesNotContain(split.Validation, e => e.Key == key);
        Assert.Single(split.CorrectionsRoutedToTraining);
    }

    /// <summary>Ví dụ do người viết hệ thống gán nhãn thì vẫn bốc thăm như thường.</summary>
    [Fact]
    public void ViDuNguoiTuGanNhanThiVanBocThamNhuThuong()
    {
        var repository = new DatasetRepository(new ManualClock(Ans.Now));
        repository.Seed(
            [
                ("tính tổng của A và B", Intent.Calculate),
                ("chào bạn nhé", Intent.SmallTalk),
                ("overfitting là gì", Intent.SearchKnowledge),
            ]);

        var split = DataSplitter.Split(repository.Current, 0.2);

        Assert.Contains(split.Validation, e => e.Input == "overfitting là gì");
        Assert.Empty(split.CorrectionsRoutedToTraining);
    }

    [Fact]
    public void NganLuonNamTrongKhoangHopLe()
    {
        for (int i = 0; i < 200; i++)
        {
            int bucket = DataSplitter.Bucket($"khoá thử {i}");

            Assert.InRange(bucket, 0, 999);
        }
    }

    /// <summary>
    /// Băm chốt cứng. string.GetHashCode của .NET được ngẫu nhiên hoá theo tiến
    /// trình, nên nếu ai đổi sang dùng nó thì test này đỏ — đúng lúc cần đỏ.
    /// </summary>
    [Fact]
    public void BamTatDinhQuaCacLanChay()
    {
        Assert.Equal(5626779061062317208UL, DataSplitter.Fnv1a("miniai"));
        Assert.Equal(DataSplitter.Fnv1a("miniai"), DataSplitter.Fnv1a("miniai"));
        Assert.NotEqual(DataSplitter.Fnv1a("miniai"), DataSplitter.Fnv1a("miniia"));
    }
}

public class EvaluationMetricsTests
{
    [Fact]
    public void AccuracyVaMacroF1TinhTuSoLieuTho()
    {
        var metrics = new EvaluationMetrics
        {
            Total = 10,
            Correct = 7,
            MeanLoss = 0.5,
            PerLabel =
            [
                new(Intent.Calculate, 5, 5, 0, 0),      // F1 = 1.0
                new(Intent.SmallTalk, 5, 2, 3, 3),      // P = R = 0.4 -> F1 = 0.4
            ],
            Predictions = new Dictionary<string, Intent> { ["k1"] = Intent.Calculate },
            Wrong = ["k2"],
            WithoutTrainedWords = [],
        };

        Assert.Equal(0.7, metrics.Accuracy, 6);
        Assert.Equal(0.7, metrics.MacroF1, 6);
    }

    [Fact]
    public void KhongCoViDuNaoThiKhongChiaChoKhong()
    {
        var metrics = Met.Make(0, 0);

        Assert.Equal(0, metrics.Accuracy);
        Assert.Equal(0, metrics.AccuracyOnTrainedVocabulary);
    }

    /// <summary>
    /// Nhãn model chưa bao giờ đoán tới là dấu hiệu nó đã bỏ hẳn một ý định —
    /// accuracy trung bình không nói ra điều đó.
    /// </summary>
    [Fact]
    public void PhatHienNhanDaChet()
    {
        var metrics = new EvaluationMetrics
        {
            Total = 6,
            Correct = 5,
            MeanLoss = 0.2,
            PerLabel =
            [
                new(Intent.Calculate, 5, 5, 0, 0),
                new(Intent.SearchInternet, 1, 0, 0, 1),   // chưa bao giờ được đoán
            ],
            Predictions = new Dictionary<string, Intent> { ["k1"] = Intent.Calculate },
            Wrong = ["k2"],
            WithoutTrainedWords = [],
        };

        Assert.Equal([Intent.SearchInternet], metrics.DeadLabels);
        Assert.True(metrics.Accuracy > 0.8);
        Assert.True(metrics.MacroF1 < 0.6, "macro F1 phải bị kéo xuống vì một ý định đã chết");
    }

    /// <summary>
    /// Tách phần sai do THIẾU DỮ LIỆU khỏi phần sai do model: 10 câu, sai 4,
    /// trong đó 3 câu không có từ nào từng được train.
    /// </summary>
    [Fact]
    public void TachDuocPhanSaiDoThieuDuLieu()
    {
        var metrics = Met.Make(10, 6,
            wrong: ["k1", "k2", "k3", "k4"],
            withoutTrainedWords: ["k1", "k2", "k3", "k10"]);

        Assert.Equal(0.6, metrics.Accuracy, 6);
        Assert.Equal(3, metrics.WithoutTrainedWordsWrong);

        // Còn 6 câu đo được, trong đó đúng 6 - 3 = ... 5 câu (k10 đúng nhưng bị loại khỏi mẫu số)
        Assert.Equal(5.0 / 6.0, metrics.AccuracyOnTrainedVocabulary, 6);
        Assert.True(metrics.AccuracyOnTrainedVocabulary > metrics.Accuracy);
    }
}

public class ModelComparisonTests
{
    private static EvaluationMetrics Metrics(string[] keys, string[] wrong, double macroF1 = 1.0,
        Intent predicted = Intent.Calculate) =>
        new()
        {
            Total = keys.Length,
            Correct = keys.Length - wrong.Length,
            MeanLoss = 0.1,
            PerLabel = [new(predicted, keys.Length, (int)Math.Round(macroF1 * keys.Length),
                keys.Length - (int)Math.Round(macroF1 * keys.Length),
                keys.Length - (int)Math.Round(macroF1 * keys.Length))],
            Predictions = keys.ToDictionary(k => k, _ => predicted, StringComparer.Ordinal),
            Wrong = wrong,
            WithoutTrainedWords = [],
        };

    [Fact]
    public void ChiRaViDuNaoTuDungThanhSaiVaNguocLai()
    {
        var baseline = Metrics(["a", "b", "c", "d"], ["c"]);
        var candidate = Metrics(["a", "b", "c", "d"], ["a"]);

        var comparison = ModelComparison.Compare(baseline, candidate);

        Assert.Equal(["a"], comparison.Regressions);
        Assert.Equal(["c"], comparison.Improvements);
    }

    /// <summary>
    /// TRƯỜNG HỢP QUAN TRỌNG NHẤT: chênh lệch accuracy bằng 0 mà vẫn có ba thứ
    /// đang chạy tốt vừa hỏng. Con số trung bình triệt tiêu mất điều đó.
    /// </summary>
    [Fact]
    public void ChenhLechAccuracyBangKhongVanCoTheLaThayDoiRatLon()
    {
        var baseline = Metrics(["a", "b", "c", "d", "e", "f"], ["a", "b", "c"]);
        var candidate = Metrics(["a", "b", "c", "d", "e", "f"], ["d", "e", "f"]);

        var comparison = ModelComparison.Compare(baseline, candidate);

        Assert.Equal(0, comparison.AccuracyDelta, 6);
        Assert.Equal(3, comparison.Regressions.Count);
        Assert.Equal(3, comparison.Improvements.Count);
        Assert.True(comparison.HasRegressions);
    }

    [Fact]
    public void ChiSoSanhNhungViDuCaHaiModelDeuDuocCham()
    {
        var baseline = Metrics(["a", "b"], []);
        var candidate = Metrics(["a", "b", "c"], ["c"]);

        var comparison = ModelComparison.Compare(baseline, candidate);

        Assert.Empty(comparison.Regressions);
        Assert.Empty(comparison.Improvements);
    }

    [Fact]
    public void PhatHienNhanBiMatHan()
    {
        var baseline = new EvaluationMetrics
        {
            Total = 2,
            Correct = 2,
            MeanLoss = 0.1,
            PerLabel = [new(Intent.Calculate, 1, 1, 0, 0), new(Intent.SearchInternet, 1, 1, 0, 0)],
            Predictions = new Dictionary<string, Intent>(StringComparer.Ordinal)
            {
                ["a"] = Intent.Calculate,
                ["b"] = Intent.SearchInternet,
            },
            Wrong = [],
            WithoutTrainedWords = [],
        };

        var candidate = Metrics(["a", "b"], ["b"]);   // chỉ còn đoán Calculate

        var comparison = ModelComparison.Compare(baseline, candidate);

        Assert.Equal([Intent.SearchInternet], comparison.LabelsLost);
        Assert.True(comparison.LostALabel);
    }
}

public class DeploymentGateTests
{
    private readonly DeploymentGate _gate = new();

    [Fact]
    public void ModelDauTienDatThiDuocTrienKhai()
    {
        var decision = _gate.Evaluate(Met.Make(20, 15), Met.Make(7, 7), null, Met.Split());

        Assert.True(decision.Approved);
        Assert.Empty(decision.Blockers);
    }

    /// <summary>Không đo được thì không triển khai — đây là điều kiện đầu tiên, không phải chi tiết.</summary>
    [Fact]
    public void TapDanhGiaRongThiTuChoi()
    {
        var decision = _gate.Evaluate(Met.Make(0, 0), Met.Make(7, 7), null, Met.Split());

        Assert.False(decision.Approved);
        Assert.Contains(decision.Blockers, b => b.Contains("RỖNG", StringComparison.Ordinal));
    }

    [Fact]
    public void ThieuNhanKhiTrainThiTuChoi()
    {
        var decision = _gate.Evaluate(Met.Make(20, 20), Met.Make(7, 7), null,
            Met.Split(Intent.SearchInternet));

        Assert.False(decision.Approved);
        Assert.Contains(decision.Blockers, b => b.Contains("SearchInternet", StringComparison.Ordinal));
    }

    [Fact]
    public void DuoiSanAccuracyThiTuChoi()
    {
        var decision = _gate.Evaluate(Met.Make(20, 5), Met.Make(7, 7), null, Met.Split());

        Assert.False(decision.Approved);
        Assert.Contains(decision.Blockers, b => b.Contains("accuracy", StringComparison.Ordinal));
    }

    [Fact]
    public void ThutLuiMacroF1SoVoiModelDangChayThiTuChoi()
    {
        var baseline = Met.Make(20, 18, macroF1Target: 0.9);
        var candidate = Met.Make(20, 15, macroF1Target: 0.6);

        var decision = _gate.Evaluate(candidate, Met.Make(7, 7),
            ModelComparison.Compare(baseline, candidate), Met.Split());

        Assert.False(decision.Approved);
        Assert.Contains(decision.Blockers, b => b.Contains("macro F1", StringComparison.Ordinal));
    }

    /// <summary>
    /// Model tốt lên về tổng thể nhưng làm hỏng thứ đang chạy được thì vẫn bị
    /// từ chối. Muốn nhận đánh đổi đó phải có người nới hạn mức, tường minh.
    /// </summary>
    [Fact]
    public void LamHongThuDangChayDuocThiTuChoiDuAccuracyTangLen()
    {
        var baseline = Met.Make(6, 3, predictionKeys: ["a", "b", "c", "d", "e", "f"], wrong: ["a", "b", "c"]);
        var candidate = Met.Make(6, 4, predictionKeys: ["a", "b", "c", "d", "e", "f"], wrong: ["d", "e"]);

        var comparison = ModelComparison.Compare(baseline, candidate);
        var decision = _gate.Evaluate(candidate, Met.Make(7, 7), comparison, Met.Split());

        Assert.True(comparison.AccuracyDelta > 0);
        Assert.False(decision.Approved);
        Assert.Contains(decision.Blockers, b => b.Contains("từ đúng thành sai", StringComparison.Ordinal));
    }

    [Fact]
    public void NoiHanMucThiChapNhanDanhDoi()
    {
        var gate = new DeploymentGate { MaxRegressions = 2 };

        var baseline = Met.Make(6, 3, predictionKeys: ["a", "b", "c", "d", "e", "f"], wrong: ["a", "b", "c"]);
        var candidate = Met.Make(6, 4, predictionKeys: ["a", "b", "c", "d", "e", "f"], wrong: ["d", "e"]);

        var decision = gate.Evaluate(candidate, Met.Make(7, 7),
            ModelComparison.Compare(baseline, candidate), Met.Split());

        Assert.True(decision.Approved);
    }

    /// <summary>Bộ vàng không cho đánh đổi: sai một câu là không triển khai, bất kể điểm tổng.</summary>
    [Fact]
    public void BoVangSaiMotCauLaKhongTrienKhai()
    {
        var decision = _gate.Evaluate(Met.Make(20, 20), Met.Make(7, 6, wrong: ["g1"]), null, Met.Split());

        Assert.False(decision.Approved);
        Assert.Contains(decision.Blockers, b => b.Contains("bộ vàng", StringComparison.Ordinal));
    }

    [Fact]
    public void MoiDieuKienDeuKemConSoThat()
    {
        var decision = _gate.Evaluate(Met.Make(20, 15), Met.Make(7, 7), null, Met.Split());

        Assert.All(decision.Checks, check => Assert.False(string.IsNullOrWhiteSpace(check.Detail)));
    }
}

public class ModelRegistryTests
{
    private readonly ManualClock _clock = new(Ans.Now);
    private readonly ModelRegistry _registry;

    public ModelRegistryTests() => _registry = new ModelRegistry(_clock);

    private ModelVersion Model(int version, double accuracy = 0.8) => new()
    {
        Version = version,
        Weights = [0.1, 0.2, 0.3],
        VocabularyWords = ["tính", "cộng"],
        Config = new TrainingConfig(),
        DatasetVersion = 1,
        DatasetFingerprint = "abc123abc123",
        Metrics = Met.Make(10, (int)(accuracy * 10)),
        TrainedAt = _clock.Now,
    };

    private static DeploymentDecision Approved() => new(true, [new GateCheck("thử", true, "ok")]);

    private static DeploymentDecision Rejected() => new(false, [new GateCheck("thử", false, "không đạt")]);

    /// <summary>Đăng ký KHÁC triển khai — đây là cả điểm của sổ này.</summary>
    [Fact]
    public void DangKyKhongLamModelThanhModelDangChay()
    {
        _registry.Register(Model(1));

        Assert.Null(_registry.Active);
        Assert.Single(_registry.Versions);
        Assert.Equal(ModelEventKind.Registered, _registry.History[0].Kind);
    }

    [Fact]
    public void KhongDatThiKhongDuocThanhModelDangChay()
    {
        var model = _registry.Register(Model(1));

        _registry.Deploy(model, Rejected());

        Assert.Null(_registry.Active);
        Assert.Equal(ModelEventKind.Rejected, _registry.History[^1].Kind);
        Assert.NotEmpty(_registry.History[^1].Reasons);
    }

    [Fact]
    public void DatThiDuocTrienKhai()
    {
        var model = _registry.Register(Model(1));

        _registry.Deploy(model, Approved());

        Assert.Same(model, _registry.Active);
        Assert.Equal(1, _registry.DeployedCount);
    }

    [Fact]
    public void ModelBiTuChoiVanDuocGiuLaiTrongSo()
    {
        var rejected = _registry.Register(Model(1, accuracy: 0.2));
        _registry.Deploy(rejected, Rejected());

        Assert.Single(_registry.Versions);
        Assert.NotNull(_registry.Get(1));
        Assert.Equal(2, _registry.NextVersionNumber);
    }

    [Fact]
    public void TrungSoPhienBanThiNem()
    {
        _registry.Register(Model(1));

        Assert.Throws<InvalidOperationException>(() => _registry.Register(Model(1)));
    }

    [Fact]
    public void TrienKhaiModelChuaDangKyThiNem()
    {
        Assert.Throws<InvalidOperationException>(() => _registry.Deploy(Model(9), Approved()));
    }

    [Fact]
    public void QuayLuiVeModelDaTungChay()
    {
        var first = _registry.Register(Model(1));
        _registry.Deploy(first, Approved());

        var second = _registry.Register(Model(2));
        _registry.Deploy(second, Approved());

        var back = _registry.Rollback("m2 gây lỗi trên thực tế");

        Assert.Same(first, back);
        Assert.Same(first, _registry.Active);
        Assert.Equal(ModelEventKind.RolledBack, _registry.History[^1].Kind);
        Assert.Contains("thực tế", _registry.History[^1].Reasons[0], StringComparison.Ordinal);
    }

    /// <summary>Quay lui về hư vô không an toàn hơn giữ nguyên — nó làm hệ thống mất luôn bộ phân loại.</summary>
    [Fact]
    public void KhongCoGiDeQuayVeThiGiuNguyen()
    {
        var first = _registry.Register(Model(1));
        _registry.Deploy(first, Approved());

        Assert.Null(_registry.Rollback());
        Assert.Same(first, _registry.Active);
    }

    /// <summary>
    /// Quay lui KHÔNG được nhảy vào một model chưa bao giờ vượt cửa đánh giá —
    /// đó không phải quay lui, đó là triển khai một model chưa được duyệt.
    /// </summary>
    [Fact]
    public void QuayLuiBoQuaNhungModelChuaTungDuocDuyet()
    {
        var first = _registry.Register(Model(1));
        _registry.Deploy(first, Approved());

        var bad = _registry.Register(Model(2, accuracy: 0.1));
        _registry.Deploy(bad, Rejected());

        var third = _registry.Register(Model(3));
        _registry.Deploy(third, Approved());

        Assert.Same(first, _registry.Rollback());
    }

    [Fact]
    public void ChuaCoModelNaoThiKhongDungLaiDuoc()
    {
        Assert.Throws<InvalidOperationException>(_registry.RestoreActive);
    }
}

public class ModelVersionTests
{
    private static ModelVersion Model(double[] weights) => new()
    {
        Version = 1,
        Weights = weights,
        VocabularyWords = ["tính", "cộng"],
        Config = new TrainingConfig(),
        DatasetVersion = 1,
        DatasetFingerprint = "abc123abc123",
        Metrics = Met.Make(10, 8),
        TrainedAt = Ans.Now,
    };

    [Fact]
    public void CungTrongSoThiCungVanTay()
    {
        Assert.Equal(Model([0.1, 0.2]).Fingerprint, Model([0.1, 0.2]).Fingerprint);
    }

    /// <summary>Lệch một tham số ở chữ số thứ 16 vẫn là một model khác — và vân tay phải nói ra.</summary>
    [Fact]
    public void LechMotChutOChuSoCuoiCungLaModelKhac()
    {
        Assert.NotEqual(Model([0.1, 0.2]).Fingerprint, Model([0.1, 0.2000000000000001]).Fingerprint);
    }

    [Fact]
    public void NapLaiTrongSoSaiSoLuongThiNem()
    {
        var classifier = new IntentClassifier(["tính cộng"], new Random(1));

        Assert.Throws<ArgumentException>(() => classifier.ImportWeights([1.0, 2.0]));
    }

    [Fact]
    public void XuatRoiNapLaiChoRaDungModelCu()
    {
        var classifier = new IntentClassifier(["tính tổng của A và B", "chào bạn nhé"], new Random(3));
        classifier.Train([("tính tổng của A và B", Intent.Calculate), ("chào bạn nhé", Intent.SmallTalk)], 100);

        double[] weights = classifier.ExportWeights();
        var before = classifier.Classify("tính tổng");

        var restored = new IntentClassifier(classifier.Vocabulary, new Random(99));
        restored.ImportWeights(weights);

        var after = restored.Classify("tính tổng");

        Assert.Equal(before.Intent, after.Intent);
        Assert.Equal(before.Confidence, after.Confidence, 15);
    }
}

public class GoldenSetTests
{
    [Fact]
    public void MoiYDinhMotCau()
    {
        Assert.Equal(Enum.GetValues<Intent>().Length, GoldenSet.Examples.Length);
        Assert.Equal(Enum.GetValues<Intent>().Length, GoldenSet.Examples.Select(e => e.Label).Distinct().Count());
    }

    /// <summary>
    /// Bộ vàng phải nằm NGOÀI tập huấn luyện. Trùng thì model đã học thuộc chính
    /// câu nó sắp bị kiểm, và bộ vàng không còn đo khả năng khái quát hoá nữa.
    /// </summary>
    [Fact]
    public void BoVangKhongTrungVoiTapHuanLuyen()
    {
        var stack = LearningStackFactory.Build(new ManualClock(Ans.Now));

        Assert.Empty(GoldenSet.OverlapWith(stack.Datasets.Current));
    }
}

/// <summary>
/// Chạy đường ống huấn luyện lại THẬT, một lần, rồi mọi test dùng chung — mỗi
/// lượt train 3000 epoch mất vài giây, train lại cho từng test thì bộ test chậm
/// mà không kiểm thêm được gì.
/// </summary>
public sealed class RetrainedModels
{
    public RetrainedModels()
    {
        Clock = new ManualClock(Ans.Now);
        Stack = LearningStackFactory.Build(Clock);
        Registry = new ModelRegistry(Clock);
        Pipeline = new RetrainingPipeline(Registry, clock: Clock);

        V1 = Stack.Datasets.Current;
        First = Pipeline.Run(V1, new TrainingConfig(), "model đầu tiên");

        // Một phản hồi thật: agent hiểu sai ý định, người dùng nói lại nhãn đúng.
        var record = Stack.Journal.Record(
            new GroundedAnswer
            {
                Question = MisreadQuestion,
                Text = "…",
                Verdict = AnswerVerdict.Answered,
                Confidence = 0.8,
                AnsweredAt = Clock.Now,
                Summary = new ReasoningSummary(),
            },
            WrongIntent,
            0.57);

        Stack.Feedback.Submit(record.Id, FeedbackRating.Incorrect, expectedIntent: RightIntent);

        V2 = Stack.Pipeline.Run().NewVersion!;
        Second = Pipeline.Run(V2, new TrainingConfig(), "sau phản hồi người dùng");
    }

    /// <summary>
    /// Một lỗi ĐO ĐƯỢC của model m1: nó xếp "overfitting là gì" vào
    /// <see cref="Intent.Calculate"/> với 57.2% tự tin.
    ///
    /// Câu này chỉ còn đúng một từ khoá sau khi bỏ từ dừng — "overfitting" —
    /// và từ đó xuất hiện trong toàn bộ tập dữ liệu đúng một lần. Nhờ vậy nó
    /// cũng là ca lỗi làm lộ ra chuyện cách chia dữ liệu tất định đang chôn
    /// vùi lời sửa của người dùng (xem <see cref="DataSplitter"/>).
    /// </summary>
    public const string MisreadQuestion = "overfitting là gì";

    public const Intent WrongIntent = Intent.Calculate;

    public const Intent RightIntent = Intent.SearchKnowledge;

    public ManualClock Clock { get; }

    public LearningStack Stack { get; }

    public ModelRegistry Registry { get; }

    public RetrainingPipeline Pipeline { get; }

    public DatasetVersion V1 { get; }

    public DatasetVersion V2 { get; }

    public RetrainingReport First { get; }

    public RetrainingReport Second { get; }
}

[CollectionDefinition("retraining")]
public class RetrainingCollection : ICollectionFixture<RetrainedModels>;

[Collection("retraining")]
public class RetrainingPipelineTests(RetrainedModels fixture)
{
    [Fact]
    public void ModelDauTienDuocTrienKhaiVaDoDuocTrenDuLieuChuaThay()
    {
        var report = fixture.First;

        Assert.True(report.Deployed);
        Assert.Equal(1, report.Candidate.Version);
        Assert.Equal(fixture.V1.Fingerprint, report.Candidate.DatasetFingerprint);
        Assert.True(report.Validation.Total > 0, "phải có ví dụ để đánh giá");
        Assert.True(report.Validation.Accuracy is > 0.5 and < 1.0,
            $"accuracy trên dữ liệu chưa thấy là {report.Validation.Accuracy:P1} — " +
            "không được là 100%, vì 100% chỉ xảy ra khi đo trên chính tập train");
    }

    /// <summary>
    /// Số liệu Phase 11 công bố (100% accuracy) là đo TRÊN TẬP TRAIN. Cùng model,
    /// cùng dữ liệu, đo trên phần chưa thấy thì thấp hơn hẳn — và đó là con số
    /// nói đúng về khả năng khái quát hoá.
    /// </summary>
    [Fact]
    public void AccuracyTrenTapTrainCaoHonHanTrenTapDanhGia()
    {
        var model = fixture.First.Candidate.Restore();

        var onTraining = Evaluator.Evaluate(model, fixture.First.Split.Training);

        Assert.True(onTraining.Accuracy > fixture.First.Validation.Accuracy + 0.2,
            $"train {onTraining.Accuracy:P1} vs đánh giá {fixture.First.Validation.Accuracy:P1}");
    }

    /// <summary>Bộ vàng là chốt chức năng: bảy ý định cơ bản phải còn chạy.</summary>
    [Fact]
    public void BoVangDungHetOCaHaiModel()
    {
        Assert.Equal(fixture.First.Golden.Total, fixture.First.Golden.Correct);
        Assert.Equal(fixture.Second.Golden.Total, fixture.Second.Golden.Correct);
    }

    /// <summary>
    /// TÁI LẬP: cùng tập dữ liệu + cùng cấu hình -> ĐÚNG bộ trọng số ấy, đến
    /// từng chữ số. Đây là lời hứa mà seed, thứ tự từ điển và thứ tự ví dụ được
    /// ghim lại để bảo đảm.
    /// </summary>
    [Fact]
    public void CungDuLieuVaCauHinhThiRaDungModelDo()
    {
        var clock = new ManualClock(Ans.Now);
        var again = new RetrainingPipeline(new ModelRegistry(clock), clock: clock)
            .Run(LearningStackFactory.Build(clock).Datasets.Current, new TrainingConfig());

        Assert.Equal(fixture.First.Candidate.Fingerprint, again.Candidate.Fingerprint);
        Assert.Equal(fixture.First.Candidate.Weights, again.Candidate.Weights);
    }

    [Fact]
    public void DoiSeedThiRaModelKhac()
    {
        var clock = new ManualClock(Ans.Now);
        var other = new RetrainingPipeline(new ModelRegistry(clock), clock: clock)
            .Run(LearningStackFactory.Build(clock).Datasets.Current, new TrainingConfig { Seed = 1234 });

        Assert.NotEqual(fixture.First.Candidate.Fingerprint, other.Candidate.Fingerprint);
    }

    /// <summary>Model lưu lại phải dựng ra một bộ phân loại hành xử y hệt.</summary>
    [Fact]
    public void DungLaiTuBanLuuChoHanhViGiongHetModelVuaTrain()
    {
        var restored = fixture.Second.Candidate.Restore();

        var metrics = Evaluator.Evaluate(restored, fixture.Second.Split.Validation);

        Assert.Equal(fixture.Second.Validation.Accuracy, metrics.Accuracy, 12);
        Assert.Equal(fixture.Second.Validation.MeanLoss, metrics.MeanLoss, 12);
    }

    /// <summary>
    /// VÒNG TRÒN ĐÓNG LẠI Ở ĐÂY: Phase 15 thu một lời sửa của người dùng thành
    /// dữ liệu, Phase 16 biến dữ liệu đó thành HÀNH VI — và chỉ sau khi vượt cửa
    /// đánh giá.
    /// </summary>
    [Fact]
    public void LoiSuaCuaNguoiDungTroThanhHanhViSauKhiTrainLai()
    {
        var before = fixture.First.Candidate.Restore().Classify(RetrainedModels.MisreadQuestion);
        var after = fixture.Second.Candidate.Restore().Classify(RetrainedModels.MisreadQuestion);

        Assert.Equal(RetrainedModels.WrongIntent, before.Intent);
        Assert.Equal(RetrainedModels.RightIntent, after.Intent);
        Assert.True(fixture.Second.Deployed);
    }

    [Fact]
    public void ModelMoiKhongLamHongThuModelCuDangLamDung()
    {
        var comparison = fixture.Second.Comparison!;

        Assert.Empty(comparison.Regressions);
        Assert.True(comparison.MacroF1Delta >= 0, $"macro F1 {comparison.MacroF1Delta:+0.###;-0.###;0}");
    }

    [Fact]
    public void ModelGhiLaiDuDeTruyVeDungPhienBanDuLieu()
    {
        var model = fixture.Second.Candidate;

        Assert.Equal(fixture.V2.Version, model.DatasetVersion);
        Assert.Equal(fixture.V2.Fingerprint, model.DatasetFingerprint);
        Assert.Equal(fixture.V1.Fingerprint, fixture.Stack.Datasets.Get(1)!.Fingerprint);
        Assert.NotNull(fixture.Stack.Datasets.FindByFingerprint(model.DatasetFingerprint));
    }
}

/// <summary>
/// Những test phải LÀM ĐỔI model đang chạy, nên mỗi test tự dựng sổ riêng —
/// dùng chung sổ với các test khác thì thứ tự chạy quyết định kết quả.
/// </summary>
public class ControlledDeploymentTests
{
    private static (RetrainingPipeline Pipeline, ModelRegistry Registry, DatasetVersion Dataset) Build()
    {
        var clock = new ManualClock(Ans.Now);
        var registry = new ModelRegistry(clock);

        return (new RetrainingPipeline(registry, clock: clock), registry,
            LearningStackFactory.Build(clock).Datasets.Current);
    }

    /// <summary>
    /// LUẬT LỚN NHẤT CỦA PHASE 16: model mới KHÔNG tự động thay model đang chạy.
    ///
    /// Cấu hình 1 epoch cho ra một model gần như chưa học gì. Nó bị từ chối, và
    /// model đang chạy vẫn là model cũ — không có nhánh nào trong đường ống đặt
    /// được Active mà không đi qua cửa đánh giá.
    /// </summary>
    [Fact]
    public void ModelTeHonKhongThayDuocModelDangChay()
    {
        var (pipeline, registry, dataset) = Build();

        var good = pipeline.Run(dataset, new TrainingConfig(), "model tốt");
        Assert.True(good.Deployed);

        var bad = pipeline.Run(dataset, new TrainingConfig { Epochs = 1 }, "chưa học gì");

        Assert.False(bad.Deployed);
        Assert.Equal(good.Candidate.Version, registry.Active!.Version);
        Assert.Equal(good.Candidate.Fingerprint, bad.ActiveAfter!.Fingerprint);
        Assert.Equal(ModelEventKind.Rejected, registry.History[^1].Kind);
    }

    [Fact]
    public void ModelBiTuChoiVanCoTrongSoVaLyDoDeDocLai()
    {
        var (pipeline, registry, dataset) = Build();

        pipeline.Run(dataset, new TrainingConfig());
        var bad = pipeline.Run(dataset, new TrainingConfig { Epochs = 1 });

        Assert.NotNull(registry.Get(bad.Candidate.Version));
        Assert.NotEmpty(bad.Decision.Blockers);
        Assert.Contains(bad.Decision.Checks, c => !c.Passed);
    }

    /// <summary>
    /// Quay lui phải đổi HÀNH VI THẬT, không chỉ đổi một con số phiên bản: model
    /// cũ dựng lại từ trọng số đã lưu và trả lời đúng như trước.
    /// </summary>
    [Fact]
    public void QuayLuiTraLaiDungHanhViCuaModelCu()
    {
        var clock = new ManualClock(Ans.Now);
        var stack = LearningStackFactory.Build(clock);
        var registry = new ModelRegistry(clock);
        var pipeline = new RetrainingPipeline(registry, clock: clock);

        var first = pipeline.Run(stack.Datasets.Current, new TrainingConfig());

        var record = stack.Journal.Record(
            new GroundedAnswer
            {
                Question = RetrainedModels.MisreadQuestion,
                Text = "…",
                Verdict = AnswerVerdict.Answered,
                Confidence = 0.8,
                AnsweredAt = clock.Now,
                Summary = new ReasoningSummary(),
            },
            RetrainedModels.WrongIntent,
            0.57);

        stack.Feedback.Submit(record.Id, FeedbackRating.Incorrect, expectedIntent: RetrainedModels.RightIntent);
        var v2 = stack.Pipeline.Run().NewVersion!;

        var second = pipeline.Run(v2, new TrainingConfig());
        Assert.True(second.Deployed);
        Assert.Equal(RetrainedModels.RightIntent,
            registry.RestoreActive().Classify(RetrainedModels.MisreadQuestion).Intent);

        var back = registry.Rollback("thử quay lui");

        Assert.Equal(first.Candidate.Version, back!.Version);
        Assert.Equal(RetrainedModels.WrongIntent,
            registry.RestoreActive().Classify(RetrainedModels.MisreadQuestion).Intent);
    }

    /// <summary>Agent chạy được bằng đúng model đang được duyệt — đây là điểm nối Phase 16 với agent.</summary>
    [Fact]
    public async Task AgentChayDuocBangModelDangDuocDuyet()
    {
        var clock = new ManualClock(Ans.Now);
        var (pipeline, registry, dataset) = Build();

        pipeline.Run(dataset, new TrainingConfig());

        var agent = AgentFactory.Build(registry.RestoreActive(), clock,
            Internet.InternetStackFactory.BuildOffline(clock));

        var response = await agent.HandleAsync("tính tổng của A và B");

        Assert.Equal(Intent.Calculate, response.Intent);
    }
}
