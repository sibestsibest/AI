using Ai.Phase16.Agent;
using Ai.Phase16.Learning;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Retraining;

/// <summary>Kết quả một lượt huấn luyện lại.</summary>
public sealed record RetrainingReport
{
    public required ModelVersion Candidate { get; init; }

    public required DataSplit Split { get; init; }

    public required EvaluationMetrics Validation { get; init; }

    public required EvaluationMetrics Golden { get; init; }

    /// <summary>So với model đang chạy — null nếu đây là model đầu tiên.</summary>
    public ModelComparison? Comparison { get; init; }

    public required DeploymentDecision Decision { get; init; }

    public required bool Deployed { get; init; }

    /// <summary>Model đang chạy SAU lượt này. Bị từ chối thì đây vẫn là model cũ.</summary>
    public ModelVersion? ActiveAfter { get; init; }

    public override string ToString() =>
        $"{Candidate.Label}: {Validation} | bộ vàng {Golden.Correct}/{Golden.Total} | " +
        (Deployed ? "ĐÃ TRIỂN KHAI" : "KHÔNG triển khai");
}

/// <summary>
/// ĐƯỜNG ỐNG HUẤN LUYỆN LẠI CÓ KIỂM SOÁT — dây chuyền đầy đủ của đề bài Phase 16.
///
///     Tập dữ liệu (Phase 15, có phiên bản + vân tay)
///        ↓
///     Chia tất định            DataSplitter (băm danh tính, không xáo ngẫu nhiên)
///        ↓
///     Train                    ModelTrainer (seed + từ điển có thứ tự)
///        ↓
///     Đánh giá                 Evaluator (tập đánh giá + bộ vàng)
///        ↓
///     So sánh                  ModelComparison (từng ví dụ, không chỉ hiệu số)
///        ↓
///     Cửa triển khai           DeploymentGate (6 điều kiện, mỗi cái một lý do)
///        ↓
///     Sổ đăng ký               ModelRegistry (triển khai / từ chối / quay lui)
///
/// HAI CHỖ DỄ LÀM SAI, và cả hai đều làm mọi con số sau đó vô nghĩa:
///
///   • MODEL CŨ PHẢI ĐƯỢC CHẤM TRÊN CHÍNH TẬP ĐÁNH GIÁ ẤY. Chấm model mới trên
///     tập hôm nay rồi đem so với điểm model cũ đo từ tháng trước là so hai
///     phép đo khác nhau. Ở đây model cũ được chấm lại, mỗi lượt, trên đúng tập
///     vừa chia.
///
///   • MODEL CŨ ĐÃ TỪNG HỌC MỘT PHẦN TẬP ĐÁNH GIÁ HÔM NAY. Tập dữ liệu lớn dần,
///     nên vài ví dụ nay nằm ở tập đánh giá vốn có mặt trong tập train của
///     model cũ. Điều đó làm model cũ trông GIỎI HƠN thực tế, nên phép so là
///     bảo thủ — nó thiên về giữ model cũ. Thà bỏ lỡ một cải tiến còn hơn
///     triển khai một model tưởng là tốt hơn.
///
/// Cả đường ống không có nhánh nào đặt <c>Active</c> mà không đi qua cửa đánh
/// giá. Đó là điều đề bài yêu cầu: model đang chạy không bị thay tự động.
/// </summary>
public sealed class RetrainingPipeline
{
    private readonly ModelRegistry _registry;
    private readonly DeploymentGate _gate;
    private readonly IClock _clock;

    public RetrainingPipeline(ModelRegistry registry, DeploymentGate? gate = null, IClock? clock = null)
    {
        ArgumentNullException.ThrowIfNull(registry);

        _registry = registry;
        _gate = gate ?? new DeploymentGate();
        _clock = clock ?? new SystemClock();
    }

    public ModelRegistry Registry => _registry;

    public RetrainingReport Run(DatasetVersion dataset, TrainingConfig? config = null, string note = "")
    {
        ArgumentNullException.ThrowIfNull(dataset);

        config ??= new TrainingConfig();
        config.Validate();

        // --- 1. Chia dữ liệu, tất định ---
        var split = DataSplitter.Split(dataset, config.ValidationShare);

        // --- 2. Train ---
        var classifier = ModelTrainer.Train(dataset, split, config);

        // --- 3. Đánh giá: tập đánh giá + bộ vàng ---
        //
        // Truyền vào từ khoá của phần TRAIN để báo cáo tách được phần sai do
        // thiếu dữ liệu ra khỏi phần sai do model — với 70 ví dụ, phần đầu
        // chiếm gần hết.
        var trainedWords = Evaluator.WordsIn(split.Training);

        var validation = Evaluator.Evaluate(classifier, split.Validation, trainedWords);
        var golden = Evaluator.Evaluate(classifier, GoldenSet.Examples, trainedWords);

        var candidate = new ModelVersion
        {
            Version = _registry.NextVersionNumber,
            Weights = classifier.ExportWeights(),
            VocabularyWords = [.. classifier.Vocabulary.Words],
            Config = config,
            DatasetVersion = dataset.Version,
            DatasetFingerprint = dataset.Fingerprint,
            Metrics = validation,
            GoldenMetrics = golden,
            TrainedAt = _clock.Now,
            ParentVersion = _registry.Active?.Version,
            Note = note,
        };

        _registry.Register(candidate);

        // --- 4. So với model đang chạy, TRÊN CÙNG TẬP ĐÁNH GIÁ ---
        ModelComparison? comparison = null;

        if (_registry.Active is { } active)
        {
            var baseline = Evaluator.Evaluate(active.Restore(), split.Validation, trainedWords);
            comparison = ModelComparison.Compare(baseline, validation);
        }

        // --- 5. Cửa triển khai ---
        var decision = _gate.Evaluate(validation, golden, comparison, split);

        // --- 6. Chỉ triển khai nếu đạt ---
        _registry.Deploy(candidate, decision);

        return new RetrainingReport
        {
            Candidate = candidate,
            Split = split,
            Validation = validation,
            Golden = golden,
            Comparison = comparison,
            Decision = decision,
            Deployed = decision.Approved,
            ActiveAfter = _registry.Active,
        };
    }
}
