using Ai.Phase16.Embeddings;
using Ai.Phase16.Feedback;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Learning;

/// <summary>Kết quả một lượt học từ phản hồi.</summary>
public sealed record LearningReport
{
    public int FeedbackProcessed { get; init; }

    public IReadOnlyList<ErrorCase> ErrorCases { get; init; } = [];

    public IReadOnlyList<TrainingCandidate> Candidates { get; init; } = [];

    public IReadOnlyList<TrainingCandidate> Approved { get; init; } = [];

    public IReadOnlyList<TrainingCandidate> Rejected { get; init; } = [];

    /// <summary>Ca lỗi không sinh được ứng viên, kèm lý do và khâu cần sửa.</summary>
    public IReadOnlyList<string> Skipped { get; init; } = [];

    public IReadOnlyList<string> Warnings { get; init; } = [];

    /// <summary>Phiên bản tập dữ liệu mới — null nếu lượt này không có gì được duyệt.</summary>
    public DatasetVersion? NewVersion { get; init; }

    public bool ProducedNewVersion => NewVersion is not null;

    public override string ToString() =>
        $"{FeedbackProcessed} phản hồi -> {ErrorCases.Count} ca lỗi -> {Candidates.Count} ứng viên -> " +
        $"duyệt {Approved.Count}" + (NewVersion is null ? ", không tạo phiên bản mới" : $" -> {NewVersion}");
}

/// <summary>
/// ĐƯỜNG ỐNG HỌC TỪ PHẢN HỒI — dây chuyền đầy đủ của đề bài Phase 15.
///
///     Phản hồi người dùng
///        ↓
///     Ca lỗi đã phân loại          ErrorCaseCollector
///        ↓
///     Ứng viên dữ liệu huấn luyện  CandidateGenerator
///        ↓
///     KIỂM                         CandidateValidator (6 cửa)
///        ↓
///     Phiên bản tập dữ liệu mới    DatasetRepository (append-only)
///
/// ĐIỀU QUAN TRỌNG NHẤT VỀ LỚP NÀY LÀ THỨ NÓ KHÔNG CÓ.
///
/// Nó không giữ tham chiếu nào tới <see cref="Agent.IntentClassifier"/>,
/// <see cref="Network.NeuralNetwork"/> hay bất kỳ optimizer nào. Đầu ra duy
/// nhất của nó là một <see cref="DatasetVersion"/> — dữ liệu, không phải trọng
/// số. Chạy nó một nghìn lần thì model vẫn nguyên như trước.
///
/// Vì sao lại cố tình tự trói tay như vậy? Vì "học từ phản hồi" mà cập nhật
/// trọng số ngay là một hệ thống không ai kiểm soát được: mỗi phản hồi làm đổi
/// hành vi, không có bước nào so sánh model mới với model cũ, và không có
/// đường quay lui. Hỏng thì chỉ biết là "hôm nay nó trả lời khác hôm qua".
///
/// Tách làm hai phase là để có một điểm dừng mà con người xem được cái sắp
/// được học, TRƯỚC khi nó thành trọng số. Việc train là của Phase 16, và ở đó
/// còn một cửa nữa: model mới chỉ được thay model cũ nếu đánh giá đạt.
/// </summary>
public sealed class LearningPipeline
{
    private readonly FeedbackStore _feedback;
    private readonly DatasetRepository _datasets;
    private readonly IErrorCaseCollector _collector;
    private readonly CandidateGenerator _generator;
    private readonly CandidateValidator _validator;
    private readonly HashSet<string> _processed = new(StringComparer.Ordinal);
    private readonly List<ErrorCase> _errorCases = [];

    public LearningPipeline(
        FeedbackStore feedback,
        DatasetRepository datasets,
        IErrorCaseCollector? collector = null,
        CandidateGenerator? generator = null,
        CandidateValidator? validator = null,
        IClock? clock = null)
    {
        ArgumentNullException.ThrowIfNull(feedback);
        ArgumentNullException.ThrowIfNull(datasets);

        _feedback = feedback;
        _datasets = datasets;
        _collector = collector ?? new ErrorCaseCollector(clock);
        _generator = generator ?? new CandidateGenerator(clock);
        _validator = validator ?? new CandidateValidator();
    }

    /// <summary>Toàn bộ ca lỗi đã thu được, cộng dồn qua các lượt chạy.</summary>
    public IReadOnlyList<ErrorCase> ErrorCases => _errorCases;

    /// <summary>Ca lỗi nặng nhất trước — thứ tự nên sửa.</summary>
    public IReadOnlyList<ErrorCase> TopErrorCases(int take = 5) =>
        [.. _errorCases.OrderByDescending(e => e.Severity).ThenBy(e => e.Id, StringComparer.Ordinal).Take(take)];

    /// <summary>Ca lỗi theo khâu — để biết nên sửa chỗ nào trước, thay vì sửa chỗ dễ nhất.</summary>
    public IReadOnlyDictionary<ErrorKind, int> ErrorsByKind() =>
        _errorCases.GroupBy(e => e.Kind).ToDictionary(g => g.Key, g => g.Count());

    /// <summary>
    /// Chạy một lượt học trên những phản hồi CHƯA XỬ LÝ.
    ///
    /// Nhớ phản hồi nào đã xử lý (<c>_processed</c>) để chạy lại nhiều lần
    /// không nhân bản ứng viên. Không có nó thì mỗi lượt chạy lại sinh lại toàn
    /// bộ ứng viên cũ; chúng bị cửa chống trùng loại nên tập dữ liệu vẫn đúng,
    /// nhưng báo cáo thì đầy "đã có trong tập" và lấp hết những lý do loại thật
    /// sự đáng đọc.
    /// </summary>
    public LearningReport Run(Vocabulary? vocabulary = null, string note = "học từ phản hồi người dùng")
    {
        var fresh = _feedback.All.Where(f => !_processed.Contains(f.Id)).ToList();

        // --- 1. Phản hồi -> ca lỗi ---
        var newCases = new List<ErrorCase>();

        foreach (var feedback in fresh)
        {
            _processed.Add(feedback.Id);

            var record = _feedback.AnswerFor(feedback);
            if (record is null) continue;          // câu trả lời đã rơi khỏi sổ

            var errorCase = _collector.Collect(feedback, record);
            if (errorCase is null) continue;       // phản hồi tích cực

            newCases.Add(errorCase);
            _errorCases.Add(errorCase);
        }

        // --- 2. Ca lỗi -> ứng viên ---
        var (candidates, skipped) = _generator.FromErrorCases(newCases);

        // --- 3. KIỂM ---
        var outcome = _validator.Validate(candidates, _datasets.Current, vocabulary);

        // --- 4. Phiên bản mới, chỉ khi có gì được duyệt ---
        //
        // Không tạo phiên bản rỗng. Một phiên bản không thêm ví dụ nào vẫn có
        // vân tay giống cha, nên nó chỉ làm lịch sử dài ra mà không nói thêm
        // điều gì — và làm "model này train từ v7 hay v8" thành câu hỏi vô
        // nghĩa vì hai phiên bản đó y hệt nhau.
        DatasetVersion? newVersion = null;

        if (outcome.Approved.Count > 0)
        {
            newVersion = _datasets.Commit(outcome.Approved, note, outcome.RejectionReasons);
        }

        return new LearningReport
        {
            FeedbackProcessed = fresh.Count,
            ErrorCases = newCases,
            Candidates = candidates,
            Approved = outcome.Approved,
            Rejected = outcome.Rejected,
            Skipped = skipped,
            Warnings = outcome.Warnings,
            NewVersion = newVersion,
        };
    }
}
