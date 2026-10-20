using Ai.Phase16.Agent;
using Ai.Phase16.Feedback;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Learning;

/// <summary>
/// Bốn mảnh của vòng phản hồi, dựng sẵn và nối đúng thứ tự.
///
/// Nối đúng thứ tự là phần dễ làm sai nhất: <see cref="FeedbackStore"/> phải
/// dùng CHÍNH <see cref="AnswerJournal"/> mà agent đang ghi vào, nếu không thì
/// mọi phản hồi đều bị từ chối vì "không có câu trả lời nào mang id đó".
/// </summary>
public sealed record LearningStack(
    AnswerJournal Journal,
    FeedbackStore Feedback,
    DatasetRepository Datasets,
    LearningPipeline Pipeline);

/// <summary>Dựng sẵn vòng phản hồi — dùng cho demo và test.</summary>
public static class LearningStackFactory
{
    /// <summary>
    /// Dựng vòng phản hồi với tập dữ liệu v1 = đúng bộ ví dụ đang dạy agent.
    ///
    /// Vì sao v1 phải là CHÍNH <see cref="IntentTrainingData.Examples"/>?
    ///
    /// Vì Phase 16 sẽ so model mới với model đang chạy, và so sánh đó chỉ có
    /// nghĩa khi biết model đang chạy học từ đâu. Nếu v1 là một tập khác thì
    /// mọi khác biệt đo được sau này không phân biệt nổi là do ví dụ mới thêm,
    /// hay chỉ do hai model vốn đã train từ hai tập khác nhau ngay từ đầu.
    /// </summary>
    public static LearningStack Build(IClock? clock = null, int journalCapacity = 200)
    {
        var journal = new AnswerJournal(clock, journalCapacity);
        var feedback = new FeedbackStore(journal, clock);

        var datasets = new DatasetRepository(clock);
        datasets.Seed(IntentTrainingData.Examples, "bộ ví dụ gốc đang dạy agent (Phase 11)");

        var pipeline = new LearningPipeline(feedback, datasets, clock: clock);

        return new LearningStack(journal, feedback, datasets, pipeline);
    }
}
