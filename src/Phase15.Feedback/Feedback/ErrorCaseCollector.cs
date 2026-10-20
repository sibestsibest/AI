using Ai.Phase15.Memory;
using Ai.Phase15.Verification;

namespace Ai.Phase15.Feedback;

public interface IErrorCaseCollector
{
    ErrorCase? Collect(UserFeedback feedback, AnswerRecord record);
}

/// <summary>
/// BỘ THU THẬP CA LỖI — biến một lần bấm "sai rồi" thành một ca lỗi đã phân loại.
///
/// THỨ TỰ XÉT KHÔNG ĐẢO ĐƯỢC, và đây là phần đáng đọc kỹ nhất của lớp này:
///
///   1. WrongIntent   xét TRƯỚC TẤT CẢ
///   2. Overconfident
///   3. MissedAnswer
///   4. WrongFact
///   5. Incomplete / Unhelpful
///
/// Vì sao ý định sai phải xét đầu tiên? Vì nó là lỗi Ở THƯỢNG NGUỒN. Hiểu sai
/// câu hỏi thì mọi khâu sau đều chạy đúng trên một câu hỏi SAI: truy hồi đúng
/// cho ý định sai, căn cứ chuẩn cho chuyện không ai hỏi, độ tự tin tính rất
/// cẩn thận cho một câu trả lời lạc đề. Nếu xét theo thứ tự khác, ca lỗi này
/// bị dán nhãn "WrongFact" và sẽ được đem đi sửa kho kiến thức — trong khi kho
/// kiến thức không có lỗi gì cả.
///
/// Và vì sao Overconfident phải tách khỏi WrongFact? Vì sai với độ tự tin 0.25
/// là chuyện hệ thống đã lường trước — nó đã nói là chưa chắc. Sai với độ tự
/// tin 0.9 là một lỗi KHÁC HẲN: nó nói chắc, người dùng tin, và hệ thống không
/// có cách nào tự biết. Gộp hai thứ đó vào một loại thì không ai phát hiện ra
/// công thức độ tự tin đang nói quá.
/// </summary>
public sealed class ErrorCaseCollector : IErrorCaseCollector
{
    private readonly IClock _clock;
    private int _nextId = 1;

    public ErrorCaseCollector(IClock? clock = null) => _clock = clock ?? new SystemClock();

    /// <summary>Từ dải này trở lên thì sai là "sai mà còn nói chắc".</summary>
    public double OverconfidentThreshold { get; init; } = 0.70;

    /// <summary>
    /// Dựng ca lỗi. Trả về null nếu phản hồi là tích cực — không có lỗi nào để thu.
    /// </summary>
    public ErrorCase? Collect(UserFeedback feedback, AnswerRecord record)
    {
        ArgumentNullException.ThrowIfNull(feedback);
        ArgumentNullException.ThrowIfNull(record);

        if (feedback.AnswerId != record.Id)
        {
            throw new ArgumentException(
                $"phản hồi '{feedback.Id}' nói về '{feedback.AnswerId}', không phải '{record.Id}'",
                nameof(record));
        }

        if (!feedback.IsNegative) return null;

        var (kind, severity) = Classify(feedback, record);

        return new ErrorCase
        {
            Id = $"e{_nextId++}",
            FeedbackId = feedback.Id,
            AnswerId = record.Id,
            Question = record.Question,
            Kind = kind,
            Severity = severity,
            Verdict = record.Verdict,
            Confidence = record.Confidence,
            PredictedIntent = record.PredictedIntent,
            ExpectedIntent = feedback.ExpectedIntent,
            Correction = feedback.Correction,
            EvidenceIds = record.Grounded?.Summary.EvidenceIds ?? [],
            Decisions = record.Grounded?.Summary.Decisions ?? [],
            CollectedAt = _clock.Now,
        };
    }

    /// <summary>
    /// Phân loại + tính độ nặng.
    ///
    /// Độ nặng KHÔNG phải một thang cảm tính. Nó là con số dùng để xếp thứ tự
    /// sửa, nên nó phải phản ánh THIỆT HẠI: một câu sai mà agent nói chắc thì
    /// người dùng tin và hành động theo, còn một câu sai mà agent đã cảnh báo
    /// là chưa chắc thì họ còn cơ hội tự kiểm. Vì vậy Overconfident lấy luôn
    /// chính độ tự tin làm độ nặng — càng nói chắc càng nặng.
    /// </summary>
    private (ErrorKind Kind, double Severity) Classify(UserFeedback feedback, AnswerRecord record)
    {
        // --- 1. Hiểu sai ý định: lỗi thượng nguồn, xét trước tất cả ---
        if (feedback.ExpectedIntent is not null && feedback.ExpectedIntent != record.PredictedIntent)
        {
            // Nặng hơn nếu bộ phân loại lại còn rất tự tin về cái nhãn sai đó.
            return (ErrorKind.WrongIntent, Math.Clamp(0.5 + 0.5 * record.IntentConfidence, 0, 1));
        }

        // --- 2. Sai mà còn nói chắc ---
        //
        // Chỉ xét được khi agent CÓ tự nhận một độ tự tin. Lượt trả lời bằng
        // công cụ không tự nhận gì, nên nó không thể "nói quá" — nó rơi xuống
        // WrongFact ở cửa 4.
        if (feedback.Rating == FeedbackRating.Incorrect
            && record.Confidence is { } confidence
            && confidence >= OverconfidentThreshold)
        {
            return (ErrorKind.Overconfident, confidence);
        }

        // --- 3. Bỏ lỡ: agent nói không biết, người dùng thì có câu trả lời ---
        if (record.Verdict == AnswerVerdict.Insufficient)
        {
            // Có correction nghĩa là câu trả lời TỒN TẠI mà agent không tìm ra
            // -> nặng hơn, vì đây là lỗ hổng truy hồi chỉ ra được ngay.
            return (ErrorKind.MissedAnswer, feedback.Correction is null ? 0.40 : 0.60);
        }

        // --- 4. Sai nội dung ---
        if (feedback.Rating == FeedbackRating.Incorrect)
        {
            return (ErrorKind.WrongFact, 0.50 + 0.3 * (record.Confidence ?? 0));
        }

        // --- 5. Không sai, nhưng chưa dùng được ---
        return feedback.Rating == FeedbackRating.Incomplete
            ? (ErrorKind.Incomplete, 0.30)
            : (ErrorKind.Unhelpful, 0.25);
    }
}
