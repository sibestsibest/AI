using Ai.Phase15.Agent;
using Ai.Phase15.Verification;

namespace Ai.Phase15.Feedback;

/// <summary>
/// Người dùng đánh giá câu trả lời như thế nào.
///
/// Bốn mức, không nhiều hơn. Thang điểm càng chi tiết (1..10) thì người dùng
/// càng chấm tuỳ tiện, và mọi phân tích sau đó thừa hưởng đúng cái tuỳ tiện
/// đó. Bốn mức này khác nhau ở HÀNH ĐỘNG mà hệ thống phải làm, chứ không khác
/// nhau ở mức độ hài lòng:
///
///   • Correct    -> không phải làm gì, chỉ ghi lại để đo hiệu chuẩn
///   • Incorrect  -> thành error case, và là nguồn dữ liệu huấn luyện nếu
///                   người dùng có nói ĐÚNG là gì
///   • Incomplete -> câu trả lời không sai, nhưng thiếu -> vấn đề của khâu
///                   thu thập căn cứ, KHÔNG phải của model
///   • Unhelpful  -> đúng mà vô dụng (ví dụ "không đủ căn cứ" khi người dùng
///                   cần một câu trả lời) -> vấn đề của kho kiến thức
/// </summary>
public enum FeedbackRating
{
    Correct,
    Incorrect,
    Incomplete,
    Unhelpful,
}

/// <summary>
/// MỘT LẦN PHẢN HỒI của người dùng về MỘT câu trả lời cụ thể.
///
/// Vì sao bắt buộc phải có <see cref="AnswerId"/>? Vì phản hồi mà không gắn
/// được vào một câu trả lời thì không truy được về căn cứ nào, cấu hình nào,
/// quyết định nào — nó chỉ còn là một lời khen hoặc một lời phàn nàn. Muốn
/// biến phản hồi thành dữ liệu sửa lỗi, phải biết chính xác nó đang nói về cái
/// gì. <see cref="FeedbackStore"/> từ chối phản hồi gắn vào một id không có
/// trong <see cref="AnswerJournal"/>.
///
/// HAI TRƯỜNG QUAN TRỌNG NHẤT, và chúng KHÔNG thay thế được nhau:
///
///   <see cref="Correction"/>     — người dùng nói câu trả lời ĐÚNG là gì
///   <see cref="ExpectedIntent"/> — người dùng nói agent đã hiểu SAI Ý ĐỊNH
///
/// Chỉ trường thứ hai sinh ra được dữ liệu huấn luyện, vì thứ duy nhất có
/// trọng số trong hệ thống này là bộ phân loại ý định. Trường thứ nhất là nội
/// dung — nó đi vào kho kiến thức qua cửa <c>TryPromote</c> của Phase 13, chứ
/// không đi vào trọng số. Lẫn hai đường này với nhau là cách nhanh nhất để một
/// câu nói của người dùng trở thành thứ model tin vĩnh viễn.
/// </summary>
public sealed record UserFeedback
{
    public required string Id { get; init; }

    /// <summary>Id do <see cref="AnswerJournal"/> cấp cho câu trả lời được đánh giá.</summary>
    public required string AnswerId { get; init; }

    public required FeedbackRating Rating { get; init; }

    /// <summary>Người dùng nói câu trả lời đúng là gì. Là NỘI DUNG, không phải nhãn.</summary>
    public string? Correction { get; init; }

    /// <summary>Người dùng nói lẽ ra agent phải hiểu câu hỏi theo ý định nào. Là NHÃN.</summary>
    public Intent? ExpectedIntent { get; init; }

    public string? Comment { get; init; }

    public required DateTimeOffset SubmittedAt { get; init; }

    public bool IsNegative => Rating != FeedbackRating.Correct;

    /// <summary>Người dùng có cung cấp NHÃN do chính họ chọn hay không.</summary>
    public bool HasHumanLabel => ExpectedIntent is not null;

    public override string ToString() =>
        $"[{Id}] {Rating} về {AnswerId}" + (ExpectedIntent is null ? "" : $" (lẽ ra: {ExpectedIntent})");
}

/// <summary>
/// Một câu trả lời đã phát ra, kèm id để phản hồi gắn vào được.
///
/// Lưu cả <see cref="Intent"/> mà bộ phân loại đã dự đoán, vì đó là thứ phản
/// hồi tiêu cực cần đối chiếu: người dùng nói "sai rồi" thì việc đầu tiên là
/// xem agent có hiểu đúng câu hỏi không, trước khi kết luận là nó thiếu kiến
/// thức.
///
/// <see cref="Grounded"/> CÓ THỂ NULL, và đó không phải thiếu sót.
///
/// Không phải câu trả lời nào cũng đi qua đường ống kiểm chứng của Phase 14:
/// ý định <c>Calculate</c> đi thẳng vào công cụ ký hiệu, <c>RememberFact</c>
/// chỉ ghi vào bộ nhớ. Những lượt đó không có kết luận và không có độ tự tin —
/// vì không có căn cứ nào được cân.
///
/// Cách xử lý ở đây là ĐỂ NULL, chứ không điền một giá trị mặc định. Điền
/// <c>Confidence = 0</c> hay lấy độ tự tin của bộ phân loại ý định thay vào
/// thì báo cáo hiệu chuẩn sẽ trộn hai đại lượng khác nhau vào cùng một cột:
/// "căn cứ mạnh đến đâu" và "câu hỏi được hiểu chắc đến đâu". Hai con số đó
/// không so được với nhau, và trộn xong thì không ai phát hiện ra là đã trộn.
/// </summary>
public sealed record AnswerRecord
{
    public required string Id { get; init; }

    public required string Question { get; init; }

    public required string AnswerText { get; init; }

    /// <summary>Kết quả kiểm chứng — null nếu lượt này không đi qua đường ống Phase 14.</summary>
    public GroundedAnswer? Grounded { get; init; }

    /// <summary>Ý định mà bộ phân loại đã chọn — null nếu câu trả lời không đi qua nó.</summary>
    public Intent? PredictedIntent { get; init; }

    public double IntentConfidence { get; init; }

    public required DateTimeOffset RecordedAt { get; init; }

    public bool WentThroughVerification => Grounded is not null;

    public AnswerVerdict? Verdict => Grounded?.Verdict;

    public double? Confidence => Grounded?.Confidence;

    public ConfidenceBand? Band => Grounded?.Band;

    public override string ToString() =>
        $"[{Id}] {(Verdict?.ToString() ?? "không qua kiểm chứng")} \"{Question}\"";
}
