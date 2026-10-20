using System.Globalization;
using Ai.Phase15.Agent;
using Ai.Phase15.Verification;

namespace Ai.Phase15.Feedback;

/// <summary>
/// LOẠI LỖI — phân theo KHÂU NÀO HỎNG, không phân theo mức độ người dùng khó chịu.
///
/// Phân loại này quyết định lỗi được sửa bằng cách nào, nên nó phải trỏ vào
/// đúng khâu:
///
///   WrongIntent   -> bộ phân loại ý định. Đây là loại DUY NHẤT sửa được bằng
///                    huấn luyện, vì nó là thứ duy nhất có trọng số.
///   Overconfident -> ngưỡng/công thức độ tự tin. Sai mà còn nói chắc.
///   WrongFact     -> căn cứ sai, hoặc kho kiến thức sai.
///   MissedAnswer  -> khâu thu thập: agent nói không biết trong khi câu trả lời
///                    tồn tại. Thiếu kênh hoặc thiếu kiến thức, không phải lỗi model.
///   Incomplete    -> đúng nhưng thiếu -> số căn cứ được dẫn, hoặc độ phủ truy hồi.
///   Unhelpful     -> đúng mà vô dụng -> khâu diễn đạt câu trả lời.
///
/// Vì sao phân loại lại quan trọng đến vậy? Vì cái bẫy lớn nhất của một vòng
/// phản hồi là đem MỌI lỗi đi huấn luyện. Bốn trong sáu loại trên không sửa
/// được bằng huấn luyện — đổ chúng vào dữ liệu train chỉ làm nhiễu tập dữ liệu
/// và che mất nguyên nhân thật.
/// </summary>
public enum ErrorKind
{
    WrongIntent,
    Overconfident,
    WrongFact,
    MissedAnswer,
    Incomplete,
    Unhelpful,
}

/// <summary>
/// MỘT CA LỖI đã được dựng lại đủ ngữ cảnh.
///
/// Giữ lại <see cref="EvidenceIds"/> và <see cref="Decisions"/> — tức là đúng
/// phần tóm tắt suy luận mà Phase 14 lưu — nên một lỗi đọc lại được sau nhiều
/// tháng: agent đã dùng căn cứ nào, ngưỡng nào đã kích hoạt, vì sao nó kết luận
/// như vậy. Không có phần này thì "câu trả lời sai" chỉ là một dòng than phiền.
/// </summary>
public sealed record ErrorCase
{
    public required string Id { get; init; }

    public required string FeedbackId { get; init; }

    public required string AnswerId { get; init; }

    public required string Question { get; init; }

    public required ErrorKind Kind { get; init; }

    /// <summary>0..1 — dùng để xếp thứ tự sửa, không phải để chấm điểm người dùng.</summary>
    public required double Severity { get; init; }

    /// <summary>Kết luận của Phase 14 — null nếu lượt đó không đi qua đường ống kiểm chứng.</summary>
    public AnswerVerdict? Verdict { get; init; }

    /// <summary>
    /// Độ tự tin agent đã tự nhận — null nếu nó không tự nhận gì.
    ///
    /// Null KHÁC 0: null nghĩa là không có lời khai nào để đối chiếu, còn 0
    /// nghĩa là agent đã nói thẳng rằng nó không có căn cứ. Gộp hai thứ thành
    /// 0 thì mọi lượt trả lời bằng công cụ trông như những lượt agent tự nhận
    /// là không biết gì.
    /// </summary>
    public double? Confidence { get; init; }

    public Intent? PredictedIntent { get; init; }

    public Intent? ExpectedIntent { get; init; }

    public string? Correction { get; init; }

    public IReadOnlyList<string> EvidenceIds { get; init; } = [];

    public IReadOnlyList<string> Decisions { get; init; } = [];

    public required DateTimeOffset CollectedAt { get; init; }

    /// <summary>Ca lỗi này có sinh ra được dữ liệu huấn luyện hay không.</summary>
    public bool IsTrainable => Kind == ErrorKind.WrongIntent && ExpectedIntent is not null;

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "[{0}] {1} (nặng {2:F2}) — \"{3}\"", Id, Kind, Severity, Question);
}
