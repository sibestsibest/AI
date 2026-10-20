using Ai.Phase15.Grounding;
using Ai.Phase15.Memory;

namespace Ai.Phase15.Verification;

/// <summary>
/// SỰ THẬT hay SUY RA — phân biệt bắt buộc của Phase 14.
///
/// Đây là phân biệt mà một model sinh văn bản thuần không làm được, vì với nó
/// mọi câu đều là "câu tiếp theo có xác suất cao". Nhưng hai loại câu dưới đây
/// khác nhau về BẢN CHẤT, và người đọc có quyền biết mình đang đọc loại nào:
///
///     Fact      : "Softmax biến logits thành phân bố xác suất."
///                 -> có MỘT mẩu căn cứ nói đúng điều này
///
///     Inference : "Vậy attention có dùng softmax, nên đầu ra của nó là phân bố."
///                 -> KHÔNG mẩu nào nói thế; nó được ghép từ hai mẩu
///
/// Inference có thể sai dù mọi căn cứ đều đúng — chỉ cần bước ghép sai. Nên nó
/// phải được gắn nhãn khác, và bị tính độ tự tin thấp hơn.
/// </summary>
public enum ClaimKind
{
    /// <summary>Nhắc lại trực tiếp một mẩu căn cứ.</summary>
    Fact,

    /// <summary>Ghép từ nhiều mẩu, hoặc do suy luận sinh ra.</summary>
    Inference,
}

/// <summary>
/// MỘT KHẲNG ĐỊNH trong câu trả lời, kèm danh sách căn cứ đỡ cho nó.
///
/// <see cref="EvidenceIds"/> là phần quan trọng nhất và nó KHÔNG được rỗng với
/// một khẳng định hợp lệ. Một câu không trỏ về được mẩu căn cứ nào là một câu
/// agent tự nghĩ ra — dù nó tình cờ đúng, ta không có cách nào biết điều đó,
/// nên nó phải bị <see cref="AnswerValidator"/> chặn lại.
/// </summary>
public sealed record Claim
{
    public required string Id { get; init; }

    public required string Text { get; init; }

    public required ClaimKind Kind { get; init; }

    /// <summary>Id các mẩu <see cref="Evidence"/> đỡ cho khẳng định này.</summary>
    public IReadOnlyList<string> EvidenceIds { get; init; } = [];

    /// <summary>
    /// Mức được đỡ, 0..1 — bao nhiêu phần nội dung của khẳng định tìm thấy
    /// trong các mẩu căn cứ đã dẫn.
    /// </summary>
    public double Support { get; init; }

    /// <summary>Mức tin cậy cao nhất trong số các mẩu căn cứ đã dẫn.</summary>
    public double Reliability { get; init; }

    /// <summary>Số kênh độc lập đỡ cho khẳng định này.</summary>
    public int Agreement { get; init; } = 1;

    /// <summary>Có mẩu căn cứ nào thuộc nguồn đã kiểm chứng.</summary>
    public bool HasVerifiedSupport { get; init; }

    /// <summary>
    /// Ngưỡng tối thiểu để coi một khẳng định là "được đỡ".
    ///
    /// 0.5 nghĩa là: quá nửa nội dung của câu phải tìm thấy trong căn cứ. Dưới
    /// mức đó thì phần lớn câu là chữ của agent, không phải của nguồn.
    /// </summary>
    public const double MinimumSupport = 0.5;

    public bool IsSupported => EvidenceIds.Count > 0 && Support >= MinimumSupport;

    public HashSet<string> Keywords() => Memory.Keywords.Extract(Text);

    public override string ToString() =>
        $"[{Kind}] {Text} (đỡ {Support:F2}, {EvidenceIds.Count} căn cứ)";
}
