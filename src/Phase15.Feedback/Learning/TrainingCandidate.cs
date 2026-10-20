using Ai.Phase15.Agent;

namespace Ai.Phase15.Learning;

/// <summary>
/// NHÃN NÀY ĐẾN TỪ ĐÂU — trường quyết định của cả Phase 15.
///
/// Thứ tự trong enum là thứ tự quyền được thành dữ liệu huấn luyện, giảm dần,
/// và nó song song với <see cref="Grounding.SourceKind"/> của Phase 13 một
/// cách có chủ ý: cùng một câu hỏi ("thông tin này đến từ đâu"), chỉ khác là ở
/// đây hậu quả nặng hơn — nó đi vào TRỌNG SỐ chứ không chỉ vào câu trả lời.
///
///   CuratedSeed     người viết hệ thống tự tay gán nhãn        -> ĐƯỢC
///   UserCorrection  người dùng NÓI RÕ lẽ ra phải là nhãn nào   -> ĐƯỢC
///   AgentAnswer     nhãn do chính model dự đoán                -> KHÔNG
///   InternetContent lấy từ web                                 -> KHÔNG BAO GIỜ
///
/// VÌ SAO <see cref="AgentAnswer"/> BỊ CHẶN, KỂ CẢ KHI NGƯỜI DÙNG BẤM "ĐÚNG"?
///
/// Vì người dùng đang chấm CÂU TRẢ LỜI, không phải xác nhận CÁI NHÃN bên trong.
/// Họ đọc "2 + 3 * 4 = 14", thấy đúng, bấm đúng — và họ không hề nói gì về
/// việc câu đó được xếp vào ý định <c>Calculate</c> hay <c>QueryDatabase</c>.
/// Lấy nhãn do model tự đoán rồi đem dạy lại chính model là đóng một vòng
/// không có thực tế nào đi vào: model tự khẳng định mình, sai lệch ban đầu
/// được nhân lên mỗi lượt, và accuracy trên tập tự gán nhãn vẫn đẹp suốt quá
/// trình đó.
///
/// Đây đúng là vòng lặp mà Phase 13 đã chặn ở mức kiến thức
/// (<see cref="Grounding.TrustLevel.Derived"/> không được thành kiến thức).
/// Phase 15 chặn cùng vòng lặp đó ở mức trọng số. Cùng một luật, hai tầng —
/// vì mất một tầng là đủ để vòng lặp đóng lại.
/// </summary>
public enum CandidateProvenance
{
    CuratedSeed,
    UserCorrection,
    AgentAnswer,
    InternetContent,
}

/// <summary>Trạng thái của một ứng viên sau khi qua khâu kiểm.</summary>
public enum CandidateStatus
{
    Pending,
    Approved,
    Rejected,
}

/// <summary>
/// MỘT ỨNG VIÊN DỮ LIỆU HUẤN LUYỆN — chưa phải dữ liệu huấn luyện.
///
/// Khoảng cách giữa hai khái niệm đó là toàn bộ nội dung của Phase 15:
///
///     phản hồi -> ứng viên -> KIỂM -> tập dữ liệu đã duyệt -> (Phase 16) train
///
/// Không có mũi tên nào đi tắt. Đặc biệt là không có mũi tên nào từ phản hồi
/// đi thẳng tới trọng số.
/// </summary>
public sealed record TrainingCandidate
{
    public required string Id { get; init; }

    /// <summary>Câu người dùng đã gõ — đầu vào của bộ phân loại.</summary>
    public required string Input { get; init; }

    /// <summary>Nhãn ĐÚNG, theo người gán nhãn.</summary>
    public required Intent Label { get; init; }

    public required CandidateProvenance Provenance { get; init; }

    /// <summary>Phản hồi nào sinh ra ứng viên này — để truy nguồn.</summary>
    public string? FeedbackId { get; init; }

    /// <summary>Ca lỗi nào sinh ra ứng viên này.</summary>
    public string? ErrorCaseId { get; init; }

    public required DateTimeOffset CreatedAt { get; init; }

    public CandidateStatus Status { get; init; } = CandidateStatus.Pending;

    /// <summary>Vì sao được duyệt / bị loại. Luôn có, kể cả khi được duyệt.</summary>
    public string? Decision { get; init; }

    /// <summary>
    /// Nguồn của nhãn này có quyền thành dữ liệu huấn luyện hay không.
    ///
    /// Viết thành MỘT biểu thức, đúng như <c>Source.MayBecomeKnowledge</c> của
    /// Phase 13, để luật đọc được ở một chỗ thay vì phải suy ra từ cả hệ thống.
    /// </summary>
    public bool MayBecomeTrainingData =>
        Provenance is CandidateProvenance.CuratedSeed or CandidateProvenance.UserCorrection;

    /// <summary>Danh tính chuẩn hoá, cùng cách tính với <see cref="DatasetExample.Key"/>.</summary>
    public string Key => $"{Label}|{Memory.Keywords.Canonical(Input)}";

    public override string ToString() => $"[{Id}] \"{Input}\" -> {Label} ({Provenance}, {Status})";
}
