using Ai.Phase16.Grounding;

namespace Ai.Phase16.Verification;

/// <summary>Kết luận cuối của Phase 14 về một câu hỏi.</summary>
public enum AnswerVerdict
{
    /// <summary>Có căn cứ đủ và nhất quán — trả lời được.</summary>
    Answered,

    /// <summary>Có căn cứ nhưng yếu hoặc thiếu nguồn kiểm chứng — trả lời kèm nói rõ là chưa chắc.</summary>
    Uncertain,

    /// <summary>Các căn cứ nói ngược nhau — trình bày cả hai phía, KHÔNG chọn bên.</summary>
    Conflicted,

    /// <summary>Không có căn cứ nào đáng kể — nói thẳng là không biết.</summary>
    Insufficient,
}

/// <summary>Dải độ tự tin, để in ra cho người đọc thay vì bắt họ đọc số thập phân.</summary>
public enum ConfidenceBand
{
    VeryLow,
    Low,
    Medium,
    High,
}

/// <summary>
/// TÓM TẮT SUY LUẬN — cố ý là TÓM TẮT, không phải diễn biến suy nghĩ.
///
/// Phase 14 KHÔNG lưu và KHÔNG in ra dòng suy nghĩ nội bộ. Thứ được lưu ở đây
/// là siêu dữ liệu có cấu trúc:
///
///   • các quyết định đã ra và ngưỡng nào đã kích hoạt
///   • id của các mẩu căn cứ được dẫn
///   • con số đếm được (bao nhiêu kênh, bao nhiêu mâu thuẫn)
///
/// Vì sao phân biệt này quan trọng? Vì hai thứ đó phục vụ hai mục đích khác
/// nhau. Người dùng cần biết KẾT LUẬN dựa trên cái gì và có đáng tin không —
/// những điều trong danh sách trên trả lời đúng câu đó, và trả lời bằng dữ
/// liệu kiểm chứng lại được. Còn diễn biến suy nghĩ thô thì dài, lẫn cả những
/// nhánh đã bị loại, và đọc nó dễ dẫn tới tin vào một lập luận mà hệ thống đã
/// tự bỏ đi.
///
/// Nên: lưu quyết định và tham chiếu, không lưu độc thoại.
/// </summary>
public sealed record ReasoningSummary
{
    /// <summary>Các bước quyết định, mỗi bước một câu ngắn.</summary>
    public IReadOnlyList<string> Decisions { get; init; } = [];

    /// <summary>Id các mẩu căn cứ đã dùng — để truy lại, không phải để kể lại.</summary>
    public IReadOnlyList<string> EvidenceIds { get; init; } = [];

    public int ChannelCount { get; init; }

    public int ContradictionCount { get; init; }

    public int FactCount { get; init; }

    public int InferenceCount { get; init; }

    public override string ToString() => string.Join("; ", Decisions);
}

/// <summary>
/// CÂU TRẢ LỜI CÓ CƠ SỞ — đầu ra cuối cùng của Phase 14.
///
/// Khác biệt với một chuỗi văn bản do model sinh ra: mỗi câu trong
/// <see cref="Text"/> đều truy được về một <see cref="Claim"/>, mỗi claim truy
/// được về các mẩu <see cref="Evidence"/>, và mỗi mẩu evidence truy được về
/// một <see cref="Source"/>. Không có chỗ nào trong chuỗi đó là "agent tự
/// nghĩ ra" — và nếu có, <see cref="AnswerValidator"/> sẽ bắt được.
/// </summary>
public sealed record GroundedAnswer
{
    public required string Question { get; init; }

    public required string Text { get; init; }

    public required AnswerVerdict Verdict { get; init; }

    /// <summary>0..1.</summary>
    public required double Confidence { get; init; }

    public IReadOnlyList<Claim> Claims { get; init; } = [];

    public IReadOnlyList<Contradiction> Contradictions { get; init; } = [];

    /// <summary>Các mẩu căn cứ thực sự được dẫn trong câu trả lời.</summary>
    public IReadOnlyList<Evidence> CitedEvidence { get; init; } = [];

    public IReadOnlyList<string> Warnings { get; init; } = [];

    public required ReasoningSummary Summary { get; init; }

    public required DateTimeOffset AnsweredAt { get; init; }

    /// <summary>Có phải nói rõ với người dùng rằng căn cứ chưa được kiểm chứng.</summary>
    public bool RequiresDisclosure { get; init; }

    /// <summary>Kết quả kiểm tra cuối — do <see cref="AnswerValidator"/> đặt vào.</summary>
    public ValidationReport? Validation { get; init; }

    public ConfidenceBand Band => Confidence switch
    {
        >= 0.70 => ConfidenceBand.High,
        >= 0.45 => ConfidenceBand.Medium,
        >= 0.20 => ConfidenceBand.Low,
        _ => ConfidenceBand.VeryLow,
    };

    /// <summary>Agent có thực sự đưa ra một câu trả lời, hay chỉ nói là nó không biết.</summary>
    public bool IsAnswer => Verdict is AnswerVerdict.Answered or AnswerVerdict.Uncertain;

    public IReadOnlyList<Claim> Facts => [.. Claims.Where(c => c.Kind == ClaimKind.Fact)];

    public IReadOnlyList<Claim> Inferences => [.. Claims.Where(c => c.Kind == ClaimKind.Inference)];

    /// <summary>Mọi khẳng định đều có căn cứ đỡ.</summary>
    public bool AllClaimsSupported => Claims.All(c => c.IsSupported);

    public override string ToString() => $"[{Verdict}, {Confidence:P0}] {Text}";
}
