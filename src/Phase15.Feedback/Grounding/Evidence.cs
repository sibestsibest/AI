using Ai.Phase15.Internet;
using Ai.Phase15.Memory;

namespace Ai.Phase15.Grounding;

/// <summary>
/// MỘT MẨU EVIDENCE — một nội dung, kèm đầy đủ lai lịch.
///
///     Evidence
///      ├── Content       nội dung (đã an toàn để đọc)
///      ├── Source        từ kênh nào, tên gì
///      ├── Url           nếu là web
///      ├── RetrievedAt   lấy về lúc nào
///      ├── Relevance     liên quan câu hỏi đến đâu   (0..1)
///      └── Reliability   đáng tin đến đâu            (0..1)
///
/// VÌ SAO TÁCH Relevance VÀ Reliability?
///
/// Vì trộn hai thứ này lại là một lỗi rất dễ mắc và rất khó thấy. Một trang
/// blog nói đúng chủ đề đang hỏi thì relevance cao mà reliability thấp. Một
/// tài liệu chuẩn của NIST nói về chuyện khác thì ngược lại. Gộp thành một
/// con số thì hai trường hợp đó trông giống nhau — và agent mất khả năng nói
/// "tôi tìm được đúng chỗ nhưng nguồn không đủ chắc".
///
/// Phase 14 sẽ cần đúng sự tách bạch này để quyết định khi nào nên nói
/// "không đủ căn cứ" thay vì trả lời.
/// </summary>
public sealed record Evidence
{
    public required string Id { get; init; }

    /// <summary>
    /// Nội dung, ở dạng ĐÃ AN TOÀN ĐỂ ĐỌC.
    ///
    /// Với nguồn Internet, đây là bản đã qua <see cref="UntrustedText.Sanitized"/>.
    /// Xem <see cref="FromWeb"/> để biết vì sao việc chuyển kiểu chỉ được xảy
    /// ra ở đúng một chỗ.
    /// </summary>
    public required string Content { get; init; }

    public required Source Source { get; init; }

    /// <summary>Liên quan tới câu hỏi đến đâu — do <see cref="EvidenceRanker"/> tính.</summary>
    public double Relevance { get; init; }

    /// <summary>Đáng tin đến đâu — gộp uy tín kênh, uy tín tên miền, và độ tươi.</summary>
    public double Reliability { get; init; }

    /// <summary>Bao nhiêu nguồn ĐỘC LẬP cùng nói điều này. Mặc định 1 = chỉ mình nó.</summary>
    public int Agreement { get; init; } = 1;

    /// <summary>Nội dung gốc có dấu hiệu cố ra lệnh cho agent (chỉ gặp ở nguồn web).</summary>
    public bool LooksLikeInjection { get; init; }

    public Uri? Url => Source.Url;

    public DateTimeOffset? RetrievedAt => Source.RetrievedAt;

    public SourceKind Kind => Source.Kind;

    public TrustLevel Trust => Source.Trust;

    /// <summary>Có được phép dùng mẩu này để cập nhật kiến thức lâu dài.</summary>
    public bool MayBecomeKnowledge => Source.MayBecomeKnowledge;

    /// <summary>Điểm tổng — do <see cref="EvidenceRanker"/> đặt vào, không tự tính ở đây.</summary>
    public double Score { get; init; }

    /// <summary>Vì sao mẩu này được chấm điểm như vậy.</summary>
    public string Explanation { get; init; } = string.Empty;

    /// <summary>Từ khoá của nội dung — dùng lại bộ tách từ của Phase 10.</summary>
    public HashSet<string> Keywords() => Memory.Keywords.Extract(Content);

    public string Preview(int maxLength = 160) =>
        Content.Length <= maxLength ? Content : string.Concat(Content.AsSpan(0, maxLength).TrimEnd(), "…");

    // ------------------------------------------------------------------
    // Hàm dựng — một hàm cho mỗi kênh
    // ------------------------------------------------------------------

    public static Evidence FromKnowledge(KnowledgeEntry entry) =>
        new()
        {
            Id = $"e-know-{entry.Id}",
            Content = entry.Content,
            Source = Source.FromKnowledge(entry.Topic, entry.Reliability),
        };

    public static Evidence FromMemory(MemoryFact fact) =>
        new()
        {
            Id = $"e-mem-{fact.Id}",
            Content = fact.Content,
            Source = Source.FromMemory(fact.Id, fact.CreatedAt, Math.Clamp(0.65 + 0.25 * fact.Importance, 0, 1)),
        };

    public static Evidence FromConversation(ConversationTurn turn) =>
        new()
        {
            Id = $"e-turn-{turn.Index}",
            Content = turn.UserInput,
            Source = Source.FromConversation(turn.Index, turn.At),
        };

    public static Evidence FromReasoning(string statement, string result) =>
        new()
        {
            Id = $"e-reason-{statement.GetHashCode(StringComparison.Ordinal):x8}",
            Content = result,
            Source = Source.FromReasoning(statement),
        };

    /// <summary>
    /// ĐIỂM GIAO DUY NHẤT giữa dữ liệu web không tin cậy và phần còn lại của hệ thống.
    ///
    /// Phase 12 đặt ra bất biến: <see cref="UntrustedText"/> không có phép
    /// chuyển ngầm sang string, nên muốn lấy chuỗi thô phải gọi tay. Phase 13
    /// cần đưa nội dung web vào một kiểu dùng chung với bốn kênh còn lại, nên
    /// việc chuyển kiểu là KHÔNG TRÁNH ĐƯỢC. Cách xử lý là dồn nó vào đúng
    /// một hàm, và hàm đó bắt buộc làm ba việc:
    ///
    ///   1. Chỉ lấy bản <see cref="UntrustedText.Sanitized"/>, không lấy Raw
    ///   2. Đóng dấu <see cref="SourceKind.Internet"/> -> Trust = Untrusted
    ///      -> <see cref="MayBecomeKnowledge"/> = false, vĩnh viễn
    ///   3. Giữ lại cờ <see cref="LooksLikeInjection"/> để khâu xếp hạng trừ điểm
    ///
    /// Nhờ dồn về một chỗ, câu hỏi "nội dung web vào hệ thống bằng đường nào"
    /// có đúng một câu trả lời, và nó nằm ở đây.
    /// </summary>
    public static Evidence FromWeb(UntrustedText content, Source source, string? idSuffix = null)
    {
        ArgumentNullException.ThrowIfNull(content);
        ArgumentNullException.ThrowIfNull(source);

        if (source.Kind != SourceKind.Internet)
        {
            throw new ArgumentException(
                "FromWeb chỉ nhận nguồn Internet — dùng đúng hàm dựng cho kênh khác.", nameof(source));
        }

        string suffix = idSuffix
            ?? (source.Url?.AbsoluteUri.GetHashCode(StringComparison.Ordinal) ?? 0).ToString("x8");

        return new Evidence
        {
            Id = $"e-web-{suffix}",
            Content = content.Sanitized,
            Source = source,
            LooksLikeInjection = content.LooksLikeInjection,
        };
    }

    public override string ToString() =>
        $"[{Kind}] {Preview(60)} (rel {Relevance:F2}, tin {Reliability:F2}, {Agreement} nguồn)";
}
