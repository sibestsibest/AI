namespace Ai.Phase14.Grounding;

/// <summary>
/// Thông tin này đến từ ĐÂU.
///
/// Thứ tự trong enum là thứ tự mức tin cậy giảm dần, và nó không tuỳ ý:
/// kiến thức nội bộ đã có người kiểm, lời người dùng nói là sự thật về CHÍNH
/// họ, suy luận đúng nếu tiền đề đúng, còn Internet thì không ai kiểm gì cả.
/// </summary>
public enum SourceKind
{
    /// <summary>Kho kiến thức nội bộ đã được biên tập (<see cref="KnowledgeStore"/>).</summary>
    Knowledge,

    /// <summary>Điều người dùng đã bảo agent ghi nhớ (Phase 10).</summary>
    Memory,

    /// <summary>Người dùng vừa nói trong lượt hội thoại này (<see cref="ConversationContext"/>).</summary>
    Conversation,

    /// <summary>Kết quả suy luận ký hiệu (Phase 9) — đúng nếu tiền đề đúng.</summary>
    Reasoning,

    /// <summary>Lấy từ Internet (Phase 12). KHÔNG ai kiểm chứng.</summary>
    Internet,
}

/// <summary>
/// Mức tin cậy của KÊNH thông tin — không phải của nội dung cụ thể.
///
/// Phân biệt này quan trọng: một trang .gov vẫn có thể viết sai, và một mẩu
/// ký ức người dùng vẫn có thể là họ nhớ lẫn. Trust nói "được phép đối xử với
/// nó như thế nào", chứ không nói "nó đúng".
/// </summary>
public enum TrustLevel
{
    /// <summary>Đã kiểm, được phép dùng như sự thật và được phép thành kiến thức lâu dài.</summary>
    Verified,

    /// <summary>Người dùng tự khai. Đúng về chính họ, nhưng không phải sự thật khách quan.</summary>
    UserProvided,

    /// <summary>Suy ra từ dữ liệu khác. Đúng tới mức tiền đề của nó đúng.</summary>
    Derived,

    /// <summary>Chưa kiểm chứng. KHÔNG được tự động thành kiến thức hay dữ liệu huấn luyện.</summary>
    Untrusted,
}

/// <summary>
/// NGUỒN của một mẩu evidence.
///
/// Vì sao phải là một kiểu riêng thay vì mấy trường rời rạc trên Evidence? Vì
/// nguồn là thứ quyết định mọi thứ khác: được tin đến đâu, có được lưu lại
/// không, có phải nói rõ với người dùng không. Gom vào một chỗ thì không thể
/// vô tình tạo ra một mẩu evidence KHÔNG có nguồn — mà evidence không nguồn
/// thì chỉ là một câu khẳng định trần, tức là chỗ để bịa.
/// </summary>
public sealed record Source
{
    public required SourceKind Kind { get; init; }

    /// <summary>Tên đọc được: tiêu đề tài liệu, id ký ức, tiêu đề trang web.</summary>
    public required string Name { get; init; }

    /// <summary>Chỉ nguồn Internet mới có URL.</summary>
    public Uri? Url { get; init; }

    /// <summary>Lấy về lúc nào. Với Internet thì đây là trường quyết định độ tươi.</summary>
    public DateTimeOffset? RetrievedAt { get; init; }

    /// <summary>
    /// Độ tin cậy nền của kênh, 0..1. Với Internet, giá trị này còn được nhân
    /// thêm uy tín tên miền (<see cref="Internet.SourceRanker"/>).
    /// </summary>
    public double BaseReliability { get; init; }

    public TrustLevel Trust => Kind switch
    {
        SourceKind.Knowledge => TrustLevel.Verified,
        SourceKind.Memory => TrustLevel.UserProvided,
        SourceKind.Conversation => TrustLevel.UserProvided,
        SourceKind.Reasoning => TrustLevel.Derived,
        _ => TrustLevel.Untrusted,
    };

    /// <summary>
    /// Nội dung từ nguồn này có được phép trở thành kiến thức lâu dài không.
    ///
    /// Đây là chốt thực thi nguyên tắc lớn nhất của Phase 13, viết thành MỘT
    /// biểu thức để không ai phải đi đọc cả hệ thống mới biết luật là gì.
    ///
    /// ĐƯỢC: Verified (đã có người kiểm) và UserProvided (người dùng tự khai
    /// về chính họ — họ là nguồn có thẩm quyền về chuyện đó).
    ///
    /// KHÔNG ĐƯỢC: Untrusted, và — chỗ này dễ bỏ sót — cả <see cref="TrustLevel.Derived"/>.
    ///
    /// Vì sao chặn luôn cả Derived, khi suy luận ký hiệu của Phase 9 vốn đúng?
    /// Vì Derived là đầu ra do CHÍNH agent sinh ra. Cho nó tự động thành kiến
    /// thức là mở một vòng lặp tự khuếch đại: agent suy ra một điều, lưu thành
    /// kiến thức, rồi lượt sau lấy chính điều đó làm căn cứ để suy ra điều
    /// tiếp theo — và không có bước nào trong vòng đó đối chiếu lại với thực
    /// tế. Một tiền đề sai lúc đầu sẽ được nhân lên thành cả một kho "kiến
    /// thức" nhất quán mà sai.
    ///
    /// Đây cũng chính là lý do Phase 15 không được huấn luyện từ câu trả lời
    /// do agent tự sinh: cùng một vòng lặp, chỉ khác là nó ăn vào trọng số.
    /// </summary>
    public bool MayBecomeKnowledge => Trust is TrustLevel.Verified or TrustLevel.UserProvided;

    /// <summary>Có phải nói rõ với người dùng rằng thông tin này chưa kiểm chứng.</summary>
    public bool RequiresDisclosure => Trust == TrustLevel.Untrusted;

    // ------------------------------------------------------------------
    // Hàm dựng cho từng kênh — mỗi kênh một chỗ, không rải rác
    // ------------------------------------------------------------------

    public static Source FromKnowledge(string topic, double reliability = 0.95) =>
        new() { Kind = SourceKind.Knowledge, Name = topic, BaseReliability = reliability };

    public static Source FromMemory(string factId, DateTimeOffset createdAt, double reliability = 0.80) =>
        new() { Kind = SourceKind.Memory, Name = factId, RetrievedAt = createdAt, BaseReliability = reliability };

    public static Source FromConversation(int turnIndex, DateTimeOffset at) =>
        new()
        {
            Kind = SourceKind.Conversation,
            Name = $"lượt #{turnIndex}",
            RetrievedAt = at,
            BaseReliability = 0.75,
        };

    public static Source FromReasoning(string statement) =>
        new() { Kind = SourceKind.Reasoning, Name = statement, BaseReliability = 0.90 };

    public static Source FromInternet(string title, Uri url, DateTimeOffset retrievedAt, double reliability) =>
        new()
        {
            Kind = SourceKind.Internet,
            Name = title,
            Url = url,
            RetrievedAt = retrievedAt,
            BaseReliability = reliability,
        };

    /// <summary>Trích dẫn ngắn để in kèm câu trả lời.</summary>
    public string Cite() => Kind switch
    {
        SourceKind.Knowledge => $"kho kiến thức nội bộ: \"{Name}\"",
        SourceKind.Memory => $"bạn đã bảo tôi ghi nhớ ({Name})",
        SourceKind.Conversation => $"bạn vừa nói ở {Name}",
        SourceKind.Reasoning => $"suy luận: {Name}",
        _ => $"web (chưa kiểm chứng): {Url}",
    };

    public override string ToString() => $"[{Kind}] {Name}";
}
