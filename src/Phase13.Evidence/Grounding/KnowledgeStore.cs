using Ai.Phase13.Memory;

namespace Ai.Phase13.Grounding;

/// <summary>
/// Một mục kiến thức nội bộ — đã được biên tập, được phép dùng như sự thật.
///
/// Khác <see cref="MemoryFact"/> ở chỗ: ký ức là chuyện về NGƯỜI DÙNG ("tôi
/// dùng C#"), còn kiến thức là chuyện về THẾ GIỚI ("softmax biến logits thành
/// phân bố xác suất"). Ký ức phai đi theo thời gian; kiến thức thì không, nó
/// chỉ bị thay thế khi có người sửa.
/// </summary>
public sealed record KnowledgeEntry
{
    public required string Id { get; init; }

    /// <summary>Chủ đề — đóng vai trò như tiêu đề tài liệu.</summary>
    public required string Topic { get; init; }

    public required string Content { get; init; }

    public required IReadOnlySet<string> TopicKeywords { get; init; }

    public required IReadOnlySet<string> AllKeywords { get; init; }

    public double Reliability { get; init; } = 0.95;

    public required DateTimeOffset AddedAt { get; init; }

    /// <summary>
    /// Mục này vào kho bằng đường nào. Phase 13 chỉ có "curated"; Phase 15 sẽ
    /// thêm đường "đã qua kiểm duyệt từ phản hồi người dùng". Ghi lại từ bây
    /// giờ để về sau truy được nguồn gốc mọi mục trong kho.
    /// </summary>
    public string Origin { get; init; } = "curated";

    public override string ToString() => $"[{Id}] {Topic}";
}

public sealed record ScoredKnowledge(KnowledgeEntry Entry, double Score);

/// <summary>Vì sao một mẩu evidence được / không được nhận vào kho kiến thức.</summary>
public sealed record PromotionResult(bool Accepted, string Reason, KnowledgeEntry? Entry = null);

/// <summary>
/// KHO KIẾN THỨC NỘI BỘ — phần "Knowledge" của Phase 13.
///
/// Tìm kiếm dùng lại đúng cách chấm điểm của SearchTool (Phase 11):
/// độ khớp toàn văn cộng nửa độ khớp CHỦ ĐỀ, vì chủ đề nói tài liệu NÓI VỀ
/// cái gì, còn thân bài chỉ nói nó CÓ NHẮC tới cái gì.
///
/// ============================================================
/// ĐÂY LÀ CHỖ THỰC THI NGUYÊN TẮC LỚN NHẤT CỦA PHASE 13
/// ============================================================
///
/// Kho này KHÔNG có hàm nào nhận <see cref="Internet.WebDocument"/>. Cách duy
/// nhất để một mẩu evidence vào được kho là <see cref="TryPromote"/>, và hàm
/// đó TỪ CHỐI mọi nguồn có <see cref="TrustLevel.Untrusted"/>.
///
/// Vì sao phải làm thành một hàm từ chối tường minh, thay vì chỉ "không viết
/// hàm thêm vào từ web"? Vì cái không tồn tại thì không test được. Một hàm
/// trả về lý do từ chối thì viết được unit test khẳng định luật vẫn còn đó —
/// và nó sẽ đỏ ngay nếu sau này có ai nới luật ra.
///
/// Nếu bỏ chốt này, đường đi sẽ là: một trang web bất kỳ -> kiến thức lâu dài
/// -> dữ liệu huấn luyện (Phase 15) -> trọng số model. Tức là bất kỳ ai đăng
/// được một trang web cũng dạy được model. Đó là lý do luật này tồn tại.
/// </summary>
public sealed class KnowledgeStore
{
    private readonly List<KnowledgeEntry> _entries = [];
    private readonly IClock _clock;
    private int _nextId = 1;

    public KnowledgeStore(IClock? clock = null) => _clock = clock ?? new SystemClock();

    public int Count => _entries.Count;

    public IReadOnlyList<KnowledgeEntry> Entries => _entries;

    /// <summary>Khớp dưới ngưỡng này thì coi như không tìm thấy — thà không trả gì hơn trả thứ lạc đề.</summary>
    public double MinimumRelevance { get; init; } = 0.25;

    /// <summary>Thêm kiến thức đã được biên tập. Đây là đường dành cho người, không phải cho máy.</summary>
    public KnowledgeEntry Add(string topic, string content, double reliability = 0.95, string origin = "curated")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(topic);
        ArgumentException.ThrowIfNullOrWhiteSpace(content);

        var topicKeywords = Keywords.Extract(topic);
        var allKeywords = new HashSet<string>(topicKeywords, StringComparer.OrdinalIgnoreCase);
        allKeywords.UnionWith(Keywords.Extract(content));

        var entry = new KnowledgeEntry
        {
            Id = $"k{_nextId++}",
            Topic = topic.Trim(),
            Content = content.Trim(),
            TopicKeywords = topicKeywords,
            AllKeywords = allKeywords,
            Reliability = Math.Clamp(reliability, 0, 1),
            AddedAt = _clock.Now,
            Origin = origin,
        };

        _entries.Add(entry);
        return entry;
    }

    /// <summary>
    /// Xin đưa một mẩu evidence vào kho kiến thức.
    ///
    /// Bốn cửa, theo thứ tự:
    ///
    ///   1. Nguồn không tin cậy       -> TỪ CHỐI (luật cứng, không có ngoại lệ)
    ///   2. Nội dung có dấu hiệu tiêm -> TỪ CHỐI
    ///   3. Đã có mục tương đương     -> TỪ CHỐI (không nhân bản)
    ///   4. Còn lại                   -> nhận
    /// </summary>
    public PromotionResult TryPromote(Evidence evidence, string? topic = null)
    {
        ArgumentNullException.ThrowIfNull(evidence);

        // --- Cửa 1: LUẬT CỨNG CỦA PHASE 13 ---
        if (!evidence.MayBecomeKnowledge)
        {
            return new PromotionResult(false,
                $"nguồn {evidence.Kind} có mức tin cậy {evidence.Trust} — nội dung chưa kiểm chứng " +
                "không được tự động thành kiến thức lâu dài");
        }

        // --- Cửa 2 ---
        if (evidence.LooksLikeInjection)
        {
            return new PromotionResult(false, "nội dung có dấu hiệu cố ra lệnh cho agent");
        }

        // --- Cửa 3 ---
        string canonical = Keywords.Canonical(evidence.Content);
        var duplicate = _entries.FirstOrDefault(e =>
            Keywords.Canonical(e.Content).Equals(canonical, StringComparison.OrdinalIgnoreCase));

        if (duplicate is not null)
        {
            return new PromotionResult(false, $"đã có trong kho ({duplicate.Id})", duplicate);
        }

        // --- Cửa 4 ---
        var entry = Add(
            topic ?? evidence.Source.Name,
            evidence.Content,
            evidence.Reliability > 0 ? evidence.Reliability : evidence.Source.BaseReliability,
            origin: $"promoted:{evidence.Kind}");

        return new PromotionResult(true, $"đã nhận vào kho ({entry.Id})", entry);
    }

    /// <summary>
    /// Tìm kiến thức liên quan, xếp theo điểm giảm dần.
    ///
    /// CÔNG THỨC — ba tín hiệu, và tín hiệu thứ ba là tín hiệu quan trọng nhất:
    ///
    ///     score = (body + 0.5·topic + 1.0·topicCoverage) / 2.5
    ///
    ///     body          = |query ∩ toàn văn| / |query|     câu hỏi được phủ bao nhiêu
    ///     topic         = |query ∩ chủ đề|   / |query|
    ///     topicCoverage = |query ∩ chủ đề|   / |chủ đề|     CHỦ ĐỀ được phủ bao nhiêu
    ///
    /// VÌ SAO CẦN topicCoverage — tức là đo theo CẢ HAI CHIỀU?
    ///
    /// Hai lỗi dưới đây đều là lỗi thật, và cả hai đều do chỉ đo một chiều:
    ///
    ///   (1) Hỏi "softmax hoạt động thế nào" (4 từ) thì tài liệu "Softmax" chỉ
    ///       khớp 1/4 = 0.25 — thấp, dù nó là tài liệu ĐÚNG TÊN chủ đề. Chia
    ///       cho độ dài câu hỏi thì câu hỏi càng dài, tài liệu đúng càng bị
    ///       phạt. topicCoverage = 1.0 cứu đúng trường hợp này.
    ///
    ///   (2) Hỏi "cách nấu phở bò Hà Nội" thì tài liệu SSRF khớp 2/6 = 0.33,
    ///       vì nó tình cờ chứa "cách" và "nội" (trong "nội bộ"). Đủ để vượt
    ///       ngưỡng, và agent trả lời chuyện SSRF cho câu hỏi về phở.
    ///       topicCoverage = 0 dìm nó xuống dưới ngưỡng.
    ///
    /// Và VÌ SAO KHÔNG clamp bằng Math.Min(1.0, ...)? Vì clamp làm bão hoà:
    /// "Softmax" (body 1.0 + topic 1.0) và "Attention" (body 1.0, topic 0 —
    /// chỉ NHẮC tới softmax) đều ra 1.0, rồi thứ tự do id quyết định, nên tài
    /// liệu sai thắng. Chia cho 2.5 giữ điểm trong 0..1 mà không mất thứ tự.
    /// </summary>
    public IReadOnlyList<ScoredKnowledge> Search(string query, int take = 3)
    {
        ArgumentNullException.ThrowIfNull(query);

        var words = Keywords.Extract(query);
        if (words.Count == 0) return [];

        return [.. _entries
            .Select(e => new ScoredKnowledge(e, Score(words, e)))
            .Where(x => x.Score >= MinimumRelevance)
            .OrderByDescending(x => x.Score)
            .ThenBy(x => x.Entry.Id, StringComparer.Ordinal)
            .Take(take)];
    }

    private static double Score(IReadOnlySet<string> query, KnowledgeEntry entry)
    {
        double body = Keywords.Relevance(query, entry.AllKeywords);
        double topic = Keywords.Relevance(query, entry.TopicKeywords);

        // Chiều ngược lại: chủ đề của tài liệu được câu hỏi phủ bao nhiêu.
        double topicCoverage = Keywords.Relevance(entry.TopicKeywords, query);

        return (body + 0.5 * topic + 1.0 * topicCoverage) / 2.5;
    }

    public bool Remove(string id) => _entries.RemoveAll(e => e.Id == id) > 0;

    public void Clear() => _entries.Clear();
}
