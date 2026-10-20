namespace Ai.Phase13.Memory;

public enum StoreOutcome
{
    /// <summary>Ký ức mới, đã ghi nhận.</summary>
    Stored,

    /// <summary>Đã biết rồi — không tạo bản sao, chỉ củng cố lại.</summary>
    Reinforced,

    /// <summary>Không đáng lưu.</summary>
    Rejected,
}

public sealed record StoreResult(StoreOutcome Outcome, string Reason, MemoryFact? Fact = null);

/// <summary>
/// MemoryStore — mặt tiền gộp bộ nhớ ngắn hạn và dài hạn, và nơi thực thi
/// nguyên tắc "KHÔNG LƯU MỌI THỨ MỘT CÁCH MÙ QUÁNG".
///
/// Bốn quy tắc lọc, theo thứ tự:
///
///   1. Quá ngắn / không có từ khoá nào    -> bỏ  ("ok", "ừ", "hehe")
///   2. Đã có ký ức y hệt                  -> củng cố, không tạo bản sao
///   3. Importance dưới ngưỡng             -> bỏ  (chuyện phiếm)
///   4. Còn lại                            -> lưu
///
/// Vì sao phải lọc? Bộ nhớ không lọc thì nhanh chóng đầy rác, và rác làm
/// hỏng khâu truy xuất: mọi câu hỏi đều lôi về hàng đống thứ vô nghĩa.
/// Chất lượng bộ nhớ nằm ở chỗ BỎ ĐI cái gì, chứ không phải giữ được bao nhiêu.
/// </summary>
public sealed class MemoryStore
{
    private readonly IClock _clock;
    private readonly Ranker _ranker;
    private int _nextId = 1;

    public ShortTermMemory ShortTerm { get; }

    public LongTermMemory LongTerm { get; } = new();

    /// <summary>Dưới ngưỡng này thì coi là chuyện phiếm, không lưu.</summary>
    public double MinimumImportance { get; init; } = 0.3;

    /// <summary>Mẩu bị đẩy khỏi bộ nhớ ngắn hạn chỉ được chuyển sang dài hạn nếu đủ quan trọng.</summary>
    public double PromotionThreshold { get; init; } = 0.5;

    public MemoryStore(IClock? clock = null, int shortTermCapacity = 5, Ranker? ranker = null)
    {
        _clock = clock ?? new SystemClock();
        _ranker = ranker ?? new Ranker();
        ShortTerm = new ShortTermMemory(shortTermCapacity);
    }

    // ------------------------------------------------------------------
    // Store
    // ------------------------------------------------------------------

    public StoreResult Store(string content, double importance = 0.5, string source = "user")
    {
        ArgumentNullException.ThrowIfNull(content);

        var keywords = Keywords.Extract(content);

        // Quy tắc 1 — không có nội dung thực chất.
        if (keywords.Count == 0)
        {
            return new StoreResult(StoreOutcome.Rejected, "không có từ khoá nào đáng lưu");
        }

        // Quy tắc 2 — đã biết rồi.
        var duplicate = LongTerm.FindDuplicate(content);
        if (duplicate is not null)
        {
            duplicate.Touch(_clock.Now);
            duplicate.Importance = Math.Min(1.0, Math.Max(duplicate.Importance, importance) + 0.05);

            return new StoreResult(StoreOutcome.Reinforced,
                $"đã biết rồi — củng cố (importance {duplicate.Importance:F2}, " +
                $"đã nhắc {duplicate.AccessCount} lần)", duplicate);
        }

        // Quy tắc 3 — không đủ quan trọng.
        if (importance < MinimumImportance)
        {
            return new StoreResult(StoreOutcome.Rejected,
                $"importance {importance:F2} dưới ngưỡng {MinimumImportance:F2}");
        }

        // Quy tắc 4 — lưu.
        var fact = new MemoryFact
        {
            Id = $"m{_nextId++}",
            Content = content.Trim(),
            Keywords = keywords,
            Importance = importance,
            CreatedAt = _clock.Now,
            LastAccessedAt = _clock.Now,
            Source = source,
        };

        LongTerm.Add(fact);

        var evicted = ShortTerm.Add(fact);
        if (evicted is not null && _ranker.RetentionScore(evicted, _clock.Now) < PromotionThreshold)
        {
            // Trôi khỏi cửa sổ ngắn hạn mà cũng không đủ giá trị -> quên luôn.
            LongTerm.Remove(evicted.Id);
        }

        return new StoreResult(StoreOutcome.Stored, "ký ức mới", fact);
    }

    // ------------------------------------------------------------------
    // Retrieve / Search / Rank
    // ------------------------------------------------------------------

    /// <summary>Lấy ra ký ức khớp nhất, và đánh dấu là vừa dùng.</summary>
    public MemoryFact? Retrieve(string query)
    {
        var best = Search(query, take: 1).FirstOrDefault();
        if (best is null) return null;

        best.Fact.Touch(_clock.Now);
        return best.Fact;
    }

    /// <summary>
    /// Tìm các ký ức liên quan, xếp theo điểm giảm dần.
    /// KHÔNG đánh dấu là đã dùng — tìm kiếm chỉ là nhìn, chưa phải nhớ ra.
    /// </summary>
    public IReadOnlyList<ScoredMemory> Search(string query, int take = 3, double minimumScore = 0.01)
    {
        ArgumentNullException.ThrowIfNull(query);

        return [.. Rank(query).Where(s => s.Score >= minimumScore).Take(take)];
    }

    /// <summary>Chấm điểm TOÀN BỘ bộ nhớ dài hạn theo một câu hỏi, xếp giảm dần.</summary>
    public IReadOnlyList<ScoredMemory> Rank(string query)
    {
        ArgumentNullException.ThrowIfNull(query);

        var keywords = Keywords.Extract(query);
        var now = _clock.Now;

        return [.. LongTerm.Facts
            .Select(f => _ranker.Score(f, keywords, now))
            .OrderByDescending(s => s.Score)
            .ThenBy(s => s.Fact.Id, StringComparer.Ordinal)];
    }

    // ------------------------------------------------------------------
    // Forget
    // ------------------------------------------------------------------

    /// <summary>Quên có chủ đích một ký ức cụ thể.</summary>
    public bool Forget(string id)
    {
        ShortTerm.Remove(id);
        return LongTerm.Remove(id);
    }

    /// <summary>
    /// Dọn bộ nhớ: bỏ những ký ức có điểm giữ lại dưới ngưỡng.
    ///
    /// Đây là điểm khác biệt với một cơ sở dữ liệu. Database giữ mọi thứ mãi
    /// mãi cho đến khi có ai đó ra lệnh xoá. Bộ nhớ thì TỰ phai đi, và phai
    /// đúng những thứ ít giá trị nhất — nhờ vậy phần còn lại vẫn dùng được.
    /// </summary>
    public IReadOnlyList<MemoryFact> ForgetWeak(double threshold = 0.25)
    {
        var now = _clock.Now;

        var doomed = LongTerm.Facts
            .Where(f => _ranker.RetentionScore(f, now) < threshold)
            .ToList();

        foreach (var fact in doomed)
        {
            Forget(fact.Id);
        }

        return doomed;
    }

    /// <summary>Điểm giữ lại của một ký ức — để quan sát và giải thích.</summary>
    public double RetentionScore(MemoryFact fact) => _ranker.RetentionScore(fact, _clock.Now);

    public void Clear()
    {
        ShortTerm.Clear();
        LongTerm.Clear();
    }
}
