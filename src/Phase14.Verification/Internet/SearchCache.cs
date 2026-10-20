using Ai.Phase14.Memory;

namespace Ai.Phase14.Internet;

/// <summary>Cache kết quả tìm kiếm theo truy vấn.</summary>
public interface ISearchCache
{
    bool TryGet(string key, out IReadOnlyList<SearchResult> results);

    void Set(string key, IReadOnlyList<SearchResult> results);

    void Clear();

    int Count { get; }
}

/// <summary>
/// Cache trong bộ nhớ, có hạn dùng (TTL) và có chặn trên số mục.
///
/// VÌ SAO CẦN CACHE? Ba lý do, theo thứ tự quan trọng:
///
///   1. Lịch sự với máy tìm kiếm — hỏi lại đúng một truy vấn 10 lần trong một
///      phút là hành vi đáng bị chặn IP.
///   2. Tất định trong một lượt — cùng một câu hỏi hỏi lại phải ra cùng kết
///      quả, nếu không thì gỡ lỗi là việc bất khả thi.
///   3. Nhanh.
///
/// VÌ SAO PHẢI CÓ TTL? Vì cache nội dung Internet mà không có hạn dùng thì
/// nó biến thành "kiến thức" — đúng cái điều mà Phase 12 tuyệt đối không được
/// làm. Kết quả tìm kiếm là ảnh chụp tại một thời điểm, hết hạn là phải bỏ.
///
/// VÌ SAO DÙNG LẠI IClock CỦA PHASE 10? Vì test phải kiểm chứng được cơ chế
/// hết hạn, và cách duy nhất làm được điều đó mà không phải Thread.Sleep là
/// tự tua đồng hồ. Phase 10 đã có sẵn ManualClock cho đúng việc này.
/// </summary>
public sealed class MemorySearchCache : ISearchCache
{
    private readonly Dictionary<string, CacheEntry> _entries = new(StringComparer.OrdinalIgnoreCase);
    private readonly IClock _clock;

    public MemorySearchCache(IClock? clock = null, TimeSpan? timeToLive = null, int maxEntries = 128)
    {
        if (maxEntries <= 0) throw new ArgumentOutOfRangeException(nameof(maxEntries));

        _clock = clock ?? new SystemClock();
        TimeToLive = timeToLive ?? TimeSpan.FromMinutes(10);
        MaxEntries = maxEntries;
    }

    public TimeSpan TimeToLive { get; }

    public int MaxEntries { get; }

    public int Count => _entries.Count;

    public int Hits { get; private set; }

    public int Misses { get; private set; }

    public bool TryGet(string key, out IReadOnlyList<SearchResult> results)
    {
        ArgumentNullException.ThrowIfNull(key);

        if (_entries.TryGetValue(key, out var entry))
        {
            if (_clock.Now - entry.StoredAt < TimeToLive)
            {
                entry.LastUsedAt = _clock.Now;
                entry.UseCount++;
                Hits++;
                results = entry.Results;
                return true;
            }

            // Hết hạn -> bỏ luôn, đừng để rác nằm lại.
            _entries.Remove(key);
        }

        Misses++;
        results = [];
        return false;
    }

    public void Set(string key, IReadOnlyList<SearchResult> results)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(results);

        if (_entries.Count >= MaxEntries && !_entries.ContainsKey(key))
        {
            EvictOldest();
        }

        _entries[key] = new CacheEntry(results, _clock.Now);
    }

    /// <summary>Đầy thì bỏ mục LÂU KHÔNG DÙNG NHẤT (LRU), không phải mục cũ nhất.</summary>
    private void EvictOldest()
    {
        string? victim = null;
        var oldest = DateTimeOffset.MaxValue;

        foreach (var (key, entry) in _entries)
        {
            if (entry.LastUsedAt < oldest)
            {
                oldest = entry.LastUsedAt;
                victim = key;
            }
        }

        if (victim is not null) _entries.Remove(victim);
    }

    /// <summary>Dọn các mục đã hết hạn. Gọi được chủ động, không phải chờ đụng tới mới dọn.</summary>
    public int RemoveExpired()
    {
        var now = _clock.Now;

        var expired = _entries
            .Where(e => now - e.Value.StoredAt >= TimeToLive)
            .Select(e => e.Key)
            .ToList();

        foreach (string key in expired) _entries.Remove(key);

        return expired.Count;
    }

    public void Clear()
    {
        _entries.Clear();
        Hits = 0;
        Misses = 0;
    }

    private sealed class CacheEntry(IReadOnlyList<SearchResult> results, DateTimeOffset storedAt)
    {
        public IReadOnlyList<SearchResult> Results { get; } = results;

        public DateTimeOffset StoredAt { get; } = storedAt;

        public DateTimeOffset LastUsedAt { get; set; } = storedAt;

        public int UseCount { get; set; }
    }
}
