namespace Ai.Phase10.Memory;

/// <summary>
/// BỘ NHỚ DÀI HẠN — kho bền, không giới hạn cứng về dung lượng.
///
/// Khác bộ nhớ ngắn hạn ở chỗ mẩu ký ức KHÔNG bị đẩy ra chỉ vì có cái mới
/// tới. Nó bị quên theo tiêu chí GIÁ TRỊ: không quan trọng, lâu không dùng,
/// ít khi cần đến — thì mới bị loại.
///
/// Đây cũng là chỗ khác biệt với trọng số của model: trọng số chỉ thay đổi
/// khi train lại, còn bộ nhớ dài hạn ghi thêm được tức thì và xoá được có
/// chọn lọc. "User dùng C#" cần nhớ ngay lúc người dùng nói ra, không thể
/// đợi train một epoch.
/// </summary>
public sealed class LongTermMemory
{
    private readonly Dictionary<string, MemoryFact> _facts = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _canonicalToId = new(StringComparer.Ordinal);

    public int Count => _facts.Count;

    public IReadOnlyCollection<MemoryFact> Facts => _facts.Values;

    public void Add(MemoryFact fact)
    {
        ArgumentNullException.ThrowIfNull(fact);

        _facts[fact.Id] = fact;
        _canonicalToId[Keywords.Canonical(fact.Content)] = fact.Id;
    }

    public bool TryGet(string id, out MemoryFact fact) => _facts.TryGetValue(id, out fact!);

    /// <summary>Tìm ký ức có nội dung trùng (sau khi chuẩn hoá) — để không lưu lặp.</summary>
    public MemoryFact? FindDuplicate(string content)
    {
        string canonical = Keywords.Canonical(content);

        return _canonicalToId.TryGetValue(canonical, out string? id) && _facts.TryGetValue(id, out var fact)
            ? fact
            : null;
    }

    public bool Remove(string id)
    {
        if (!_facts.TryGetValue(id, out var fact)) return false;

        _facts.Remove(id);
        _canonicalToId.Remove(Keywords.Canonical(fact.Content));
        return true;
    }

    public void Clear()
    {
        _facts.Clear();
        _canonicalToId.Clear();
    }
}
