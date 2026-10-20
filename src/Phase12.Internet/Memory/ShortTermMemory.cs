namespace Ai.Phase12.Memory;

/// <summary>
/// BỘ NHỚ NGẮN HẠN — cửa sổ những gì vừa xảy ra.
///
/// Dung lượng CÓ GIỚI HẠN và cố định. Thêm mẩu thứ (N+1) thì mẩu cũ nhất
/// bị đẩy ra. Đây chính là "context window" của một LLM: 8K token, 128K
/// token — dù lớn đến đâu vẫn là hữu hạn, và cái gì trôi ra khỏi cửa sổ
/// là biến mất.
///
/// Mẩu bị đẩy ra KHÔNG nhất thiết mất hẳn: nếu đủ quan trọng, nó được
/// CHUYỂN sang bộ nhớ dài hạn. Đây là mô hình củng cố ký ức của con người
/// — phần lớn những gì ta trải qua trong ngày bị bỏ đi, chỉ phần đáng nhớ
/// mới được ghi lại lâu dài.
/// </summary>
public sealed class ShortTermMemory(int capacity = 5)
{
    private readonly LinkedList<MemoryFact> _items = new();

    public int Capacity { get; } = capacity > 0
        ? capacity
        : throw new ArgumentOutOfRangeException(nameof(capacity));

    public int Count => _items.Count;

    /// <summary>Các mẩu theo thứ tự mới nhất trước.</summary>
    public IReadOnlyList<MemoryFact> Items => [.. _items];

    public bool IsFull => _items.Count >= Capacity;

    /// <summary>Thêm vào đầu; trả về mẩu bị đẩy ra (nếu có).</summary>
    public MemoryFact? Add(MemoryFact fact)
    {
        ArgumentNullException.ThrowIfNull(fact);

        _items.AddFirst(fact);

        if (_items.Count <= Capacity) return null;

        var evicted = _items.Last!.Value;
        _items.RemoveLast();
        return evicted;
    }

    public bool Contains(string id) => _items.Any(f => f.Id == id);

    public bool Remove(string id)
    {
        var node = _items.FirstOrDefault(f => f.Id == id);
        return node is not null && _items.Remove(node);
    }

    public void Clear() => _items.Clear();
}
