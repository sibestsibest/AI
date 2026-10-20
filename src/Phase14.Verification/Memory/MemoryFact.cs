namespace Ai.Phase14.Memory;

/// <summary>Đồng hồ trừu tượng — để test kiểm soát được thời gian.</summary>
public interface IClock
{
    DateTimeOffset Now { get; }
}

public sealed class SystemClock : IClock
{
    public DateTimeOffset Now => DateTimeOffset.UtcNow;
}

/// <summary>Đồng hồ tay: test tự tua thời gian để kiểm tra cơ chế quên.</summary>
public sealed class ManualClock(DateTimeOffset start) : IClock
{
    public DateTimeOffset Now { get; private set; } = start;

    public void Advance(TimeSpan amount) => Now += amount;
}

/// <summary>
/// Một mẩu ký ức.
///
/// Không chỉ lưu nội dung — phải lưu cả SIÊU DỮ LIỆU, vì chính siêu dữ liệu
/// mới quyết định ký ức nào đáng nhớ và ký ức nào nên quên:
///
///   Importance   con người đánh giá mức quan trọng (0..1)
///   CreatedAt    để tính độ cũ
///   LastAccessed để biết gần đây có dùng không
///   AccessCount  dùng càng nhiều càng đáng giữ
///   Keywords     để tìm kiếm
///
/// Ba tín hiệu quan trọng / gần đây / hay dùng chính là những gì bộ nhớ con
/// người dùng để chọn lọc. Thứ bạn vừa dùng sáng nay, hoặc dùng mỗi ngày,
/// hoặc gắn với chuyện hệ trọng — thứ đó ở lại.
/// </summary>
public sealed class MemoryFact
{
    public required string Id { get; init; }

    public required string Content { get; init; }

    public required IReadOnlySet<string> Keywords { get; init; }

    public double Importance { get; set; }

    public required DateTimeOffset CreatedAt { get; init; }

    public DateTimeOffset LastAccessedAt { get; set; }

    public int AccessCount { get; set; }

    public string Source { get; init; } = "user";

    /// <summary>Đánh dấu là vừa được dùng — làm ký ức "tươi" trở lại.</summary>
    public void Touch(DateTimeOffset now)
    {
        LastAccessedAt = now;
        AccessCount++;
    }

    public double AgeHours(DateTimeOffset now) => (now - CreatedAt).TotalHours;

    public double HoursSinceLastAccess(DateTimeOffset now) => (now - LastAccessedAt).TotalHours;

    public override string ToString() => $"[{Id}] {Content}";
}
