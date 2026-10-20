namespace Ai.Phase15.Internet;

/// <summary>
/// Một câu truy vấn gửi tới máy tìm kiếm.
///
/// Tách thành kiểu riêng thay vì truyền string, vì truy vấn còn mang theo
/// giới hạn số kết quả và ngôn ngữ — và vì nó là KHOÁ CACHE.
/// </summary>
public sealed record SearchQuery(string Text, int MaxResults = 5)
{
    public string CacheKey => $"{Text.Trim().ToLowerInvariant()}|{MaxResults}";

    public override string ToString() => Text;
}

/// <summary>
/// Một dòng kết quả do máy tìm kiếm trả về — CHƯA tải trang, chỉ có tiêu đề,
/// đường dẫn và đoạn trích.
///
/// Tiêu đề và đoạn trích đến từ Internet nên là <see cref="UntrustedText"/>.
/// Url là <see cref="Uri"/> đã phân tích được, không phải string — sai định
/// dạng thì bị loại ngay ở biên, không lọt vào sâu trong hệ thống.
/// </summary>
public sealed record SearchResult
{
    public required Uri Url { get; init; }

    public required UntrustedText Title { get; init; }

    public required UntrustedText Snippet { get; init; }

    /// <summary>Máy tìm kiếm nào trả về dòng này — để truy nguồn và để chấm điểm.</summary>
    public string Provider { get; init; } = "unknown";

    /// <summary>Thứ tự do chính máy tìm kiếm xếp, bắt đầu từ 1.</summary>
    public int Rank { get; init; } = 1;

    public string Host => Url.Host;

    public static SearchResult Create(string title, string url, string snippet, string provider = "unknown", int rank = 1) =>
        new()
        {
            Url = new Uri(url, UriKind.Absolute),
            Title = UntrustedText.FromWeb(title),
            Snippet = UntrustedText.FromWeb(snippet),
            Provider = provider,
            Rank = rank,
        };

    public override string ToString() => $"[{Provider}#{Rank}] {Title.Preview(60)} — {Url}";
}

/// <summary>
/// Kết quả đã được chấm điểm và xếp hạng lại bởi chính agent
/// (<see cref="SourceRanker"/>), kèm lời giải thích vì sao.
///
/// Giữ lời giải thích không phải để cho đẹp: Phase 14 sẽ phải trả lời câu
/// "vì sao bạn tin nguồn này", và câu trả lời đó phải dựng được từ dữ liệu
/// thật chứ không phải viết lại bằng văn.
/// </summary>
public sealed record RankedResult(SearchResult Result, double Score, string Explanation)
{
    public Uri Url => Result.Url;

    public override string ToString() => $"{Score:F3}  {Result.Url}  ({Explanation})";
}
