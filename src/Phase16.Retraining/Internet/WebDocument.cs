namespace Ai.Phase16.Internet;

/// <summary>
/// Một trang web đã tải về và đã bóc lấy phần văn bản.
///
/// Khác <see cref="SearchResult"/> ở chỗ: search result chỉ là lời máy tìm
/// kiếm NÓI về trang đó; WebDocument là nội dung THẬT của trang.
///
/// <see cref="RetrievedAt"/> quan trọng hơn vẻ ngoài của nó. Nội dung Internet
/// có tuổi: câu trả lời đúng hôm nay có thể sai tháng sau. Phase 12 sẽ dùng
/// đúng trường này để tính độ tin cậy của evidence theo thời gian.
/// </summary>
public sealed record WebDocument
{
    public required Uri Url { get; init; }

    public required UntrustedText Title { get; init; }

    /// <summary>Văn bản thuần đã bóc khỏi HTML. Không tin cậy, luôn là DATA.</summary>
    public required UntrustedText Text { get; init; }

    public required DateTimeOffset RetrievedAt { get; init; }

    public string? ContentType { get; init; }

    /// <summary>Số byte thực sự đọc từ mạng.</summary>
    public long ByteCount { get; init; }

    /// <summary>Đã chạm giới hạn kích thước và bị cắt giữa đường.</summary>
    public bool Truncated { get; init; }

    public string Host => Url.Host;

    public bool HasText => !Text.IsEmpty;

    public override string ToString() =>
        $"{Url} — \"{Title.Preview(60)}\" ({Text.Length} ký tự, tải lúc {RetrievedAt:HH:mm:ss})";
}

/// <summary>Kết quả thô của một lượt tải trang — trước khi bóc nội dung.</summary>
public sealed record FetchResult
{
    public required bool Success { get; init; }

    public required Uri Url { get; init; }

    /// <summary>Thân phản hồi. Không tin cậy ngay cả khi <see cref="Success"/> là true.</summary>
    public UntrustedText Body { get; init; } = UntrustedText.Empty;

    public int StatusCode { get; init; }

    public string? ContentType { get; init; }

    public long ByteCount { get; init; }

    public bool Truncated { get; init; }

    /// <summary>Vì sao thất bại — để agent nói thật thay vì im lặng bỏ qua.</summary>
    public string? Error { get; init; }

    public static FetchResult Fail(Uri url, string error, int statusCode = 0) =>
        new() { Success = false, Url = url, Error = error, StatusCode = statusCode };
}
