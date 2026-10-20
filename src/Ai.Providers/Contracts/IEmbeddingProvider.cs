namespace Ai.Providers.Contracts;

/// <summary>
/// MỘT VECTOR kèm đủ thứ cần để tìm lại và để giải thích nó ở đâu ra.
///
/// Đề bài đòi lưu: vector, document id, chunk id, metadata, source, thời điểm.
/// Tất cả đều có ở đây, và không phải để cho đủ — thiếu bất kỳ cái nào thì một
/// kết quả tra cứu sẽ không nói được nó lấy từ tài liệu nào, đoạn nào, lúc nào.
///
/// <see cref="Model"/> cũng được lưu, và đây là trường hay bị quên nhất: vector
/// của hai model khác nhau KHÔNG so sánh được với nhau. Đổi model nhúng mà
/// không đánh dấu thì kho vector cũ và mới trộn lẫn, và mọi khoảng cách tính ra
/// đều vô nghĩa — nhưng vẫn ra một con số, nên không có gì báo lỗi.
/// </summary>
public sealed record EmbeddingVector
{
    public required string DocumentId { get; init; }

    public required string ChunkId { get; init; }

    public required IReadOnlyList<float> Values { get; init; }

    public required string Model { get; init; }

    /// <summary>Nguồn gốc: tên file, URL, hay tên kho nội bộ.</summary>
    public string Source { get; init; } = "";

    public IReadOnlyDictionary<string, string> Metadata { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);

    public required DateTimeOffset CreatedAt { get; init; }

    public int Dimensions => Values.Count;

    public override string ToString() =>
        $"{DocumentId}#{ChunkId} — {Dimensions} chiều ({Model})";
}

/// <summary>Kết quả một lượt nhúng: hoặc có vector, hoặc có lý do không có.</summary>
public sealed record EmbeddingResult
{
    public required bool Success { get; init; }

    public IReadOnlyList<EmbeddingVector> Vectors { get; init; } = [];

    public required string Provider { get; init; }

    public string? Model { get; init; }

    public AiUsage Usage { get; init; } = AiUsage.Unknown;

    public TimeSpan Latency { get; init; }

    public AiError? Error { get; init; }

    public static EmbeddingResult Ok(string provider, string model,
        IReadOnlyList<EmbeddingVector> vectors, TimeSpan latency, AiUsage? usage = null) =>
        new()
        {
            Success = true,
            Provider = provider,
            Model = model,
            Vectors = vectors,
            Latency = latency,
            Usage = usage ?? AiUsage.Unknown,
        };

    public static EmbeddingResult Fail(string provider, string? model, AiError error) =>
        new() { Success = false, Provider = provider, Model = model, Error = error };
}

/// <summary>Một mẩu văn bản chờ nhúng, kèm danh tính của nó.</summary>
public sealed record EmbeddingInput(string DocumentId, string ChunkId, string Text)
{
    public string Source { get; init; } = "";

    public IReadOnlyDictionary<string, string> Metadata { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);
}

/// <summary>
/// NHÀ CUNG CẤP VECTOR NHÚNG — tách hẳn khỏi <see cref="IAiProvider"/>.
///
/// VÌ SAO KHÔNG GỘP VÀO <c>IAiProvider</c>? Đúng lý lẽ đã ghi ở đó: mỗi thành
/// viên thêm vào là một thứ MỌI provider phải hiện thực. Ollama và một endpoint
/// chat thuần sẽ phải ném <c>NotSupportedException</c>, và chỗ gọi lại phải bọc
/// try/catch để biết ai làm được gì.
///
/// Giao diện riêng thì một lớp có thể hiện thực CẢ HAI khi nó làm được cả hai,
/// và chỉ hiện thực một khi nó chỉ làm được một. Năng lực trở thành chuyện của
/// kiểu, không phải chuyện của ngoại lệ lúc chạy.
///
/// LUẬT DÙNG: KHÔNG nhúng mọi lượt hội thoại. Chỉ nhúng thứ có giá trị tra cứu
/// lâu dài. Nhúng tự động mọi tin nhắn là cách biến một hoá đơn nhỏ thành một
/// hoá đơn lớn mà không ai nhận ra, và kho vector đầy rác thì tra cứu cũng tệ đi.
/// </summary>
public interface IEmbeddingProvider
{
    string ProviderName { get; }

    /// <summary>Số chiều của vector model này sinh ra. 0 = chưa biết cho tới lần gọi đầu.</summary>
    int Dimensions { get; }

    Task<EmbeddingResult> EmbedAsync(IReadOnlyList<EmbeddingInput> inputs,
        CancellationToken cancellationToken = default);
}
