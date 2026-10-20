using System.Globalization;
using Ai.Phase16.Memory;
using Ai.Providers.Contracts;

namespace Ai.Providers.Usage;

/// <summary>Một lượt gọi đã ghi lại. Không chứa prompt hay câu trả lời — chỉ siêu dữ liệu.</summary>
public sealed record UsageEntry(
    string Provider,
    string? Model,
    AiTaskType TaskType,
    bool Success,
    TimeSpan Latency,
    AiUsage Usage,
    AiErrorKind ErrorKind,
    DateTimeOffset At)
{
    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture, "[{0:HH:mm:ss}] {1}/{2} {3} {4:F0} ms, {5}{6}",
            At, Provider, Model ?? "?", TaskType, Latency.TotalMilliseconds, Usage,
            Success ? "" : $", LỖI {ErrorKind}");
}

/// <summary>Tổng hợp theo provider+model.</summary>
public sealed record UsageSummary(
    string Provider,
    string? Model,
    int Requests,
    int Failures,
    int? InputTokens,
    int? OutputTokens,
    TimeSpan AverageLatency,
    decimal? EstimatedCost)
{
    public double FailureRate => Requests == 0 ? 0 : (double)Failures / Requests;

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "{0}/{1}: {2} lượt ({3} lỗi), token {4}/{5}, trễ tb {6:F0} ms, chi phí {7}",
            Provider, Model ?? "?", Requests, Failures,
            InputTokens?.ToString(CultureInfo.InvariantCulture) ?? "?",
            OutputTokens?.ToString(CultureInfo.InvariantCulture) ?? "?",
            AverageLatency.TotalMilliseconds,
            EstimatedCost is null ? "không biết" : EstimatedCost.Value.ToString("C4", CultureInfo.InvariantCulture));
}

/// <summary>
/// THỐNG KÊ SỬ DỤNG — và một lời từ chối tường minh về chi phí.
///
/// Lớp này KHÔNG tự biết giá. Nó chỉ tính được chi phí cho những model mà
/// người dùng đã TỰ KHAI giá qua <see cref="SetPrice"/>; còn lại
/// <see cref="UsageSummary.EstimatedCost"/> là null — nghĩa là "không biết",
/// chứ không phải "miễn phí".
///
/// Phân biệt đó là điểm chính của cả lớp. Một hệ thống đoán chi phí bằng cách
/// nhìn tên nhà cung cấp sẽ sai vào đúng lúc tệ nhất: bậc miễn phí đổi điều
/// kiện, model đổi giá, hoặc tài khoản hết hạn dùng thử. Điều duy nhất lớp này
/// chắc chắn là provider nào chạy TẠI MÁY (không tốn tiền mỗi lượt gọi) —
/// và đó là thông tin lấy từ cấu hình, không phải từ phỏng đoán.
///
/// KHÔNG lưu prompt và câu trả lời: thống kê sử dụng sẽ được in ra, được ghi
/// log, có thể được gửi đi — nội dung hội thoại của người dùng không có việc gì
/// ở đó.
/// </summary>
public sealed class AiUsageTracker
{
    private readonly List<UsageEntry> _entries = [];
    private readonly Dictionary<string, (decimal Input, decimal Output)> _prices = new(StringComparer.OrdinalIgnoreCase);
    private readonly IClock _clock;
    private readonly Lock _gate = new();

    public AiUsageTracker(IClock? clock = null) => _clock = clock ?? new SystemClock();

    public int MaxEntries { get; init; } = 500;

    public IReadOnlyList<UsageEntry> Entries
    {
        get
        {
            lock (_gate) return [.. _entries];
        }
    }

    public int TotalRequests => Entries.Count;

    public int TotalFailures => Entries.Count(e => !e.Success);

    /// <summary>
    /// Khai giá cho một model, tính theo MỘT TRIỆU token.
    ///
    /// Người dùng tự khai, vì chỉ họ biết hợp đồng của họ. Không khai thì chi
    /// phí là "không biết" — và đó là câu trả lời trung thực duy nhất.
    /// </summary>
    public void SetPrice(string model, decimal inputPerMillion, decimal outputPerMillion)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);

        lock (_gate) _prices[model] = (inputPerMillion, outputPerMillion);
    }

    public void Record(AiResponse response, AiTaskType taskType)
    {
        ArgumentNullException.ThrowIfNull(response);

        lock (_gate)
        {
            _entries.Add(new UsageEntry(
                response.Provider,
                response.Model,
                taskType,
                response.Success,
                response.Latency,
                response.Usage,
                response.Error?.Kind ?? AiErrorKind.None,
                _clock.Now));

            while (_entries.Count > MaxEntries) _entries.RemoveAt(0);
        }
    }

    public IReadOnlyList<UsageSummary> Summarize()
    {
        lock (_gate)
        {
            return [.. _entries
                .GroupBy(e => (e.Provider, e.Model))
                .Select(group =>
                {
                    var measured = group.Where(e => e.Usage.IsMeasured).ToList();

                    int? input = measured.Count == 0 ? null : measured.Sum(e => e.Usage.InputTokens ?? 0);
                    int? output = measured.Count == 0 ? null : measured.Sum(e => e.Usage.OutputTokens ?? 0);

                    return new UsageSummary(
                        group.Key.Provider,
                        group.Key.Model,
                        group.Count(),
                        group.Count(e => !e.Success),
                        input,
                        output,
                        TimeSpan.FromMilliseconds(group.Average(e => e.Latency.TotalMilliseconds)),
                        EstimateCost(group.Key.Model, input, output));
                })
                .OrderByDescending(s => s.Requests)];
        }
    }

    /// <summary>Chi phí ước lượng — null nếu chưa khai giá hoặc không đo được token.</summary>
    private decimal? EstimateCost(string? model, int? inputTokens, int? outputTokens)
    {
        if (model is null || inputTokens is null || outputTokens is null) return null;
        if (!_prices.TryGetValue(model, out var price)) return null;

        return (inputTokens.Value * price.Input / 1_000_000m)
               + (outputTokens.Value * price.Output / 1_000_000m);
    }

    public void Clear()
    {
        lock (_gate) _entries.Clear();
    }
}
