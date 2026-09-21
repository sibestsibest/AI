using System.Globalization;
using Ai.Phase16.Memory;
using Ai.Providers.Contracts;

namespace Ai.Providers.Resilience;

/// <summary>Sức khoẻ của một provider, đo được và đọc được.</summary>
public sealed record ProviderHealth
{
    public required string Provider { get; init; }

    public int Successes { get; init; }

    public int Failures { get; init; }

    public int ConsecutiveFailures { get; init; }

    public int Timeouts { get; init; }

    public int RateLimits { get; init; }

    /// <summary>Tạm bị loại tới thời điểm này. Null = đang dùng được.</summary>
    public DateTimeOffset? UnavailableUntil { get; init; }

    public TimeSpan AverageLatency { get; init; }

    public AiError? LastError { get; init; }

    public int Total => Successes + Failures;

    public double SuccessRate => Total == 0 ? 1.0 : (double)Successes / Total;

    public bool IsCoolingDown(DateTimeOffset now) => UnavailableUntil is { } until && now < until;

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "{0}: {1}/{2} thành công ({3:P0}), trễ tb {4:F0} ms{5}",
            Provider, Successes, Total, SuccessRate, AverageLatency.TotalMilliseconds,
            UnavailableUntil is null ? "" : $", tạm loại tới {UnavailableUntil:HH:mm:ss}");
}

/// <summary>
/// THEO DÕI SỨC KHOẺ PROVIDER — và tạm loại provider đang hỏng.
///
/// VÌ SAO CẦN, KHI ĐÃ CÓ FALLBACK?
///
/// Vì fallback trả giá bằng THỜI GIAN. Một provider đang chết vẫn được thử đầu
/// tiên mỗi lượt, và mỗi lượt lại chờ hết hạn thời gian (ví dụ 30 giây) trước
/// khi chuyển sang provider sau. Người dùng thấy mọi câu trả lời chậm đi 30
/// giây, trong khi hệ thống vẫn "hoạt động bình thường" theo mọi số liệu.
///
/// Nên sau <see cref="FailureThreshold"/> lần hỏng LIÊN TIẾP, provider bị tạm
/// loại trong một khoảng nghỉ. Khoảng nghỉ TĂNG DẦN theo số lần hỏng, nhưng có
/// TRẦN (<see cref="MaxCooldown"/>) — không có trần thì một provider hỏng vài
/// giờ sẽ bị loại hàng ngày, và nó không bao giờ được thử lại để phát hiện là
/// đã sống.
///
/// Hỏng vì HẾT HẠN MỨC (429) thì tôn trọng đúng thời gian provider yêu cầu
/// (<c>Retry-After</c>) nếu có: đó là con số họ nói, không phải con số ta đoán.
///
/// KHÔNG có vòng thử lại vô hạn ở đây. Lớp này chỉ trả lời một câu: "có nên
/// thử provider này lúc này không".
/// </summary>
public sealed class ProviderHealthTracker
{
    private readonly Dictionary<string, ProviderHealth> _health = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<double>> _latencies = new(StringComparer.OrdinalIgnoreCase);
    private readonly IClock _clock;
    private readonly Lock _gate = new();

    public ProviderHealthTracker(IClock? clock = null) => _clock = clock ?? new SystemClock();

    /// <summary>Số lần hỏng liên tiếp trước khi tạm loại. 3 là đủ để bỏ qua nhiễu mạng lẻ.</summary>
    public int FailureThreshold { get; init; } = 3;

    public TimeSpan BaseCooldown { get; init; } = TimeSpan.FromSeconds(30);

    public TimeSpan MaxCooldown { get; init; } = TimeSpan.FromMinutes(10);

    public IReadOnlyCollection<ProviderHealth> All
    {
        get
        {
            lock (_gate) return [.. _health.Values];
        }
    }

    public ProviderHealth For(string provider)
    {
        ArgumentNullException.ThrowIfNull(provider);

        lock (_gate)
        {
            return _health.TryGetValue(provider, out var health)
                ? health
                : new ProviderHealth { Provider = provider };
        }
    }

    /// <summary>Provider này có đáng thử lúc này không.</summary>
    public bool ShouldTry(string provider) => !For(provider).IsCoolingDown(_clock.Now);

    public void RecordSuccess(string provider, TimeSpan latency)
    {
        ArgumentNullException.ThrowIfNull(provider);

        lock (_gate)
        {
            var health = For(provider);

            _health[provider] = health with
            {
                Successes = health.Successes + 1,
                ConsecutiveFailures = 0,
                // Thành công thì XOÁ hẳn khoảng nghỉ: provider đã sống lại,
                // giữ lại hạn nghỉ cũ chỉ làm nó bị loại oan ở lượt sau.
                UnavailableUntil = null,
                AverageLatency = Track(provider, latency),
                LastError = null,
            };
        }
    }

    public void RecordFailure(string provider, AiError error, TimeSpan latency = default)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentNullException.ThrowIfNull(error);

        lock (_gate)
        {
            var health = For(provider);
            int consecutive = health.ConsecutiveFailures + 1;

            _health[provider] = health with
            {
                Failures = health.Failures + 1,
                ConsecutiveFailures = consecutive,
                Timeouts = health.Timeouts + (error.Kind == AiErrorKind.Timeout ? 1 : 0),
                RateLimits = health.RateLimits + (error.Kind == AiErrorKind.RateLimited ? 1 : 0),
                AverageLatency = latency > TimeSpan.Zero ? Track(provider, latency) : health.AverageLatency,
                LastError = error,
                UnavailableUntil = CooldownUntil(error, consecutive),
            };
        }
    }

    /// <summary>
    /// Tính thời điểm được thử lại.
    ///
    /// Ba loại lỗi được đối xử khác nhau, và khác biệt đó là điểm chính:
    ///
    ///   Unauthorized     -> loại NGAY, nghỉ dài nhất. Khoá sai thì thử lại
    ///                       không bao giờ thành công; chỉ có người sửa cấu
    ///                       hình mới cứu được, nên đừng đốt thời gian vào nó.
    ///   RateLimited      -> nghỉ đúng thời gian provider yêu cầu
    ///   còn lại          -> nghỉ tăng dần, chỉ khi đã hỏng đủ nhiều lần
    /// </summary>
    private DateTimeOffset? CooldownUntil(AiError error, int consecutiveFailures)
    {
        var now = _clock.Now;

        if (error.Kind == AiErrorKind.Unauthorized) return now + MaxCooldown;

        if (error.Kind == AiErrorKind.RateLimited)
        {
            var wait = error.RetryAfter ?? BaseCooldown;
            return now + (wait > MaxCooldown ? MaxCooldown : wait);
        }

        if (consecutiveFailures < FailureThreshold) return null;

        // Tăng gấp đôi theo số lần hỏng vượt ngưỡng, có trần.
        int steps = Math.Min(consecutiveFailures - FailureThreshold, 6);
        var cooldown = BaseCooldown * Math.Pow(2, steps);

        return now + (cooldown > MaxCooldown ? MaxCooldown : cooldown);
    }

    /// <summary>Trung bình trễ trên 20 lượt gần nhất — đủ để thấy xu hướng, không phình bộ nhớ.</summary>
    private TimeSpan Track(string provider, TimeSpan latency)
    {
        if (!_latencies.TryGetValue(provider, out var samples))
        {
            samples = [];
            _latencies[provider] = samples;
        }

        samples.Add(latency.TotalMilliseconds);

        if (samples.Count > 20) samples.RemoveAt(0);

        return TimeSpan.FromMilliseconds(samples.Average());
    }

    /// <summary>Bỏ mọi khoảng nghỉ — dùng khi người dùng sửa cấu hình rồi muốn thử lại ngay.</summary>
    public void ResetCooldowns()
    {
        lock (_gate)
        {
            foreach (string provider in _health.Keys.ToList())
            {
                _health[provider] = _health[provider] with { UnavailableUntil = null, ConsecutiveFailures = 0 };
            }
        }
    }
}
