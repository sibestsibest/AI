using Ai.Providers.Contracts;

namespace Ai.Providers.Resilience;

/// <summary>
/// THỬ LẠI CÓ GIỚI HẠN — lùi theo cấp số nhân, có nhiễu, và biết khi nào đừng thử.
///
/// BA LUẬT, và luật thứ ba là luật hay bị bỏ:
///
///   1. CÓ TRẦN. Tối đa <see cref="MaxAttempts"/> lần. Thử lại không giới hạn
///      khi provider đang quá tải chính là cách làm nó quá tải thêm.
///
///   2. LÙI THEO CẤP SỐ NHÂN. Thử lại ngay lập tức gửi lại đúng lượt vừa làm
///      provider sặc, chỉ sớm hơn vài milligiây.
///
///   3. CÓ NHIỄU (jitter). Không có nhiễu thì mọi tiến trình cùng hỏng sẽ cùng
///      thử lại vào đúng một thời điểm — provider vừa gượng dậy đã bị cả đàn
///      đập vào cùng lúc. Nhiễu làm chúng tản ra.
///
/// Và một luật phủ định: lỗi KHÔNG đáng thử lại thì dừng ngay. Khoá sai (401)
/// có thử một trăm lần cũng vẫn sai; model không tồn tại thì càng thử càng lâu.
/// <see cref="AiError.IsRetryable"/> là chỗ duy nhất định nghĩa điều đó.
/// </summary>
public sealed class RetryPolicy
{
    private readonly Random _random;

    public RetryPolicy(int maxAttempts = 3, TimeSpan? baseDelay = null, Random? random = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxAttempts, 1);

        MaxAttempts = Math.Min(maxAttempts, 6);
        BaseDelay = baseDelay ?? TimeSpan.FromMilliseconds(250);

        // Random tiêm vào được để test tất định — nhiễu ngẫu nhiên làm test
        // chạy lâu khác nhau mỗi lần, và không kiểm được khoảng chờ.
        _random = random ?? Random.Shared;
    }

    public int MaxAttempts { get; }

    public TimeSpan BaseDelay { get; }

    public TimeSpan MaxDelay { get; init; } = TimeSpan.FromSeconds(8);

    /// <summary>
    /// Chạy một lời gọi, thử lại khi lỗi đáng thử lại.
    ///
    /// <paramref name="onRetry"/> để bên gọi ghi log từng lần thử — không tự
    /// ghi ở đây, vì lớp này không nên biết hệ thống log ra đâu.
    /// </summary>
    public async Task<AiResponse> ExecuteAsync(
        Func<CancellationToken, Task<AiResponse>> operation,
        CancellationToken cancellationToken,
        Action<int, AiError, TimeSpan>? onRetry = null)
    {
        ArgumentNullException.ThrowIfNull(operation);

        AiResponse response = null!;

        for (int attempt = 1; attempt <= MaxAttempts; attempt++)
        {
            response = await operation(cancellationToken).ConfigureAwait(false);

            if (response.Success) return response;

            var error = response.Error ?? new AiError(AiErrorKind.ProviderError, "không rõ nguyên nhân");

            if (!error.IsRetryable || attempt == MaxAttempts) return response;

            var delay = DelayFor(attempt, error);

            onRetry?.Invoke(attempt, error, delay);

            await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
        }

        return response;
    }

    /// <summary>
    /// Khoảng chờ trước lần thử thứ <paramref name="attempt"/> + 1.
    ///
    /// Provider nói <c>Retry-After</c> thì dùng ĐÚNG con số đó (có trần): họ
    /// biết hạn mức của họ, ta thì đoán.
    /// </summary>
    public TimeSpan DelayFor(int attempt, AiError error)
    {
        ArgumentNullException.ThrowIfNull(error);
        ArgumentOutOfRangeException.ThrowIfLessThan(attempt, 1);

        if (error.RetryAfter is { } requested && requested > TimeSpan.Zero)
        {
            return requested > MaxDelay ? MaxDelay : requested;
        }

        double milliseconds = BaseDelay.TotalMilliseconds * Math.Pow(2, attempt - 1);

        // Nhiễu ±25%: đủ để tản các tiến trình ra, không đủ để làm khoảng chờ
        // trở nên khó đoán khi đọc log.
        double jitter = 1 + ((_random.NextDouble() - 0.5) / 2);
        var delay = TimeSpan.FromMilliseconds(milliseconds * jitter);

        return delay > MaxDelay ? MaxDelay : delay;
    }
}
