using System.ClientModel;
using Ai.Providers.Contracts;
using Ai.Providers.Security;

namespace Ai.Providers.OpenAi;

/// <summary>
/// DỊCH LỖI CỦA SDK VỀ <see cref="AiErrorKind"/> — phân theo CÓ NÊN THỬ LẠI.
///
/// Đây là chỗ nối giữa hai cách phân loại lỗi khác nhau: SDK nói bằng mã HTTP,
/// còn <c>RetryPolicy</c> và <c>ProviderHealthTracker</c> cần biết "thử lại có
/// ích không". Hai câu đó không trùng nhau — 401 và 429 đều là 4xx, nhưng thử
/// lại 401 thì sai mấy lần cũng sai.
///
/// VỀ VIỆC THỬ LẠI CHỒNG LÊN NHAU: SDK chính thức đã tự thử lại vài lỗi tạm
/// thời ở tầng HTTP của nó. Nên <c>MaxRetries</c> trong cấu hình provider này
/// nên để THẤP (0–1): đặt 3 ở đây nghĩa là 3 × số lần SDK tự thử, và một lượt
/// 429 có thể thành cả chục lượt gọi thật.
/// </summary>
public static class OpenAiErrors
{
    public static AiError FromStatus(ClientResultException exception)
    {
        ArgumentNullException.ThrowIfNull(exception);

        // Thông điệp của SDK có thể mang URL đầy đủ và tiêu đề request. Lọc
        // trước khi nó đi vào log hay lên màn hình.
        string message = OutboundRedactor.Redact(exception.Message);

        var kind = exception.Status switch
        {
            400 => AiErrorKind.Rejected,
            401 or 403 => AiErrorKind.Unauthorized,
            404 => AiErrorKind.ModelUnavailable,
            408 => AiErrorKind.Timeout,
            409 => AiErrorKind.ProviderError,
            413 => AiErrorKind.Rejected,
            422 => AiErrorKind.Rejected,
            429 => AiErrorKind.RateLimited,
            >= 500 and < 600 => AiErrorKind.ProviderError,

            // Status 0 nghĩa là chưa có phản hồi HTTP nào: hỏng ở tầng mạng.
            0 => AiErrorKind.Network,

            _ => AiErrorKind.ProviderError,
        };

        return new AiError(kind, $"HTTP {exception.Status}: {message}")
        {
            RetryAfter = ReadRetryAfter(exception),
        };
    }

    /// <summary>
    /// Đọc Retry-After nếu nhà cung cấp có nói.
    ///
    /// Tôn trọng con số họ đưa thay vì tự chọn khoảng chờ: khi bị giới hạn tần
    /// suất, chờ ĐÚNG khoảng họ yêu cầu là cách nhanh nhất để được phục vụ lại,
    /// còn thử sớm hơn chỉ làm cửa sổ giới hạn trượt thêm.
    /// </summary>
    private static TimeSpan? ReadRetryAfter(ClientResultException exception)
    {
        var response = exception.GetRawResponse();

        if (response is null) return null;

        if (!response.Headers.TryGetValue("retry-after", out string? value) || value is null)
        {
            return null;
        }

        if (int.TryParse(value, out int seconds) && seconds is > 0 and <= 300)
        {
            return TimeSpan.FromSeconds(seconds);
        }

        return DateTimeOffset.TryParse(value, out var when) && when > DateTimeOffset.UtcNow
            ? when - DateTimeOffset.UtcNow
            : null;
    }
}
