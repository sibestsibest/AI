using System.Net;
using System.Net.Http.Headers;
using System.Text;
using Ai.Providers.Contracts;
using Ai.Providers.Security;

namespace Ai.Providers.Http;

/// <summary>Kết quả một lời gọi HTTP, đã chuẩn hoá về lỗi của lớp provider.</summary>
public sealed record HttpOutcome(bool Ok, string Body, AiError? Error)
{
    public static HttpOutcome Success(string body) => new(true, body, null);

    public static HttpOutcome Failure(AiError error) => new(false, "", error);
}

/// <summary>
/// TẦNG HTTP DÙNG CHUNG cho mọi provider — timeout, chặn kích thước, dịch lỗi.
///
/// VỀ VÒNG ĐỜI HttpClient: dùng đúng cách của <c>HttpWebFetcher</c> (Phase 12)
/// — một <see cref="SocketsHttpHandler"/> dùng chung với
/// <c>PooledConnectionLifetime</c>, thay vì <c>IHttpClientFactory</c> (nằm
/// trong package Microsoft.Extensions.Http, mà dự án này không có package nào
/// ngoài bộ test). Cách này giải đúng hai vấn đề mà factory giải: không cạn
/// socket, và vẫn nhận DNS mới.
///
/// <c>AllowAutoRedirect = false</c>, và ở đây nó là một quyết định BẢO MẬT chứ
/// không phải sở thích: HttpClient khi tự đi theo chuyển hướng sẽ gửi lại
/// header <c>Authorization</c> cho địa chỉ mới. Một endpoint bị đổi chủ (hoặc
/// một cấu hình sai) trả về 302 sang máy khác là đủ để khoá API của người dùng
/// bị gửi tới đó. API không cần chuyển hướng, nên tắt hẳn.
/// </summary>
public static class ProviderHttp
{
    private static readonly Lazy<HttpClient> SharedClient = new(CreateSharedClient);

    private static HttpClient CreateSharedClient()
    {
        var handler = new SocketsHttpHandler
        {
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
            PooledConnectionIdleTimeout = TimeSpan.FromMinutes(2),
            AllowAutoRedirect = false,
            ConnectTimeout = TimeSpan.FromSeconds(5),
            AutomaticDecompression = DecompressionMethods.All,
        };

        return new HttpClient(handler)
        {
            // Hạn thời gian do từng lời gọi tự đặt bằng CancellationToken, nên
            // ở đây phải là vô hạn — nếu không thì hai hạn thời gian chồng nhau
            // và cái nào bắn trước thì thông điệp lỗi nói sai nguyên nhân.
            Timeout = Timeout.InfiniteTimeSpan,
        };
    }

    public static HttpClient Shared => SharedClient.Value;

    /// <summary>
    /// Gửi JSON, nhận chuỗi, có hạn thời gian và hạn kích thước.
    ///
    /// <paramref name="maxCharacters"/> chặn ở tầng ĐỌC, không phải sau khi đã
    /// đọc xong: một provider hỏng (hoặc bị chiếm) trả về luồng vô hạn sẽ làm
    /// hết bộ nhớ trước khi có ai kịp kiểm tra độ dài.
    /// </summary>
    public static async Task<HttpOutcome> PostJsonAsync(
        HttpClient http,
        Uri uri,
        string json,
        string? bearerToken,
        TimeSpan timeout,
        int maxCharacters,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? extraHeaders = null)
    {
        ArgumentNullException.ThrowIfNull(http);
        ArgumentNullException.ThrowIfNull(uri);

        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutSource.CancelAfter(timeout);

        using var request = new HttpRequestMessage(HttpMethod.Post, uri)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json"),
        };

        if (!string.IsNullOrWhiteSpace(bearerToken))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);
        }

        foreach (var (name, value) in extraHeaders ?? new Dictionary<string, string>())
        {
            request.Headers.TryAddWithoutValidation(name, value);
        }

        try
        {
            using var response = await http
                .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeoutSource.Token)
                .ConfigureAwait(false);

            string body = await ReadBoundedAsync(response, maxCharacters, timeoutSource.Token)
                .ConfigureAwait(false);

            return response.IsSuccessStatusCode
                ? HttpOutcome.Success(body)
                : HttpOutcome.Failure(FromStatus(response, body));
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return HttpOutcome.Failure(new AiError(AiErrorKind.Timeout,
                $"hết thời gian sau {timeout.TotalSeconds:F0}s"));
        }
        catch (HttpRequestException ex)
        {
            // Thông điệp của HttpRequestException có thể chứa URL đầy đủ, và URL
            // của một số provider mang khoá trong query string — nên lọc trước.
            return HttpOutcome.Failure(new AiError(AiErrorKind.Network,
                OutboundRedactor.Redact(ex.Message)));
        }
    }

    /// <summary>GET đơn giản, dùng cho việc kiểm tra provider có sẵn sàng.</summary>
    public static async Task<HttpOutcome> GetAsync(
        HttpClient http,
        Uri uri,
        string? bearerToken,
        TimeSpan timeout,
        int maxCharacters,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(http);
        ArgumentNullException.ThrowIfNull(uri);

        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutSource.CancelAfter(timeout);

        using var request = new HttpRequestMessage(HttpMethod.Get, uri);

        if (!string.IsNullOrWhiteSpace(bearerToken))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);
        }

        try
        {
            using var response = await http
                .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeoutSource.Token)
                .ConfigureAwait(false);

            string body = await ReadBoundedAsync(response, maxCharacters, timeoutSource.Token)
                .ConfigureAwait(false);

            return response.IsSuccessStatusCode
                ? HttpOutcome.Success(body)
                : HttpOutcome.Failure(FromStatus(response, body));
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return HttpOutcome.Failure(new AiError(AiErrorKind.Timeout,
                $"hết thời gian sau {timeout.TotalSeconds:F0}s"));
        }
        catch (HttpRequestException ex)
        {
            return HttpOutcome.Failure(new AiError(AiErrorKind.Network, OutboundRedactor.Redact(ex.Message)));
        }
    }

    /// <summary>Đọc nội dung nhưng dừng tại hạn mức — không đọc hết rồi mới cắt.</summary>
    private static async Task<string> ReadBoundedAsync(HttpResponseMessage response, int maxCharacters,
        CancellationToken cancellationToken)
    {
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var reader = new StreamReader(stream, Encoding.UTF8);

        char[] buffer = new char[Math.Min(8192, Math.Max(1, maxCharacters))];
        var builder = new StringBuilder();

        while (builder.Length < maxCharacters)
        {
            int read = await reader
                .ReadAsync(buffer.AsMemory(0, Math.Min(buffer.Length, maxCharacters - builder.Length)), cancellationToken)
                .ConfigureAwait(false);

            if (read == 0) break;

            builder.Append(buffer, 0, read);
        }

        return builder.ToString();
    }

    /// <summary>
    /// Dịch mã HTTP thành loại lỗi của lớp provider.
    ///
    /// Phân loại theo "có nên thử lại" chứ không theo mã số, vì đó là điều duy
    /// nhất bộ định tuyến cần biết. Thân phản hồi được CẮT NGẮN và lọc bí mật
    /// trước khi đưa vào thông điệp: nhiều provider trả lại chính request
    /// (kèm header) trong phần lỗi.
    /// </summary>
    private static AiError FromStatus(HttpResponseMessage response, string body)
    {
        string detail = OutboundRedactor.Redact(body.Length > 300 ? body[..300] + "…" : body);

        var kind = response.StatusCode switch
        {
            HttpStatusCode.TooManyRequests => AiErrorKind.RateLimited,
            HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden => AiErrorKind.Unauthorized,
            HttpStatusCode.NotFound => AiErrorKind.ModelUnavailable,
            >= HttpStatusCode.InternalServerError => AiErrorKind.ProviderError,
            _ => AiErrorKind.ProviderError,
        };

        return new AiError(kind, $"HTTP {(int)response.StatusCode} {response.StatusCode}: {detail}")
        {
            RetryAfter = response.Headers.RetryAfter?.Delta
                         ?? (response.Headers.RetryAfter?.Date is { } date
                             ? date - DateTimeOffset.UtcNow
                             : null),
        };
    }
}
