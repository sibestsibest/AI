using System.Net;
using System.Text;

namespace Ai.Phase12.Internet;

/// <summary>Tải một trang web về. Không bóc nội dung — đó là việc của <see cref="IContentExtractor"/>.</summary>
public interface IWebFetcher
{
    Task<FetchResult> FetchAsync(Uri url, CancellationToken cancellationToken = default);
}

public sealed record WebFetcherOptions
{
    /// <summary>Hết thời gian cho MỘT lượt tải. Mạng chậm không được làm agent treo.</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Chặn trên số byte đọc từ mạng.
    ///
    /// Không có chặn trên thì một URL trỏ tới file 50 GB sẽ làm agent hết bộ
    /// nhớ — mà chẳng cần lỗ hổng nào, chỉ cần một đường dẫn. Đây là biện pháp
    /// tự bảo vệ, không phải tối ưu hiệu năng.
    /// </summary>
    public long MaxBytes { get; init; } = 2 * 1024 * 1024;

    /// <summary>Kiểu nội dung nhận. Rỗng = nhận tất cả.</summary>
    public IReadOnlySet<string> AllowedContentTypes { get; init; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "text/html", "application/xhtml+xml", "text/plain", "application/json", "text/markdown",
        };

    public string UserAgent { get; init; } = "AI/12.0 (+https://github.com/ai; hoc-tap)";

    /// <summary>
    /// Số lần chuyển hướng tối đa. Phải TỰ đi từng bước thay vì để HttpClient
    /// tự chuyển hướng, vì mỗi URL mới phải được guard soi lại: một trang công
    /// cộng hoàn toàn có thể 302 sang http://169.254.169.254.
    /// </summary>
    public int MaxRedirects { get; init; } = 3;
}

/// <summary>
/// Tải trang bằng HttpClient, có guard, có hạn thời gian, có chặn kích thước.
///
/// VỀ VÒNG ĐỜI HttpClient — và vì sao ở đây KHÔNG dùng IHttpClientFactory:
///
/// Factory ra đời để giải hai vấn đề: (1) `new HttpClient()` mỗi request làm
/// cạn socket, (2) một HttpClient sống mãi thì không bao giờ nhận DNS mới.
/// Cả hai vấn đề đó ở .NET hiện đại giải được trực tiếp bằng một
/// SocketsHttpHandler dùng chung với PooledConnectionLifetime — đúng cách
/// Microsoft khuyến nghị cho ứng dụng không dùng DI container.
///
/// Project này chủ trương không kéo thêm dependency nào (cả 12 phase hiện
/// không có PackageReference nào ngoài bộ test), còn IHttpClientFactory nằm
/// trong Microsoft.Extensions.Http và kéo theo cả DI container. Nên ở đây
/// dùng handler dùng chung. Chỗ tiếp nhận HttpClient vẫn là hàm khởi tạo, nên
/// ứng dụng nào đã có DI thì tiêm client của factory vào là xong — không phải
/// sửa lớp này.
/// </summary>
public sealed class HttpWebFetcher : IWebFetcher, IDisposable
{
    private static readonly Lazy<HttpClient> SharedClient = new(CreateSharedClient);

    private readonly HttpClient _http;
    private readonly bool _ownsClient;
    private readonly IUrlGuard _guard;
    private readonly WebFetcherOptions _options;

    public HttpWebFetcher(HttpClient? http = null, IUrlGuard? guard = null, WebFetcherOptions? options = null)
    {
        _http = http ?? SharedClient.Value;
        _ownsClient = false;
        _guard = guard ?? new UrlGuard();
        _options = options ?? new WebFetcherOptions();
    }

    public WebFetcherOptions Options => _options;

    private static HttpClient CreateSharedClient()
    {
        var handler = new SocketsHttpHandler
        {
            // Đóng và mở lại kết nối sau 2 phút -> DNS mới được áp dụng.
            PooledConnectionLifetime = TimeSpan.FromMinutes(2),

            // TỰ xử lý chuyển hướng, để guard soi được từng chặng.
            AllowAutoRedirect = false,

            AutomaticDecompression = DecompressionMethods.All,
            ConnectTimeout = TimeSpan.FromSeconds(5),
        };

        return new HttpClient(handler)
        {
            // Hạn thời gian đặt bằng CancellationToken cho từng lượt, không
            // đặt ở client dùng chung — để mỗi lượt tự chọn hạn của mình.
            Timeout = Timeout.InfiniteTimeSpan,
        };
    }

    public async Task<FetchResult> FetchAsync(Uri url, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(url);

        // Đã bị huỷ trước cả khi bắt đầu thì dừng ngay, đừng mở kết nối nào.
        // HttpClient không bảo đảm kiểm tra token trước khi gọi tới handler.
        cancellationToken.ThrowIfCancellationRequested();

        // Hạn thời gian của lượt này, ghép với token của người gọi. Người gọi
        // huỷ -> dừng; quá hạn -> cũng dừng. Thứ nào đến trước thì thắng.
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(_options.Timeout);

        var current = url;

        for (int hop = 0; hop <= _options.MaxRedirects; hop++)
        {
            var verdict = await _guard.CheckAsync(current, timeout.Token).ConfigureAwait(false);
            if (!verdict.Allowed)
            {
                return FetchResult.Fail(current, $"URL bị chặn: {verdict.Reason}");
            }

            HttpResponseMessage response;
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, current);
                request.Headers.TryAddWithoutValidation("User-Agent", _options.UserAgent);
                request.Headers.TryAddWithoutValidation("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1");

                response = await _http
                    .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                // Người gọi huỷ thật -> để ngoại lệ bay lên, đây không phải lỗi tải.
                throw;
            }
            catch (OperationCanceledException)
            {
                return FetchResult.Fail(current, $"hết thời gian sau {_options.Timeout.TotalSeconds:F0}s");
            }
            catch (HttpRequestException ex)
            {
                return FetchResult.Fail(current, $"lỗi mạng: {ex.Message}");
            }

            using (response)
            {
                // --- Chuyển hướng: đi tiếp một chặng, guard sẽ soi lại URL mới ---
                if (IsRedirect(response.StatusCode))
                {
                    var location = response.Headers.Location;
                    if (location is null)
                    {
                        return FetchResult.Fail(current, $"HTTP {(int)response.StatusCode} nhưng thiếu header Location",
                            (int)response.StatusCode);
                    }

                    current = location.IsAbsoluteUri ? location : new Uri(current, location);
                    continue;
                }

                if (!response.IsSuccessStatusCode)
                {
                    return FetchResult.Fail(current, $"HTTP {(int)response.StatusCode} {response.ReasonPhrase}",
                        (int)response.StatusCode);
                }

                string? contentType = response.Content.Headers.ContentType?.MediaType;

                if (_options.AllowedContentTypes.Count > 0 &&
                    contentType is not null &&
                    !_options.AllowedContentTypes.Contains(contentType))
                {
                    return FetchResult.Fail(current, $"kiểu nội dung '{contentType}' không được nhận",
                        (int)response.StatusCode);
                }

                // Content-Length có thể nói dối hoặc vắng mặt, nên vẫn phải đếm
                // khi đọc. Kiểm tra ở đây chỉ để khỏi tải vô ích.
                if (response.Content.Headers.ContentLength > _options.MaxBytes)
                {
                    return FetchResult.Fail(current,
                        $"trang khai báo {response.Content.Headers.ContentLength} byte, quá giới hạn {_options.MaxBytes}",
                        (int)response.StatusCode);
                }

                try
                {
                    var (body, byteCount, truncated) = await ReadCappedAsync(response, timeout.Token).ConfigureAwait(false);

                    return new FetchResult
                    {
                        Success = true,
                        Url = current,
                        Body = UntrustedText.FromWeb(body),
                        StatusCode = (int)response.StatusCode,
                        ContentType = contentType,
                        ByteCount = byteCount,
                        Truncated = truncated,
                    };
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    throw;
                }
                catch (OperationCanceledException)
                {
                    return FetchResult.Fail(current, $"hết thời gian khi đang đọc nội dung ({_options.Timeout.TotalSeconds:F0}s)",
                        (int)response.StatusCode);
                }
                catch (IOException ex)
                {
                    return FetchResult.Fail(current, $"lỗi đọc nội dung: {ex.Message}", (int)response.StatusCode);
                }
            }
        }

        return FetchResult.Fail(url, $"vượt quá {_options.MaxRedirects} lần chuyển hướng");
    }

    /// <summary>
    /// Đọc thân phản hồi nhưng DỪNG khi chạm giới hạn.
    ///
    /// Không dùng ReadAsStringAsync vì hàm đó đọc hết mới trả về — giới hạn
    /// kích thước kiểm tra sau khi đã nạp xong thì chẳng bảo vệ được gì.
    /// </summary>
    private async Task<(string Body, long ByteCount, bool Truncated)> ReadCappedAsync(
        HttpResponseMessage response, CancellationToken cancellationToken)
    {
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);

        var buffer = new byte[8192];
        var collected = new MemoryStream();
        long total = 0;
        bool truncated = false;

        while (true)
        {
            int read = await stream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false);
            if (read == 0) break;

            long remaining = _options.MaxBytes - total;
            if (read > remaining)
            {
                collected.Write(buffer, 0, (int)remaining);
                total += remaining;
                truncated = true;
                break;
            }

            collected.Write(buffer, 0, read);
            total += read;
        }

        var encoding = ResolveEncoding(response.Content.Headers.ContentType?.CharSet);
        return (encoding.GetString(collected.GetBuffer(), 0, (int)collected.Length), total, truncated);
    }

    private static Encoding ResolveEncoding(string? charSet)
    {
        if (string.IsNullOrWhiteSpace(charSet)) return Encoding.UTF8;

        try
        {
            return Encoding.GetEncoding(charSet.Trim('"', '\''));
        }
        catch (ArgumentException)
        {
            return Encoding.UTF8;   // charset lạ hoặc bịa -> UTF-8, đừng vì thế mà bỏ cả trang
        }
    }

    private static bool IsRedirect(HttpStatusCode code) => code is
        HttpStatusCode.MovedPermanently or HttpStatusCode.Found or HttpStatusCode.SeeOther or
        HttpStatusCode.TemporaryRedirect or HttpStatusCode.PermanentRedirect;

    public void Dispose()
    {
        if (_ownsClient) _http.Dispose();
    }
}
