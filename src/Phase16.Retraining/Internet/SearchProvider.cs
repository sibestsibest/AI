using System.Text.Json;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Internet;

/// <summary>
/// Một máy tìm kiếm.
///
/// Trừu tượng hoá ra interface không phải để cho "đúng SOLID". Có ba lý do rất
/// cụ thể: (1) test phải chạy được KHÔNG CẦN MẠNG, (2) mỗi máy tìm kiếm có
/// API và giới hạn khác nhau, (3) hỏi nhiều máy rồi đối chiếu là cách duy nhất
/// để biết một thông tin có được nhiều nguồn độc lập xác nhận hay không —
/// việc mà Phase 14 sẽ cần đến.
/// </summary>
public interface ISearchProvider
{
    string Name { get; }

    Task<IReadOnlyList<SearchResult>> SearchAsync(SearchQuery query, CancellationToken cancellationToken = default);
}

/// <summary>
/// Máy tìm kiếm ngoại tuyến trên một kho tài liệu nạp sẵn.
///
/// Đây KHÔNG phải mã tạm để test cho qua. Nó là thứ giữ cho toàn bộ Phase 12
/// chạy và kiểm chứng được mà không cần mạng, không cần API key, và cho ra
/// kết quả TẤT ĐỊNH. Một bộ test phụ thuộc Internet là bộ test hỏng: hôm nay
/// xanh, mai đỏ, mà chẳng ai sửa gì.
/// </summary>
public sealed class StaticSearchProvider : ISearchProvider
{
    private readonly List<SearchResult> _corpus = [];

    public StaticSearchProvider(string name = "static")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
    }

    public string Name { get; }

    public int DocumentCount => _corpus.Count;

    /// <summary>Trùng dưới ngưỡng này thì không trả về — cùng lý do như SearchTool của Phase 11.</summary>
    public double MinimumRelevance { get; init; } = 0.25;

    public StaticSearchProvider Add(string title, string url, string snippet)
    {
        _corpus.Add(SearchResult.Create(title, url, snippet, Name, _corpus.Count + 1));
        return this;
    }

    public Task<IReadOnlyList<SearchResult>> SearchAsync(SearchQuery query, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);
        cancellationToken.ThrowIfCancellationRequested();

        var words = Keywords.Extract(query.Text);

        var hits = _corpus
            .Select(r => new
            {
                Result = r,
                Score = Keywords.Relevance(words, r.Title.Keywords()) * 0.6
                      + Keywords.Relevance(words, r.Snippet.Keywords()) * 0.4,
            })
            .Where(x => x.Score >= MinimumRelevance)
            .OrderByDescending(x => x.Score)
            .ThenBy(x => x.Result.Url.AbsoluteUri, StringComparer.Ordinal)
            .Take(query.MaxResults)
            .Select((x, i) => x.Result with { Rank = i + 1 })
            .ToList();

        return Task.FromResult<IReadOnlyList<SearchResult>>(hits);
    }
}

public sealed record SearxProviderOptions
{
    /// <summary>
    /// Địa chỉ một máy SearxNG. Mặc định là localhost để KHÔNG có hành vi
    /// mạng nào xảy ra ngoài ý muốn — muốn dùng thật thì phải khai báo rõ.
    /// </summary>
    public required Uri Endpoint { get; init; }

    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(8);

    /// <summary>Danh mục SearxNG ("general", "science", "it"...).</summary>
    public string Categories { get; init; } = "general";

    public string? Language { get; init; }
}

/// <summary>
/// Máy tìm kiếm thật: SearxNG qua API JSON.
///
/// Vì sao chọn SearxNG chứ không phải Google/Bing/Brave? Vì nó tự dựng được,
/// không cần API key, không ràng buộc hợp đồng — nên project này chạy được
/// end-to-end mà người đọc không phải đăng ký gì cả. Endpoint là tham số
/// cấu hình, nên muốn đổi sang nhà cung cấp khác thì viết một ISearchProvider
/// mới, không phải sửa chỗ nào khác.
///
/// Mọi trường đọc từ JSON đều là dữ liệu KHÔNG TIN CẬY — kể cả URL. Một máy
/// tìm kiếm bị chiếm quyền hoàn toàn có thể trả về "file:///etc/passwd" hoặc
/// "http://169.254.169.254/...", nên URL phải qua guard trước khi vào danh
/// sách kết quả.
/// </summary>
public sealed class SearxSearchProvider(
    SearxProviderOptions options,
    HttpClient? http = null,
    IUrlGuard? guard = null) : ISearchProvider
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private readonly SearxProviderOptions _options = options ?? throw new ArgumentNullException(nameof(options));
    private readonly HttpClient _http = http ?? new HttpClient();
    private readonly IUrlGuard _guard = guard ?? new UrlGuard();

    public string Name => "searx";

    public async Task<IReadOnlyList<SearchResult>> SearchAsync(SearchQuery query, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(_options.Timeout);

        var requestUrl = BuildRequestUrl(query);

        using var response = await _http.GetAsync(requestUrl, timeout.Token).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token).ConfigureAwait(false);
        var payload = await JsonSerializer
            .DeserializeAsync<SearxResponse>(stream, JsonOptions, timeout.Token)
            .ConfigureAwait(false);

        if (payload?.Results is null) return [];

        var results = new List<SearchResult>();
        int rank = 1;

        foreach (var item in payload.Results)
        {
            if (results.Count >= query.MaxResults) break;
            if (string.IsNullOrWhiteSpace(item.Url)) continue;

            if (!Uri.TryCreate(item.Url, UriKind.Absolute, out var url)) continue;

            // Máy tìm kiếm không được tin — URL của nó phải qua guard.
            if (!_guard.Check(url).Allowed) continue;

            results.Add(new SearchResult
            {
                Url = url,
                Title = UntrustedText.FromWeb(item.Title ?? url.Host),
                Snippet = UntrustedText.FromWeb(item.Content ?? string.Empty),
                Provider = Name,
                Rank = rank++,
            });
        }

        return results;
    }

    /// <summary>URL thật sẽ gửi đi — công khai để test kiểm chứng được phần escape tham số.</summary>
    public Uri BuildRequestUrl(SearchQuery query)
    {
        var parameters = new List<string>
        {
            "q=" + Uri.EscapeDataString(query.Text),
            "format=json",
            "categories=" + Uri.EscapeDataString(_options.Categories),
        };

        if (!string.IsNullOrWhiteSpace(_options.Language))
        {
            parameters.Add("language=" + Uri.EscapeDataString(_options.Language));
        }

        var builder = new UriBuilder(_options.Endpoint) { Query = string.Join('&', parameters) };
        return builder.Uri;
    }

    private sealed record SearxResponse(List<SearxItem>? Results);

    private sealed record SearxItem(string? Url, string? Title, string? Content);
}

/// <summary>
/// Hỏi nhiều máy tìm kiếm một lượt.
///
/// ĐIỂM QUAN TRỌNG NHẤT ở đây là xử lý lỗi: MỘT máy chết KHÔNG được làm cả
/// lượt tìm kiếm chết. Nếu chờ tất cả cùng thành công thì độ tin cậy của hệ
/// thống bằng độ tin cậy của mắt yếu nhất. Nên mỗi máy được bọc riêng, lỗi
/// được GHI LẠI thành cảnh báo (không im lặng bỏ qua), và phần còn lại vẫn
/// được dùng.
///
/// Chỉ khi TẤT CẢ đều chết thì mới coi là thất bại — và khi đó agent phải nói
/// thật là nó không tra được, thay vì trả lời như thể đã tra.
/// </summary>
public sealed class CompositeSearchProvider(params ISearchProvider[] providers) : ISearchProvider
{
    private readonly ISearchProvider[] _providers = providers is { Length: > 0 }
        ? providers
        : throw new ArgumentException("Cần ít nhất một máy tìm kiếm.", nameof(providers));

    public string Name => "composite(" + string.Join('+', _providers.Select(p => p.Name)) + ")";

    public IReadOnlyList<ISearchProvider> Providers => _providers;

    /// <summary>Lỗi của lượt gần nhất — để <see cref="SearchService"/> đưa vào cảnh báo.</summary>
    public IReadOnlyList<string> LastErrors { get; private set; } = [];

    public async Task<IReadOnlyList<SearchResult>> SearchAsync(SearchQuery query, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);

        var tasks = _providers.Select(async provider =>
        {
            try
            {
                var results = await provider.SearchAsync(query, cancellationToken).ConfigureAwait(false);
                return (provider.Name, Results: results, Error: (string?)null);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (OperationCanceledException)
            {
                return (provider.Name, Results: (IReadOnlyList<SearchResult>)[], Error: "hết thời gian");
            }
            catch (Exception ex) when (ex is HttpRequestException or JsonException or InvalidOperationException)
            {
                return (provider.Name, Results: (IReadOnlyList<SearchResult>)[], Error: ex.Message);
            }
        });

        var outcomes = await Task.WhenAll(tasks).ConfigureAwait(false);

        LastErrors = [.. outcomes
            .Where(o => o.Error is not null)
            .Select(o => $"máy tìm kiếm '{o.Name}' lỗi: {o.Error}")];

        return [.. outcomes.SelectMany(o => o.Results)];
    }
}
