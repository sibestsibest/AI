using Ai.Phase15.Memory;

namespace Ai.Phase15.Internet;

/// <summary>Toàn bộ khâu "lấy thông tin từ Internet", gói thành một lời gọi.</summary>
public interface ISearchService
{
    Task<InternetSearchReport> SearchAsync(string question, CancellationToken cancellationToken = default);
}

public sealed record SearchServiceOptions
{
    /// <summary>Số kết quả xin từ mỗi máy tìm kiếm cho mỗi truy vấn.</summary>
    public int ResultsPerQuery { get; init; } = 5;

    /// <summary>Số nguồn giữ lại sau khi xếp hạng.</summary>
    public int MaxSources { get; init; } = 5;

    /// <summary>Có tải nội dung trang về hay chỉ dùng đoạn trích của máy tìm kiếm.</summary>
    public bool FetchPages { get; init; } = true;

    /// <summary>Tải tối đa bao nhiêu trang. Tách khỏi MaxSources vì tải trang tốn thời gian.</summary>
    public int MaxPagesToFetch { get; init; } = 3;

    /// <summary>Số lượt tải song song. Cao quá thì thành hành vi tấn công nhẹ.</summary>
    public int FetchConcurrency { get; init; } = 3;

    /// <summary>Hạn thời gian cho CẢ lượt tìm kiếm, tính cả tra cứu và tải trang.</summary>
    public TimeSpan OverallTimeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>Điểm xếp hạng dưới ngưỡng này thì bỏ — thà ít nguồn còn hơn nguồn lạc đề.</summary>
    public double MinimumScore { get; init; } = 0.15;
}

/// <summary>
/// Kết quả một lượt tra Internet.
///
/// <see cref="Warnings"/> không phải phần phụ. Một lượt tra "thành công" mà
/// thầm lặng bỏ qua 4 trong 5 nguồn do lỗi mạng thì rất khác một lượt tra
/// lấy được cả 5 — và khác biệt đó PHẢI đi được tới người dùng, chứ không
/// biến mất trong log. Phase 14 sẽ dùng chính danh sách này để hạ độ tự tin.
/// </summary>
public sealed record InternetSearchReport
{
    public required string Question { get; init; }

    /// <summary>Các truy vấn thực sự đã gửi đi — để giải trình được.</summary>
    public IReadOnlyList<SearchQuery> Queries { get; init; } = [];

    public IReadOnlyList<RankedResult> Sources { get; init; } = [];

    /// <summary>Trang đã tải và bóc nội dung. Có thể ít hơn <see cref="Sources"/>.</summary>
    public IReadOnlyList<WebDocument> Documents { get; init; } = [];

    public IReadOnlyList<string> Warnings { get; init; } = [];

    public bool FromCache { get; init; }

    public required DateTimeOffset SearchedAt { get; init; }

    public int DuplicatesRemoved { get; init; }

    public bool HasSources => Sources.Count > 0;

    /// <summary>Số tên miền KHÁC NHAU — thước đo "nhiều nguồn độc lập" thật sự, không phải số dòng kết quả.</summary>
    public int DistinctHostCount => Sources.Select(s => s.Result.Host).Distinct(StringComparer.OrdinalIgnoreCase).Count();

    public static InternetSearchReport Empty(string question, DateTimeOffset at, params string[] warnings) =>
        new() { Question = question, SearchedAt = at, Warnings = warnings };
}

/// <summary>
/// SEARCH SERVICE — điều phối cả dây chuyền lấy tin từ Internet.
///
///     Câu hỏi
///        ↓
///     Sinh truy vấn          KeywordSearchQueryGenerator
///        ↓
///     Cache?  ──có──→ trả về ngay
///        ↓ không
///     Máy tìm kiếm           ISearchProvider (nhiều máy, lỗi riêng lẻ không làm chết cả lượt)
///        ↓
///     Bỏ trùng               ResultDeduplicator
///        ↓
///     Xếp hạng               SourceRanker
///        ↓
///     Tải trang              IWebFetcher  (+ UrlGuard chống SSRF)
///        ↓
///     Bóc nội dung           IContentExtractor
///        ↓
///     InternetSearchReport   — toàn bộ là DATA không tin cậy
///
/// RANH GIỚI PHẢI GIỮ: lớp này TRẢ VỀ thông tin, nó không kết luận gì cả, và
/// nó KHÔNG ghi gì vào memory. Biến nội dung Internet thành ký ức lâu dài hay
/// thành dữ liệu huấn luyện là việc của Phase 12 và Phase 15, sau khi đã qua
/// khâu evidence và kiểm chứng. Trộn hai việc đó vào đây thì agent sẽ học
/// thuộc mọi thứ nó đọc được trên mạng — kể cả thứ sai.
/// </summary>
public sealed class SearchService : ISearchService
{
    private readonly ISearchProvider _provider;
    private readonly IWebFetcher? _fetcher;
    private readonly IContentExtractor _extractor;
    private readonly ISearchQueryGenerator _queryGenerator;
    private readonly ISourceRanker _ranker;
    private readonly ResultDeduplicator _deduplicator;
    private readonly ISearchCache? _cache;
    private readonly IClock _clock;
    private readonly SearchServiceOptions _options;

    public SearchService(
        ISearchProvider provider,
        IWebFetcher? fetcher = null,
        IContentExtractor? extractor = null,
        ISearchQueryGenerator? queryGenerator = null,
        ISourceRanker? ranker = null,
        ResultDeduplicator? deduplicator = null,
        ISearchCache? cache = null,
        IClock? clock = null,
        SearchServiceOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(provider);

        _provider = provider;
        _fetcher = fetcher;
        _extractor = extractor ?? new HtmlContentExtractor();
        _queryGenerator = queryGenerator ?? new KeywordSearchQueryGenerator();
        _ranker = ranker ?? new SourceRanker();
        _deduplicator = deduplicator ?? new ResultDeduplicator();
        _cache = cache;
        _clock = clock ?? new SystemClock();
        _options = options ?? new SearchServiceOptions();
    }

    public SearchServiceOptions Options => _options;

    public async Task<InternetSearchReport> SearchAsync(string question, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(question);

        var startedAt = _clock.Now;

        using var overall = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        overall.CancelAfter(_options.OverallTimeout);
        var token = overall.Token;

        // --- 1. Sinh truy vấn ---
        var queries = _queryGenerator.Generate(question, _options.ResultsPerQuery);

        if (queries.Count == 0)
        {
            return InternetSearchReport.Empty(question, startedAt,
                "câu hỏi không có từ khoá nào để tra cứu");
        }

        // --- 2. Tra cứu (có cache) ---
        var warnings = new List<string>();
        var raw = new List<SearchResult>();
        bool allFromCache = true;

        foreach (var query in queries)
        {
            token.ThrowIfCancellationRequested();

            if (_cache is not null && _cache.TryGet(query.CacheKey, out var cached))
            {
                raw.AddRange(cached);
                continue;
            }

            allFromCache = false;

            try
            {
                var results = await _provider.SearchAsync(query, token).ConfigureAwait(false);

                _cache?.Set(query.CacheKey, results);
                raw.AddRange(results);

                if (_provider is CompositeSearchProvider composite)
                {
                    warnings.AddRange(composite.LastErrors);
                }
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;   // người gọi huỷ -> tôn trọng, không nuốt
            }
            catch (OperationCanceledException)
            {
                warnings.Add($"truy vấn \"{query.Text}\" hết thời gian");
            }
            catch (Exception ex) when (ex is HttpRequestException or InvalidOperationException)
            {
                warnings.Add($"truy vấn \"{query.Text}\" lỗi: {ex.Message}");
            }
        }

        if (raw.Count == 0)
        {
            return new InternetSearchReport
            {
                Question = question,
                Queries = queries,
                SearchedAt = startedAt,
                Warnings = [.. warnings, "không máy tìm kiếm nào trả về kết quả"],
            };
        }

        // --- 3. Bỏ trùng ---
        var dedup = _deduplicator.Deduplicate(raw);

        // --- 4. Xếp hạng ---
        var ranked = _ranker.Rank(question, dedup.Unique, dedup)
            .Where(r => r.Score >= _options.MinimumScore)
            .Take(_options.MaxSources)
            .ToList();

        if (ranked.Count == 0)
        {
            return new InternetSearchReport
            {
                Question = question,
                Queries = queries,
                SearchedAt = startedAt,
                DuplicatesRemoved = dedup.DuplicatesRemoved,
                FromCache = allFromCache,
                Warnings = [.. warnings,
                    $"{dedup.Unique.Count} kết quả đều dưới ngưỡng liên quan {_options.MinimumScore:F2}"],
            };
        }

        // --- 5 & 6. Tải trang + bóc nội dung ---
        var documents = new List<WebDocument>();

        if (_options.FetchPages && _fetcher is not null)
        {
            var (fetched, fetchWarnings) = await FetchDocumentsAsync(ranked, token).ConfigureAwait(false);
            documents.AddRange(fetched);
            warnings.AddRange(fetchWarnings);
        }

        return new InternetSearchReport
        {
            Question = question,
            Queries = queries,
            Sources = ranked,
            Documents = documents,
            Warnings = warnings,
            FromCache = allFromCache,
            SearchedAt = startedAt,
            DuplicatesRemoved = dedup.DuplicatesRemoved,
        };
    }

    /// <summary>
    /// Tải song song nhưng CÓ CHẶN TRÊN số lượt cùng lúc.
    ///
    /// Bắn 50 request một lúc không phải "nhanh hơn" — đó là hành vi mà phía
    /// bên kia đọc thành tấn công, và họ chặn IP là đúng. SemaphoreSlim giữ
    /// số lượt đồng thời trong mức đã khai báo.
    /// </summary>
    private async Task<(List<WebDocument> Documents, List<string> Warnings)> FetchDocumentsAsync(
        IReadOnlyList<RankedResult> ranked, CancellationToken cancellationToken)
    {
        using var gate = new SemaphoreSlim(_options.FetchConcurrency);
        var targets = ranked.Take(_options.MaxPagesToFetch).ToList();

        var tasks = targets.Select(async Task<FetchOutcome> (source) =>
        {
            await gate.WaitAsync(cancellationToken).ConfigureAwait(false);

            try
            {
                var fetched = await _fetcher!.FetchAsync(source.Url, cancellationToken).ConfigureAwait(false);

                if (!fetched.Success)
                {
                    return new FetchOutcome(null, $"không tải được {source.Url}: {fetched.Error}");
                }

                if (!_extractor.CanExtract(fetched.ContentType))
                {
                    return new FetchOutcome(null, $"không bóc được nội dung {source.Url} (kiểu '{fetched.ContentType}')");
                }

                var document = _extractor.Extract(fetched, _clock.Now);

                return document is null
                    ? new FetchOutcome(null, $"{source.Url} không có nội dung văn bản đáng kể")
                    : new FetchOutcome(document, null);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (OperationCanceledException)
            {
                return new FetchOutcome(null, $"tải {source.Url} hết thời gian");
            }
            finally
            {
                gate.Release();
            }
        });

        var outcomes = await Task.WhenAll(tasks).ConfigureAwait(false);

        return (
            [.. outcomes.Where(o => o.Document is not null).Select(o => o.Document!)],
            [.. outcomes.Where(o => o.Warning is not null).Select(o => o.Warning!)]);
    }

    /// <summary>Một lượt tải: hoặc ra tài liệu, hoặc ra lý do vì sao không ra tài liệu.</summary>
    private sealed record FetchOutcome(WebDocument? Document, string? Warning);
}
