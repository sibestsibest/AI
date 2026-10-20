using Ai.Phase15.Internet;
using Ai.Phase15.Memory;

namespace Ai.Phase15.Grounding;

/// <summary>Thu thập evidence từ mọi kênh cho một câu hỏi.</summary>
public interface IEvidenceCollector
{
    Task<EvidenceSet> CollectAsync(string question, CancellationToken cancellationToken = default);
}

public sealed record EvidenceCollectorOptions
{
    public bool UseKnowledge { get; init; } = true;

    public bool UseMemory { get; init; } = true;

    public bool UseConversation { get; init; } = true;

    /// <summary>
    /// Có ra Internet hay không. Mặc định BẬT nhưng chỉ có hiệu lực khi
    /// thực sự có <see cref="ISearchService"/> được tiêm vào.
    /// </summary>
    public bool UseInternet { get; init; } = true;

    public int MaxPerChannel { get; init; } = 3;

    /// <summary>Số mẩu giữ lại sau khi xếp hạng.</summary>
    public int MaxEvidence { get; init; } = 8;

    /// <summary>Điểm dưới ngưỡng này thì bỏ — thà ít evidence hơn evidence lạc đề.</summary>
    public double MinimumScore { get; init; } = 0.12;

    /// <summary>
    /// Chỉ ra Internet khi các kênh nội bộ không đủ.
    ///
    /// Bật thì tiết kiệm và nhanh hơn, nhưng đổi lại agent mất khả năng phát
    /// hiện kho nội bộ đã lạc hậu. Mặc định TẮT: hỏi cả hai rồi để phần xếp
    /// hạng quyết định, vì Phase 14 cần thấy được khi các kênh nói khác nhau.
    /// </summary>
    public bool InternetOnlyWhenInternalIsWeak { get; init; }

    /// <summary>Ngưỡng "nội bộ đã đủ" cho tuỳ chọn ở trên.</summary>
    public double InternalSufficiencyThreshold { get; init; } = 0.6;
}

/// <summary>
/// Toàn bộ evidence gom được cho một câu hỏi, đã bỏ trùng và xếp hạng.
///
/// Đây là đầu ra chính của Phase 13, và là đầu vào của Phase 14.
/// </summary>
public sealed record EvidenceSet
{
    public required string Question { get; init; }

    /// <summary>Câu hỏi sau khi được ngữ cảnh bổ sung (nếu có).</summary>
    public string? ExpandedQuestion { get; init; }

    /// <summary>Evidence đã xếp hạng, tốt nhất trước.</summary>
    public IReadOnlyList<Evidence> Items { get; init; } = [];

    public IReadOnlyList<string> Warnings { get; init; } = [];

    public required DateTimeOffset CollectedAt { get; init; }

    public int DuplicatesRemoved { get; init; }

    /// <summary>Báo cáo tra Internet, nếu có — giữ lại để truy nguồn.</summary>
    public InternetSearchReport? InternetReport { get; init; }

    public bool HasEvidence => Items.Count > 0;

    public Evidence? Best => Items.Count > 0 ? Items[0] : null;

    /// <summary>Số KÊNH khác nhau đã góp evidence.</summary>
    public int DistinctChannelCount => Items.Select(e => e.Kind).Distinct().Count();

    public IReadOnlyList<Evidence> FromKind(SourceKind kind) => [.. Items.Where(e => e.Kind == kind)];

    /// <summary>Có mẩu nào đến từ nguồn đã kiểm chứng hay không.</summary>
    public bool HasVerifiedEvidence => Items.Any(e => e.Trust == TrustLevel.Verified);

    /// <summary>
    /// Câu trả lời dựng từ tập này có phải cảnh báo "chưa kiểm chứng" hay không.
    ///
    /// Điều kiện: có dùng nguồn không tin cậy VÀ không có nguồn đã kiểm nào
    /// đỡ lưng. Có knowledge xác nhận thì không cần cảnh báo nữa.
    /// </summary>
    public bool RequiresDisclosure => Items.Any(e => e.Source.RequiresDisclosure) && !HasVerifiedEvidence;

    /// <summary>Độ tin cậy cao nhất trong tập — Phase 14 sẽ dùng để chặn dưới.</summary>
    public double BestReliability => Items.Count == 0 ? 0 : Items.Max(e => e.Reliability);

    public static EvidenceSet Empty(string question, DateTimeOffset at, params string[] warnings) =>
        new() { Question = question, CollectedAt = at, Warnings = warnings };
}

/// <summary>
/// EVIDENCE COLLECTOR — nơi năm kênh gặp nhau.
///
///     Câu hỏi
///        ↓
///     (+ ngữ cảnh hội thoại nếu câu hỏi quá ngắn)
///        ↓
///     ┌───────────┬────────┬──────────────┬───────────┬──────────┐
///     ↓           ↓        ↓              ↓           ↓          ↓
///  Knowledge   Memory  Conversation   Reasoning   Internet
///  (Phase 13)  (P10)      (P13)          (P9)       (P12)
///     └───────────┴────────┴──────────────┴───────────┴──────────┘
///        ↓
///     Bỏ trùng      EvidenceDeduplicator  (gộp chéo kênh, đếm đồng thuận)
///        ↓
///     Xếp hạng      EvidenceRanker        (relevance × reliability × tươi)
///        ↓
///     EvidenceSet
///
/// HAI ĐIỀU LỚP NÀY KHÔNG LÀM, và đó là chủ ý:
///
///   1. KHÔNG kết luận gì. Nó trả về căn cứ, không trả về câu trả lời. Việc
///      dựng câu trả lời và kiểm chứng là của Phase 14.
///   2. KHÔNG ghi gì vào memory hay knowledge. Đọc thì đọc cả năm kênh, ghi
///      thì không ghi kênh nào. Nội dung Internet muốn thành kiến thức phải
///      đi qua <see cref="KnowledgeStore.TryPromote"/>, và cửa đó đóng với nó.
/// </summary>
public sealed class EvidenceCollector : IEvidenceCollector
{
    private readonly KnowledgeStore? _knowledge;
    private readonly MemoryStore? _memory;
    private readonly ConversationContext? _context;
    private readonly ISearchService? _internet;
    private readonly EvidenceDeduplicator _deduplicator;
    private readonly IEvidenceRanker _ranker;
    private readonly IClock _clock;
    private readonly EvidenceCollectorOptions _options;

    public EvidenceCollector(
        KnowledgeStore? knowledge = null,
        MemoryStore? memory = null,
        ConversationContext? context = null,
        ISearchService? internet = null,
        EvidenceDeduplicator? deduplicator = null,
        IEvidenceRanker? ranker = null,
        IClock? clock = null,
        EvidenceCollectorOptions? options = null)
    {
        _knowledge = knowledge;
        _memory = memory;
        _context = context;
        _internet = internet;
        _deduplicator = deduplicator ?? new EvidenceDeduplicator();
        _ranker = ranker ?? new EvidenceRanker();
        _clock = clock ?? new SystemClock();
        _options = options ?? new EvidenceCollectorOptions();
    }

    public EvidenceCollectorOptions Options => _options;

    public bool HasInternet => _internet is not null;

    public async Task<EvidenceSet> CollectAsync(string question, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(question);

        var startedAt = _clock.Now;

        if (Keywords.Extract(question).Count == 0 && _context is null)
        {
            return EvidenceSet.Empty(question, startedAt, "câu hỏi không có từ khoá nào để tìm căn cứ");
        }

        // --- Ngữ cảnh: bổ sung cho câu hỏi quá ngắn ---
        string effective = _options.UseConversation && _context is not null
            ? _context.Expand(question)
            : question;

        var warnings = new List<string>();
        var collected = new List<Evidence>();

        // --- Kênh 1: Knowledge ---
        if (_options.UseKnowledge && _knowledge is not null)
        {
            collected.AddRange(_knowledge
                .Search(effective, _options.MaxPerChannel)
                .Select(hit => Evidence.FromKnowledge(hit.Entry)));
        }

        // --- Kênh 2: Memory ---
        if (_options.UseMemory && _memory is not null)
        {
            collected.AddRange(_memory
                .Search(effective, _options.MaxPerChannel)
                .Select(hit => Evidence.FromMemory(hit.Fact)));
        }

        // --- Kênh 3: Conversation ---
        if (_options.UseConversation && _context is not null)
        {
            var questionWords = Keywords.Extract(effective);

            collected.AddRange(_context
                .Recent(_options.MaxPerChannel)
                // Chỉ lấy lượt THỰC SỰ liên quan. Lấy hết mấy lượt gần nhất
                // thì mọi câu hỏi đều kéo theo cả đoạn chat vừa rồi làm "căn cứ".
                .Where(t => Keywords.Relevance(questionWords, Keywords.Extract(t.UserInput)) >= 0.3)
                .Select(Evidence.FromConversation));
        }

        // --- Kênh 4: Internet ---
        InternetSearchReport? report = null;
        bool internalIsStrong = IsInternalStrong(collected, effective);
        bool skipInternet = _options.InternetOnlyWhenInternalIsWeak && internalIsStrong;

        if (_options.UseInternet && _internet is not null && !skipInternet)
        {
            try
            {
                report = await _internet.SearchAsync(effective, cancellationToken).ConfigureAwait(false);
                collected.AddRange(FromInternet(report));
                warnings.AddRange(report.Warnings);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (OperationCanceledException)
            {
                warnings.Add("tra Internet hết thời gian");
            }
            catch (Exception ex) when (ex is HttpRequestException or InvalidOperationException)
            {
                warnings.Add($"tra Internet lỗi: {ex.Message}");
            }
        }
        else if (skipInternet)
        {
            warnings.Add("không tra Internet vì các kênh nội bộ đã đủ căn cứ");
        }

        if (collected.Count == 0)
        {
            return new EvidenceSet
            {
                Question = question,
                ExpandedQuestion = effective == question ? null : effective,
                CollectedAt = startedAt,
                InternetReport = report,
                Warnings = [.. warnings, "không tìm được căn cứ nào từ bất kỳ kênh nào"],
            };
        }

        // --- Bỏ trùng rồi xếp hạng. Thứ tự này không đảo được: Agreement do
        //     bước bỏ trùng tính ra, mà xếp hạng lại cần Agreement. ---
        var dedup = _deduplicator.Deduplicate(collected);

        var ranked = _ranker.Rank(effective, dedup.Unique, _clock.Now)
            .Where(e => e.Score >= _options.MinimumScore)
            .Take(_options.MaxEvidence)
            .ToList();

        if (ranked.Count == 0)
        {
            warnings.Add($"{dedup.Unique.Count} mẩu căn cứ đều dưới ngưỡng liên quan {_options.MinimumScore:F2}");
        }

        return new EvidenceSet
        {
            Question = question,
            ExpandedQuestion = effective == question ? null : effective,
            Items = ranked,
            Warnings = warnings,
            CollectedAt = startedAt,
            DuplicatesRemoved = dedup.DuplicatesRemoved,
            InternetReport = report,
        };
    }

    /// <summary>
    /// Dựng evidence từ báo cáo tra Internet.
    ///
    /// Ưu tiên nội dung trang ĐÃ TẢI (<see cref="WebDocument"/>) vì nó là nội
    /// dung thật; chỉ khi không tải được mới dùng đoạn trích của máy tìm kiếm,
    /// vì đoạn trích chỉ là lời máy tìm kiếm NÓI về trang đó.
    /// </summary>
    private static IEnumerable<Evidence> FromInternet(InternetSearchReport report)
    {
        foreach (var ranked in report.Sources)
        {
            var document = report.Documents.FirstOrDefault(d => d.Url == ranked.Url);

            var content = document?.Text ?? ranked.Result.Snippet;
            if (content.IsEmpty) continue;

            var title = document?.Title ?? ranked.Result.Title;

            var source = Source.FromInternet(
                title.Sanitized,
                ranked.Url,
                document?.RetrievedAt ?? report.SearchedAt,
                // Điểm xếp hạng của Phase 12 làm độ tin cậy nền. Nó đã tính
                // uy tín tên miền, số nguồn đồng thuận và thứ hạng.
                Math.Clamp(ranked.Score, 0.05, 1.0));

            string suffix = ranked.Url.AbsoluteUri.GetHashCode(StringComparison.Ordinal).ToString("x8");

            yield return Evidence.FromWeb(content, source, suffix);
        }
    }

    /// <summary>Các kênh nội bộ đã đủ mạnh chưa — cho tuỳ chọn bỏ qua Internet.</summary>
    private bool IsInternalStrong(List<Evidence> internalEvidence, string question)
    {
        if (internalEvidence.Count == 0) return false;

        var words = Keywords.Extract(question);

        return internalEvidence.Any(e =>
            e.Trust == TrustLevel.Verified &&
            Keywords.Relevance(words, e.Keywords()) >= _options.InternalSufficiencyThreshold);
    }
}
