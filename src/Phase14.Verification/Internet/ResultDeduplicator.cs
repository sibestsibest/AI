namespace Ai.Phase14.Internet;

/// <summary>
/// Gộp kết quả trùng lặp.
///
/// Trùng lặp là chuyện thường xuyên, không phải ngoại lệ: hỏi hai truy vấn
/// gần nhau, hoặc hỏi hai máy tìm kiếm, thì cùng một trang sẽ xuất hiện nhiều
/// lần dưới nhiều dạng URL khác nhau. Không gộp thì phần xếp hạng ở sau sẽ
/// hiểu sai — mười dòng cùng trỏ về một trang trông như "mười nguồn cùng
/// khẳng định", trong khi thực chất chỉ có MỘT nguồn.
///
/// Với Phase 14 (kiểm chứng) thì đây không còn là chuyện gọn gàng nữa mà là
/// chuyện đúng/sai: đếm số nguồn độc lập là cơ sở để tính độ tự tin.
///
/// HAI MỨC PHÁT HIỆN:
///
///   1. Trùng URL sau khi chuẩn hoá — chắc chắn cùng một trang
///   2. Gần trùng nội dung (Jaccard) — cùng bài nhưng khác đường dẫn,
///                                     ví dụ bản gốc và bản đăng lại
/// </summary>
public sealed class ResultDeduplicator
{
    /// <summary>Tham số theo dõi (tracking) — không ảnh hưởng nội dung trang, phải bỏ khi so sánh.</summary>
    private static readonly string[] TrackingPrefixes = ["utm_", "fbclid", "gclid", "msclkid", "ref", "ref_src", "mc_cid", "mc_eid", "igshid", "spm"];

    /// <summary>Giống nhau từ mức này trở lên thì coi là cùng một nội dung.</summary>
    public double NearDuplicateThreshold { get; init; } = 0.85;

    /// <summary>
    /// Chuẩn hoá URL để so sánh.
    ///
    /// Bốn URL dưới đây là CÙNG một trang, và nếu không chuẩn hoá thì chúng là
    /// bốn "nguồn" khác nhau:
    ///
    ///     https://Example.COM/a/     http://example.com/a
    ///     https://www.example.com/a  https://example.com/a?utm_source=x#phan-2
    /// </summary>
    public static string CanonicalUrl(Uri url)
    {
        ArgumentNullException.ThrowIfNull(url);

        string host = url.Host.ToLowerInvariant();
        if (host.StartsWith("www.", StringComparison.Ordinal)) host = host[4..];

        string path = url.AbsolutePath.TrimEnd('/');
        if (path.Length == 0) path = "/";

        // Giữ lại các tham số THỰC SỰ chọn nội dung (?id=42, ?page=3), bỏ tham số theo dõi.
        var kept = url.Query
            .TrimStart('?')
            .Split('&', StringSplitOptions.RemoveEmptyEntries)
            .Where(p => p.Length > 0)
            .Where(p => !IsTracking(p.Split('=', 2)[0]))
            .OrderBy(p => p, StringComparer.Ordinal)
            .ToArray();

        string query = kept.Length == 0 ? string.Empty : "?" + string.Join('&', kept);

        // Scheme và fragment bị bỏ hẳn: http vs https là cùng trang, và
        // fragment (#phan-2) chỉ là vị trí cuộn trong CÙNG một trang.
        return $"{host}{path}{query}";
    }

    private static bool IsTracking(string key) =>
        TrackingPrefixes.Any(p => key.StartsWith(p, StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Bỏ trùng, giữ lại bản TỐT NHẤT của mỗi nhóm (thứ hạng nhỏ nhất), và
    /// đếm xem mỗi nhóm được bao nhiêu máy tìm kiếm nhắc tới.
    /// </summary>
    public DeduplicationReport Deduplicate(IEnumerable<SearchResult> results)
    {
        ArgumentNullException.ThrowIfNull(results);

        var groups = new List<DuplicateGroup>();
        var byUrl = new Dictionary<string, DuplicateGroup>(StringComparer.Ordinal);
        int duplicates = 0;

        foreach (var result in results)
        {
            string canonical = CanonicalUrl(result.Url);

            // --- Mức 1: trùng URL chuẩn hoá ---
            if (byUrl.TryGetValue(canonical, out var existing))
            {
                existing.Absorb(result);
                duplicates++;
                continue;
            }

            // --- Mức 2: gần trùng nội dung ---
            var fingerprint = Fingerprint(result);
            var similar = groups.FirstOrDefault(g => Jaccard(g.Fingerprint, fingerprint) >= NearDuplicateThreshold);

            if (similar is not null)
            {
                similar.Absorb(result);
                duplicates++;
                continue;
            }

            var group = new DuplicateGroup(result, canonical, fingerprint);
            groups.Add(group);
            byUrl[canonical] = group;
        }

        var unique = groups
            .Select(g => g.Best)
            .ToList();

        return new DeduplicationReport(unique, duplicates,
            groups.ToDictionary(g => CanonicalUrl(g.Best.Url), g => g.Providers.Count, StringComparer.Ordinal));
    }

    /// <summary>Dấu vân nội dung: tập từ khoá của tiêu đề + đoạn trích.</summary>
    private static HashSet<string> Fingerprint(SearchResult result)
    {
        var words = result.Title.Keywords();
        words.UnionWith(result.Snippet.Keywords());
        return words;
    }

    /// <summary>|A ∩ B| / |A ∪ B|. Hai tập rỗng coi là KHÔNG giống nhau, tránh gộp bừa mọi kết quả rỗng.</summary>
    public static double Jaccard(IReadOnlySet<string> a, IReadOnlySet<string> b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Count == 0 || b.Count == 0) return 0;

        int intersection = a.Count(b.Contains);
        int union = a.Count + b.Count - intersection;

        return union == 0 ? 0 : (double)intersection / union;
    }

    private sealed class DuplicateGroup(SearchResult first, string canonical, HashSet<string> fingerprint)
    {
        public SearchResult Best { get; private set; } = first;

        public string Canonical { get; } = canonical;

        public HashSet<string> Fingerprint { get; } = fingerprint;

        public HashSet<string> Providers { get; } = new(StringComparer.OrdinalIgnoreCase) { first.Provider };

        public void Absorb(SearchResult other)
        {
            Providers.Add(other.Provider);

            // Giữ bản có thứ hạng tốt hơn; bằng nhau thì giữ bản có đoạn trích dài hơn.
            if (other.Rank < Best.Rank ||
                (other.Rank == Best.Rank && other.Snippet.Length > Best.Snippet.Length))
            {
                Best = other;
            }
        }
    }
}

/// <summary>
/// Kết quả bỏ trùng. <see cref="ProviderAgreement"/> là phần có giá trị nhất:
/// nó nói mỗi trang được BAO NHIÊU máy tìm kiếm độc lập nhắc tới.
/// </summary>
public sealed record DeduplicationReport(
    IReadOnlyList<SearchResult> Unique,
    int DuplicatesRemoved,
    IReadOnlyDictionary<string, int> ProviderAgreement)
{
    public int AgreementFor(Uri url) =>
        ProviderAgreement.TryGetValue(ResultDeduplicator.CanonicalUrl(url), out int count) ? count : 1;
}
