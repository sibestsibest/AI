using Ai.Phase15.Agent;
using Ai.Phase15.Memory;

namespace Ai.Phase15.Learning;

/// <summary>Khác biệt giữa hai phiên bản tập dữ liệu.</summary>
public sealed record DatasetDiff(
    int FromVersion,
    int ToVersion,
    IReadOnlyList<DatasetExample> Added,
    IReadOnlyList<DatasetExample> Removed)
{
    public bool IsEmpty => Added.Count == 0 && Removed.Count == 0;

    public override string ToString() =>
        $"v{FromVersion} -> v{ToVersion}: +{Added.Count} / -{Removed.Count}";
}

/// <summary>
/// KHO TẬP DỮ LIỆU CÓ PHIÊN BẢN — append-only.
///
/// <see cref="Commit"/> không bao giờ sửa phiên bản đang có; nó tạo phiên bản
/// mới trỏ về cha. Nhờ vậy ba câu hỏi sau luôn trả lời được, kể cả nhiều tháng
/// sau:
///
///   • model này train từ đúng những ví dụ nào?      -> vân tay + phiên bản
///   • ví dụ nào vừa được thêm trước khi nó tệ đi?   -> Diff
///   • ai/điều gì đưa ví dụ đó vào?                  -> Provenance + FeedbackId
///
/// Không có <c>Update</c> và không có <c>Delete</c>. Muốn bỏ một ví dụ thì tạo
/// phiên bản mới không có nó (<see cref="CommitWithout"/>) — lịch sử vẫn còn
/// nguyên. Đây là điều kiện tối thiểu để Phase 16 quay lui được: quay lui một
/// model mà tập dữ liệu của nó đã bị sửa tại chỗ thì không dựng lại được gì.
/// </summary>
public sealed class DatasetRepository
{
    private readonly List<DatasetVersion> _versions = [];
    private readonly IClock _clock;

    public DatasetRepository(IClock? clock = null) => _clock = clock ?? new SystemClock();

    public IReadOnlyList<DatasetVersion> Versions => _versions;

    public int VersionCount => _versions.Count;

    /// <summary>Phiên bản mới nhất. Ném nếu kho còn rỗng — tránh trả về "tập rỗng" trông như hợp lệ.</summary>
    public DatasetVersion Current =>
        _versions.Count > 0
            ? _versions[^1]
            : throw new InvalidOperationException("kho chưa có phiên bản nào — hãy Seed trước");

    public bool IsEmpty => _versions.Count == 0;

    /// <summary>
    /// Đặt phiên bản đầu tiên từ dữ liệu người tự gán nhãn.
    ///
    /// v1 luôn là <see cref="CandidateProvenance.CuratedSeed"/>: gốc của tập
    /// dữ liệu phải là dữ liệu có người chịu trách nhiệm. Nếu để phản hồi tự
    /// dựng nên v1 thì không có nền nào để so, và mọi kiểm tra chất lượng về
    /// sau đều so với chính phản hồi.
    /// </summary>
    public DatasetVersion Seed(IEnumerable<(string Text, Intent Label)> examples, string note = "hạt giống ban đầu")
    {
        ArgumentNullException.ThrowIfNull(examples);

        if (!IsEmpty) throw new InvalidOperationException("kho đã có phiên bản — Seed chỉ gọi một lần");

        var now = _clock.Now;
        var seeded = examples.Select(e => new DatasetExample
        {
            Input = e.Text,
            Label = e.Label,
            Provenance = CandidateProvenance.CuratedSeed,
            AddedAt = now,
        });

        var version = DatasetVersion.Create(1, seeded, now, parentVersion: null, note: note);
        _versions.Add(version);

        return version;
    }

    /// <summary>
    /// Tạo phiên bản mới = phiên bản hiện tại + các ứng viên ĐÃ DUYỆT.
    ///
    /// Nhận cả danh sách lý do loại để ghi vào phiên bản: một phiên bản tập dữ
    /// liệu phải kể được cả những gì KHÔNG vào được và vì sao. Chỉ ghi phần
    /// được nhận thì mất hẳn dấu vết của một đợt phản hồi bị loại hàng loạt —
    /// mà đó chính là dấu hiệu đáng điều tra nhất.
    /// </summary>
    public DatasetVersion Commit(IEnumerable<TrainingCandidate> approved, string note = "",
        IEnumerable<string>? rejectedReasons = null)
    {
        ArgumentNullException.ThrowIfNull(approved);

        var parent = Current;
        var now = _clock.Now;

        var added = approved
            .Where(c => c.MayBecomeTrainingData)
            .Select(c => new DatasetExample
            {
                Input = c.Input,
                Label = c.Label,
                Provenance = c.Provenance,
                FeedbackId = c.FeedbackId,
                AddedAt = now,
            })
            .ToList();

        var version = DatasetVersion.Create(
            parent.Version + 1,
            parent.Examples.Concat(added),
            now,
            parentVersion: parent.Version,
            note: note,
            added: added,
            rejectedReasons: rejectedReasons);

        _versions.Add(version);
        return version;
    }

    /// <summary>
    /// Tạo phiên bản mới BỎ một số ví dụ — cách duy nhất để "xoá" dữ liệu.
    ///
    /// Dùng khi phát hiện một ví dụ đã duyệt hoá ra là sai. Ví dụ cũ vẫn nằm
    /// trong phiên bản cũ, nên model đã train từ phiên bản đó vẫn tái lập được.
    /// </summary>
    public DatasetVersion CommitWithout(IEnumerable<string> keysToRemove, string note)
    {
        ArgumentNullException.ThrowIfNull(keysToRemove);

        var removeSet = keysToRemove.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var parent = Current;

        var kept = parent.Examples.Where(e => !removeSet.Contains(e.Key)).ToList();

        var version = DatasetVersion.Create(
            parent.Version + 1,
            kept,
            _clock.Now,
            parentVersion: parent.Version,
            note: note);

        _versions.Add(version);
        return version;
    }

    public DatasetVersion? Get(int version) => _versions.FirstOrDefault(v => v.Version == version);

    /// <summary>Phiên bản nào có đúng vân tay này — dùng khi chỉ còn biết vân tay của một model.</summary>
    public DatasetVersion? FindByFingerprint(string fingerprint)
    {
        ArgumentNullException.ThrowIfNull(fingerprint);

        return _versions.FirstOrDefault(v =>
            v.Fingerprint.Equals(fingerprint, StringComparison.OrdinalIgnoreCase));
    }

    public DatasetDiff Diff(int fromVersion, int toVersion)
    {
        var from = Get(fromVersion) ?? throw new ArgumentException($"không có phiên bản v{fromVersion}", nameof(fromVersion));
        var to = Get(toVersion) ?? throw new ArgumentException($"không có phiên bản v{toVersion}", nameof(toVersion));

        var fromKeys = from.Examples.Select(e => e.Key).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var toKeys = to.Examples.Select(e => e.Key).ToHashSet(StringComparer.OrdinalIgnoreCase);

        return new DatasetDiff(
            fromVersion,
            toVersion,
            [.. to.Examples.Where(e => !fromKeys.Contains(e.Key))],
            [.. from.Examples.Where(e => !toKeys.Contains(e.Key))]);
    }
}
