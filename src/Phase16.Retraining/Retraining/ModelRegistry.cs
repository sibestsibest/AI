using Ai.Phase16.Agent;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Retraining;

/// <summary>Chuyện gì đã xảy ra với một phiên bản model.</summary>
public enum ModelEventKind
{
    Registered,
    Deployed,
    Rejected,
    RolledBack,
}

/// <summary>Một dòng trong lịch sử triển khai. Ghi cả lý do, không chỉ kết quả.</summary>
public sealed record ModelEvent(
    ModelEventKind Kind,
    int Version,
    DateTimeOffset At,
    IReadOnlyList<string> Reasons)
{
    public override string ToString() =>
        $"[{At:yyyy-MM-dd HH:mm}] {Kind} m{Version}" +
        (Reasons.Count == 0 ? "" : $" — {string.Join("; ", Reasons)}");
}

/// <summary>
/// SỔ ĐĂNG KÝ MODEL — nơi duy nhất quyết định model nào đang chạy.
///
/// Ba tính chất, và cả ba đều là điều kiện để "quay lui" có nghĩa:
///
///   1. ĐĂNG KÝ ≠ TRIỂN KHAI. Một model được train xong thì được ghi vào sổ,
///      nhưng chỉ <see cref="Deploy"/> mới đổi model đang chạy — và
///      <see cref="Deploy"/> từ chối nếu cửa đánh giá không cho qua. Không có
///      đường nào khác để đổi <see cref="Active"/>.
///
///   2. MỌI PHIÊN BẢN ĐỀU ĐƯỢC GIỮ, kể cả phiên bản bị từ chối. Model bị từ
///      chối là dữ liệu quý: nó cho biết cấu hình nào đã thử và hỏng ra sao.
///      Xoá đi thì lần sau có người thử lại đúng cấu hình đó.
///
///   3. LỊCH SỬ GHI CẢ LÝ DO. "m3 bị từ chối" không giúp được gì; "m3 bị từ
///      chối vì làm hỏng 2 câu đang chạy được, cụ thể là X và Y" thì hành động
///      được ngay.
///
/// QUAY LUI CHỈ VỀ ĐƯỢC MODEL ĐÃ TỪNG CHẠY. Quay về một model chưa bao giờ
/// vượt cửa đánh giá không phải là quay lui — đó là triển khai một model chưa
/// được duyệt, bằng một cái tên nghe an toàn hơn.
/// </summary>
public sealed class ModelRegistry
{
    private readonly List<ModelVersion> _versions = [];
    private readonly List<ModelEvent> _history = [];
    private readonly List<int> _deployedVersions = [];
    private readonly IClock _clock;

    public ModelRegistry(IClock? clock = null) => _clock = clock ?? new SystemClock();

    public IReadOnlyList<ModelVersion> Versions => _versions;

    public IReadOnlyList<ModelEvent> History => _history;

    /// <summary>Model đang chạy. Null khi chưa có model nào vượt được cửa đánh giá.</summary>
    public ModelVersion? Active { get; private set; }

    public int NextVersionNumber => _versions.Count == 0 ? 1 : _versions.Max(v => v.Version) + 1;

    /// <summary>Số phiên bản đã từng được triển khai — không tính những bản bị từ chối.</summary>
    public int DeployedCount => _deployedVersions.Distinct().Count();

    public ModelVersion? Get(int version) => _versions.FirstOrDefault(v => v.Version == version);

    /// <summary>Ghi một model vào sổ. KHÔNG làm nó thành model đang chạy.</summary>
    public ModelVersion Register(ModelVersion model)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (_versions.Any(v => v.Version == model.Version))
        {
            throw new InvalidOperationException($"đã có phiên bản m{model.Version} trong sổ");
        }

        _versions.Add(model);
        _history.Add(new ModelEvent(ModelEventKind.Registered, model.Version, _clock.Now,
            [$"train từ dữ liệu v{model.DatasetVersion}, {model.Metrics}"]));

        return model;
    }

    /// <summary>
    /// Triển khai một model — chỉ khi cửa đánh giá cho qua.
    ///
    /// Trả về chính quyết định được truyền vào, đã ghi vào lịch sử. Người gọi
    /// không có cách nào bỏ qua bước này: <see cref="Active"/> chỉ có setter
    /// riêng tư và chỉ hàm này cùng <see cref="Rollback"/> đụng tới.
    /// </summary>
    public DeploymentDecision Deploy(ModelVersion model, DeploymentDecision decision)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(decision);

        if (Get(model.Version) is null)
        {
            throw new InvalidOperationException($"m{model.Version} chưa được ghi vào sổ — hãy Register trước");
        }

        if (!decision.Approved)
        {
            _history.Add(new ModelEvent(ModelEventKind.Rejected, model.Version, _clock.Now, decision.Blockers));
            return decision;
        }

        Active = model;
        _deployedVersions.Add(model.Version);
        _history.Add(new ModelEvent(ModelEventKind.Deployed, model.Version, _clock.Now,
            [$"vượt {decision.Checks.Count} điều kiện"]));

        return decision;
    }

    /// <summary>
    /// Quay về phiên bản ĐÃ TỪNG CHẠY gần nhất trước phiên bản hiện tại.
    ///
    /// Trả về null nếu không có gì để quay về — và khi đó model đang chạy giữ
    /// nguyên. Quay lui về hư vô không an toàn hơn giữ nguyên: nó chỉ làm hệ
    /// thống mất luôn bộ phân loại.
    /// </summary>
    public ModelVersion? Rollback(string reason = "quay lui thủ công")
    {
        if (Active is null || _deployedVersions.Count < 2) return null;

        _deployedVersions.RemoveAt(_deployedVersions.Count - 1);

        int target = _deployedVersions[^1];
        var previous = Get(target)!;

        Active = previous;
        _history.Add(new ModelEvent(ModelEventKind.RolledBack, previous.Version, _clock.Now, [reason]));

        return previous;
    }

    /// <summary>Dựng lại bộ phân loại đang chạy. Ném nếu chưa có model nào được duyệt.</summary>
    public IntentClassifier RestoreActive() =>
        Active?.Restore()
        ?? throw new InvalidOperationException("chưa có model nào vượt được cửa đánh giá");
}
