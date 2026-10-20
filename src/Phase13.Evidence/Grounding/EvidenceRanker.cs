using System.Globalization;
using Ai.Phase13.Internet;
using Ai.Phase13.Memory;

namespace Ai.Phase13.Grounding;

public interface IEvidenceRanker
{
    IReadOnlyList<Evidence> Rank(string question, IEnumerable<Evidence> evidence, DateTimeOffset now);
}

/// <summary>
/// CHẤM ĐIỂM EVIDENCE.
///
/// Cùng dạng tổ hợp có trọng số đã dùng ở <see cref="Ranker"/> (Phase 10) và
/// <see cref="SourceRanker"/> (Phase 12) — giữ một lối tính xuyên suốt cả
/// project thì đọc chỗ nào cũng hiểu ngay:
///
///     score = 0.45·relevance + 0.30·reliability + 0.15·agreement + 0.10·freshness
///
/// VÌ SAO relevance NẶNG NHẤT (0.45)? Vì một nguồn cực uy tín nói về chuyện
/// khác thì vô dụng cho câu hỏi đang hỏi. Không liên quan thì không dùng
/// được, bất kể đáng tin đến đâu.
///
/// VÌ SAO reliability CHỈ 0.30, không cao hơn? Vì nếu để nó quá nặng thì kênh
/// knowledge luôn thắng tuyệt đối, và agent sẽ bỏ qua thông tin mới đúng lúc
/// nó cần nhất — những câu hỏi mà kho nội bộ chưa có gì. Reliability đủ nặng
/// để nguồn đã kiểm thắng khi hai bên nói cùng một điều, nhưng không nặng đến
/// mức bịt luôn kênh Internet.
///
/// VÌ SAO freshness NHẸ NHẤT (0.10)? Vì mới không có nghĩa là đúng. Một trang
/// đăng hôm nay vẫn có thể sai hơn một tài liệu ba năm trước. Độ tươi chỉ
/// đáng kể với thông tin thực sự thay đổi theo thời gian.
///
/// RELIABILITY ĐƯỢC TÍNH LẠI, KHÔNG LẤY NGUYÊN của nguồn:
///
///     reliability = BaseReliability × uy tín tên miền × suy giảm theo tuổi × trừ phạt tiêm chỉ thị
///
/// Đây là chỗ Phase 13 nối vào Phase 12: bảng uy tín tên miền của
/// <see cref="SourceRanker"/> được dùng lại nguyên vẹn, không viết lại.
/// </summary>
public sealed class EvidenceRanker : IEvidenceRanker
{
    private readonly SourceRanker _domainAuthority;

    public EvidenceRanker(SourceRanker? domainAuthority = null) =>
        _domainAuthority = domainAuthority ?? new SourceRanker();

    /// <summary>
    /// Nửa chu kỳ suy giảm độ tươi của nội dung web, tính bằng giờ.
    ///
    /// Dùng lại đúng công thức của Phase 10: <c>0.5 ^ (giờ / halfLife)</c>.
    /// 72 giờ là mức thận trọng vừa phải — sau 3 ngày một trang web chỉ còn
    /// được tính nửa trọng số độ tươi.
    /// </summary>
    public double WebFreshnessHalfLifeHours { get; init; } = 72;

    /// <summary>Kiến thức nội bộ không phai — nó chỉ đổi khi có người sửa.</summary>
    public double KnowledgeFreshness { get; init; } = 1.0;

    public IReadOnlyList<Evidence> Rank(string question, IEnumerable<Evidence> evidence, DateTimeOffset now)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(evidence);

        var words = Keywords.Extract(question);

        return [.. evidence
            .Select(e => Score(e, words, now))
            .OrderByDescending(e => e.Score)
            .ThenBy(e => e.Kind)                                   // bằng điểm -> kênh tin hơn trước
            .ThenBy(e => e.Id, StringComparer.Ordinal)];           // rồi mới tất định theo id
    }

    private Evidence Score(Evidence item, IReadOnlySet<string> questionWords, DateTimeOffset now)
    {
        // --- 1. relevance ---
        double relevance = Keywords.Relevance(questionWords, item.Keywords());

        // Chủ đề/tên nguồn khớp thì cộng thêm — cùng lý do như mọi chỗ khác:
        // tên nói nó NÓI VỀ cái gì, nội dung chỉ nói nó CÓ NHẮC tới cái gì.
        double nameMatch = Keywords.Relevance(questionWords, Keywords.Extract(item.Source.Name));
        relevance = Math.Min(1.0, relevance + 0.3 * nameMatch);

        // --- 2. reliability ---
        double freshness = Freshness(item, now);
        double reliability = item.Source.BaseReliability;

        if (item.Kind == SourceKind.Internet && item.Url is not null)
        {
            // Nhân với uy tín tên miền: cùng là web nhưng .gov khác blog cá nhân.
            reliability *= _domainAuthority.AuthorityOf(item.Url);
        }

        // Nội dung cũ thì bớt tin — nhưng chỉ bớt một phần, không xoá sạch.
        reliability *= 0.7 + 0.3 * freshness;

        if (item.LooksLikeInjection) reliability *= 0.3;

        reliability = Math.Clamp(reliability, 0, 1);

        // --- 3. agreement: 1 kênh -> 0, 2 -> 0.5, 3 -> 0.67 (bão hoà) ---
        double agreement = 1.0 - 1.0 / Math.Max(1, item.Agreement);

        double score = 0.45 * relevance + 0.30 * reliability + 0.15 * agreement + 0.10 * freshness;

        string explanation = string.Format(CultureInfo.InvariantCulture,
            "{0}: khớp {1:P0}, tin {2:F2}, {3} kênh, tươi {4:F2}",
            item.Kind, relevance, reliability, item.Agreement, freshness);

        if (item.LooksLikeInjection) explanation += ", TRỪ NẶNG vì có dấu hiệu tiêm chỉ thị";

        return item with
        {
            Relevance = relevance,
            Reliability = reliability,
            Score = Math.Clamp(score, 0, 1),
            Explanation = explanation,
        };
    }

    /// <summary>Độ tươi 0..1. Kiến thức không phai; web và ký ức thì phai theo nửa chu kỳ.</summary>
    public double Freshness(Evidence item, DateTimeOffset now)
    {
        ArgumentNullException.ThrowIfNull(item);

        if (item.Kind == SourceKind.Knowledge) return KnowledgeFreshness;
        if (item.RetrievedAt is null) return 0.5;          // không biết tuổi -> trung tính

        double hours = (now - item.RetrievedAt.Value).TotalHours;
        if (hours <= 0) return 1.0;

        return Math.Pow(0.5, hours / WebFreshnessHalfLifeHours);
    }
}
