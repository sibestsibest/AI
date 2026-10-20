using System.Globalization;
using Ai.Phase14.Grounding;

namespace Ai.Phase14.Verification;

public sealed record ConfidenceAssessment(double Confidence, string Explanation);

public interface IConfidenceEstimator
{
    ConfidenceAssessment Estimate(EvidenceSet evidence, IReadOnlyList<Claim> claims,
        IReadOnlyList<Contradiction> contradictions);
}

/// <summary>
/// ĐỘ TỰ TIN — ước lượng thô, và phải thô một cách trung thực.
///
/// CÔNG THỨC (cùng dạng tổ hợp có trọng số như Phase 10/12/13):
///
///     base = 0.35·reliability + 0.25·support + 0.20·agreement + 0.20·relevance
///
/// rồi nhân các HỆ SỐ PHẠT:
///
///     × (1 − 0.6·severity)   nếu có mâu thuẫn
///     × 0.65                 nếu KHÔNG có nguồn nào đã kiểm chứng
///     × 0.80                 nếu câu trả lời dựa chủ yếu vào suy ra, không phải sự thật
///
/// VÌ SAO NHÂN CHỨ KHÔNG TRỪ? Vì phạt phải theo TỈ LỆ. Trừ đi một lượng cố
/// định thì một câu trả lời vốn yếu (0.2) bị trừ xuống âm, còn một câu rất
/// mạnh (0.95) gần như không bị ảnh hưởng — ngược hẳn điều ta muốn. Nhân thì
/// mâu thuẫn làm giảm một nửa dù điểm gốc là bao nhiêu.
///
/// PHẢI NÓI THẲNG con số này là gì: nó KHÔNG phải xác suất câu trả lời đúng.
/// Không có cách nào tính được xác suất đó mà không biết sự thật. Nó là một
/// điểm tổng hợp cho biết CĂN CỨ mạnh đến đâu — nguồn có đáng tin không, có
/// nhiều kênh xác nhận không, có mâu thuẫn không. Căn cứ mạnh mà vẫn sai là
/// chuyện hoàn toàn có thể. Vì vậy nó được in ra kèm dải chữ
/// (<see cref="ConfidenceBand"/>) thay vì chỉ một số thập phân trông như đo
/// được chính xác.
/// </summary>
public sealed class ConfidenceEstimator : IConfidenceEstimator
{
    /// <summary>Dưới ngưỡng này thì agent nói "không đủ căn cứ" thay vì trả lời.</summary>
    public double InsufficientThreshold { get; init; } = 0.20;

    /// <summary>Dưới ngưỡng này thì trả lời nhưng phải nói rõ là chưa chắc.</summary>
    public double UncertainThreshold { get; init; } = 0.45;

    /// <summary>Mâu thuẫn từ mức này trở lên thì không được chọn bên nào.</summary>
    public double ConflictThreshold { get; init; } = 0.35;

    public ConfidenceAssessment Estimate(EvidenceSet evidence, IReadOnlyList<Claim> claims,
        IReadOnlyList<Contradiction> contradictions)
    {
        ArgumentNullException.ThrowIfNull(evidence);
        ArgumentNullException.ThrowIfNull(claims);
        ArgumentNullException.ThrowIfNull(contradictions);

        if (!evidence.HasEvidence || claims.Count == 0)
        {
            return new ConfidenceAssessment(0, "không có căn cứ nào");
        }

        double reliability = evidence.BestReliability;
        double support = claims.Average(c => c.Support);
        double relevance = evidence.Items.Max(e => e.Relevance);

        // Đồng thuận tính theo số KÊNH độc lập, bão hoà: 1 -> 0, 2 -> 0.5, 3 -> 0.67
        int channels = evidence.DistinctChannelCount;
        double agreement = 1.0 - 1.0 / Math.Max(1, channels);

        double score = 0.35 * reliability + 0.25 * support + 0.20 * agreement + 0.20 * relevance;

        var notes = new List<string>
        {
            string.Format(CultureInfo.InvariantCulture,
                "tin cậy {0:F2}, đỡ {1:F2}, {2} kênh, khớp {3:F2}", reliability, support, channels, relevance),
        };

        // --- Phạt 1: mâu thuẫn ---
        if (contradictions.Count > 0)
        {
            double severity = contradictions.Max(c => c.Severity);
            score *= 1.0 - 0.6 * severity;
            notes.Add($"trừ vì {contradictions.Count} mâu thuẫn (nặng nhất {severity:F2})");
        }

        // --- Phạt 2: không có nguồn đã kiểm ---
        if (!evidence.HasVerifiedEvidence)
        {
            score *= 0.65;
            notes.Add("trừ vì không có nguồn đã kiểm chứng");
        }

        // --- Phạt 3: chủ yếu là suy ra, không phải sự thật trực tiếp ---
        int inferences = claims.Count(c => c.Kind == ClaimKind.Inference);
        if (inferences > claims.Count / 2)
        {
            score *= 0.80;
            notes.Add($"trừ vì {inferences}/{claims.Count} khẳng định là suy ra, không phải sự thật trực tiếp");
        }

        return new ConfidenceAssessment(Math.Clamp(score, 0, 1), string.Join("; ", notes));
    }

    /// <summary>
    /// Chốt kết luận từ độ tự tin và mâu thuẫn.
    ///
    /// Thứ tự xét KHÔNG đảo được: mâu thuẫn phải được xét TRƯỚC độ tự tin. Hai
    /// nguồn đã kiểm nói ngược nhau thì cả hai đều đáng tin, nên điểm vẫn có
    /// thể cao — mà đó chính là lúc tuyệt đối không được chọn bừa một phía.
    /// </summary>
    public AnswerVerdict Decide(double confidence, IReadOnlyList<Contradiction> contradictions, bool hasEvidence)
    {
        ArgumentNullException.ThrowIfNull(contradictions);

        if (!hasEvidence) return AnswerVerdict.Insufficient;

        if (contradictions.Any(c => c.Severity >= ConflictThreshold)) return AnswerVerdict.Conflicted;

        if (confidence < InsufficientThreshold) return AnswerVerdict.Insufficient;

        return confidence < UncertainThreshold ? AnswerVerdict.Uncertain : AnswerVerdict.Answered;
    }
}
