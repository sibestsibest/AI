using System.Globalization;
using Ai.Phase16.Agent;

namespace Ai.Phase16.Retraining;

/// <summary>
/// SO HAI MODEL TRÊN CÙNG MỘT TẬP ĐÁNH GIÁ.
///
/// ĐIỀU QUAN TRỌNG NHẤT: chênh lệch accuracy KHÔNG đủ để kết luận.
///
/// Một model sửa được 3 câu và làm hỏng 3 câu khác có chênh lệch accuracy bằng
/// 0 — trông như "không đổi gì". Nhưng với người dùng thì ba chức năng đang
/// chạy tốt vừa hỏng, đổi lấy ba chức năng khác vừa chạy được. Đó là một thay
/// đổi rất lớn, và nó bị con số trung bình triệt tiêu mất.
///
/// Nên ở đây so TỪNG VÍ DỤ:
///
///     Regressions  = model cũ đúng, model mới sai   -> thứ phải đếm
///     Improvements = model cũ sai, model mới đúng   -> thứ muốn có
///
/// <see cref="DeploymentGate"/> nhìn vào Regressions, không nhìn vào hiệu số.
/// </summary>
public sealed record ModelComparison
{
    public required double AccuracyDelta { get; init; }

    public required double MacroF1Delta { get; init; }

    public required double LossDelta { get; init; }

    /// <summary>Ví dụ model CŨ làm đúng mà model MỚI làm sai. Đây là thứ đáng sợ.</summary>
    public required IReadOnlyList<string> Regressions { get; init; }

    /// <summary>Ví dụ model CŨ làm sai mà model MỚI làm đúng.</summary>
    public required IReadOnlyList<string> Improvements { get; init; }

    /// <summary>Nhãn model cũ còn dùng mà model mới không bao giờ đoán tới nữa.</summary>
    public required IReadOnlyList<Intent> LabelsLost { get; init; }

    public bool HasRegressions => Regressions.Count > 0;

    public bool LostALabel => LabelsLost.Count > 0;

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "accuracy {0:+0.###;-0.###;0}, macro F1 {1:+0.###;-0.###;0}, sửa được {2}, làm hỏng {3}",
            AccuracyDelta, MacroF1Delta, Improvements.Count, Regressions.Count);

    /// <summary>
    /// So hai bộ số liệu. Chỉ xét những ví dụ CẢ HAI model cùng được chấm —
    /// ví dụ chỉ có ở một bên thì không so được, và lặng lẽ coi nó là tiến bộ
    /// hay thụt lùi đều là bịa.
    /// </summary>
    public static ModelComparison Compare(EvaluationMetrics baseline, EvaluationMetrics candidate)
    {
        ArgumentNullException.ThrowIfNull(baseline);
        ArgumentNullException.ThrowIfNull(candidate);

        var regressions = new List<string>();
        var improvements = new List<string>();

        var baselineWrong = baseline.Wrong.ToHashSet(StringComparer.Ordinal);
        var candidateWrong = candidate.Wrong.ToHashSet(StringComparer.Ordinal);

        foreach (string key in baseline.Predictions.Keys.Where(candidate.Predictions.ContainsKey))
        {
            bool wasRight = !baselineWrong.Contains(key);
            bool isRight = !candidateWrong.Contains(key);

            if (wasRight && !isRight) regressions.Add(key);
            if (!wasRight && isRight) improvements.Add(key);
        }

        var baselineLabels = baseline.Predictions.Values.Distinct().ToHashSet();
        var candidateLabels = candidate.Predictions.Values.Distinct().ToHashSet();

        return new ModelComparison
        {
            AccuracyDelta = candidate.Accuracy - baseline.Accuracy,
            MacroF1Delta = candidate.MacroF1 - baseline.MacroF1,
            LossDelta = candidate.MeanLoss - baseline.MeanLoss,
            Regressions = [.. regressions.Order(StringComparer.Ordinal)],
            Improvements = [.. improvements.Order(StringComparer.Ordinal)],
            LabelsLost = [.. baselineLabels.Where(l => !candidateLabels.Contains(l)).OrderBy(l => (int)l)],
        };
    }
}
