using System.Globalization;

namespace Ai.Phase16.Retraining;

/// <summary>Một điều kiện đã được kiểm, kèm kết quả và con số thật.</summary>
public sealed record GateCheck(string Name, bool Passed, string Detail)
{
    public override string ToString() => $"{(Passed ? "ĐẠT " : "KHÔNG")} — {Name}: {Detail}";
}

/// <summary>Kết luận của cửa triển khai.</summary>
public sealed record DeploymentDecision(bool Approved, IReadOnlyList<GateCheck> Checks)
{
    public IReadOnlyList<string> Blockers =>
        [.. Checks.Where(c => !c.Passed).Select(c => $"{c.Name}: {c.Detail}")];

    public override string ToString() =>
        Approved
            ? $"TRIỂN KHAI ({Checks.Count} điều kiện đều đạt)"
            : $"TỪ CHỐI — {string.Join("; ", Blockers)}";
}

/// <summary>
/// CỬA TRIỂN KHAI — chốt chặn của Phase 16.
///
/// Đây là chỗ thực thi luật lớn nhất của phase này: MODEL MỚI KHÔNG TỰ ĐỘNG
/// THAY MODEL ĐANG CHẠY. Nó phải vượt được từng điều kiện dưới đây, và mỗi
/// điều kiện trả về một con số thật kèm lý do — nên khi bị từ chối thì đọc
/// được ngay là vì sao, thay vì chỉ thấy "đánh giá không đạt".
///
/// SÁU ĐIỀU KIỆN:
///
///   1. Có dữ liệu để đánh giá        không đo được thì không triển khai
///   2. Mọi nhãn đều có mặt khi train  model không học được nhãn nó chưa thấy
///   3. Accuracy >= sàn tuyệt đối      chặn model hỏng hẳn
///   4. Không thụt lùi so với model cũ macro F1, không phải accuracy
///   5. KHÔNG có ví dụ nào từ đúng thành sai
///   6. Bộ vàng đúng 100%
///
/// VÌ SAO ĐIỀU KIỆN 4 DÙNG MACRO F1 CHỨ KHÔNG DÙNG ACCURACY?
///
/// Vì accuracy đánh trọng số theo số mẫu, nên một model bỏ hẳn ý định hiếm
/// nhất vẫn có thể có accuracy cao hơn model cũ. Macro F1 tính đều cho mọi
/// nhãn, nên một ý định chết kéo điểm xuống đúng như nó đáng bị kéo.
///
/// VÌ SAO ĐIỀU KIỆN 5 KHÔNG CHO ĐÁNH ĐỔI?
///
/// Vì "sửa 5 câu, hỏng 3 câu" nghe như lãi, nhưng ba câu hỏng kia là ba thứ
/// ĐANG CHẠY ĐƯỢC mà người dùng đã tin. Mất niềm tin vào thứ từng chạy tốn
/// hơn nhiều so với việc chưa có thêm tính năng mới. Muốn nhận đánh đổi đó thì
/// phải là một người quyết định, một cách tường minh — nới
/// <see cref="MaxRegressions"/> và ghi lý do vào lịch sử, chứ không phải để hệ
/// thống tự cân.
/// </summary>
public sealed class DeploymentGate
{
    /// <summary>
    /// SÀN TUYỆT ĐỐI — mức "chưa hỏng hẳn", KHÔNG phải mức chất lượng mong muốn.
    ///
    /// Con số 0.50 được chọn từ số đo thật, không phải từ cảm giác. Model hiện
    /// tại đạt 58.8% trên tập đánh giá 17 câu của bộ dữ liệu 70 ví dụ — trong
    /// khi Phase 11 vẫn báo "100% accuracy", vì Phase 11 đo TRÊN CHÍNH TẬP
    /// TRAIN. Đặt sàn 0.85 thì không model nào của dự án này qua được, và một
    /// cửa luôn từ chối thì chẳng bảo vệ được gì — người ta sẽ vô hiệu hoá nó.
    ///
    /// Sàn này chỉ chặn thảm hoạ (đoán bừa là 1/7 ≈ 14%). Hai chốt chất lượng
    /// thật nằm ở chỗ khác: bộ vàng phải đúng 100%, và model mới không được
    /// làm hỏng thứ model cũ đang làm đúng.
    /// </summary>
    public double MinimumAccuracy { get; init; } = 0.50;

    /// <summary>Bộ vàng phải đúng bao nhiêu phần. Mặc định: tất cả.</summary>
    public double MinimumGoldenAccuracy { get; init; } = 1.0;

    /// <summary>Số ví dụ được phép từ đúng thành sai. Mặc định 0 — phải có người quyết mới nới.</summary>
    public int MaxRegressions { get; init; }

    /// <summary>Macro F1 được phép kém model cũ bao nhiêu. Mặc định 0.</summary>
    public double MacroF1Tolerance { get; init; }

    public DeploymentDecision Evaluate(
        EvaluationMetrics candidate,
        EvaluationMetrics? goldenMetrics,
        ModelComparison? comparison,
        DataSplit split)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        ArgumentNullException.ThrowIfNull(split);

        var checks = new List<GateCheck>
        {
            // --- 1 ---
            new("có dữ liệu đánh giá", candidate.Total > 0,
                candidate.Total > 0
                    ? $"{candidate.Total} ví dụ"
                    : "tập đánh giá RỖNG — không đo được thì không triển khai"),

            // --- 2 ---
            new("mọi nhãn đều có mặt khi train", split.LabelsMissingFromTraining.Count == 0,
                split.LabelsMissingFromTraining.Count == 0
                    ? "đủ cả 7 ý định"
                    : $"thiếu {string.Join(", ", split.LabelsMissingFromTraining)} — model không học được nhãn nó chưa thấy"),

            // --- 3 ---
            new("accuracy đạt sàn", candidate.Accuracy >= MinimumAccuracy,
                string.Format(CultureInfo.InvariantCulture, "{0:P1} so với sàn {1:P0}",
                    candidate.Accuracy, MinimumAccuracy)),
        };

        // --- 4 và 5: chỉ so được khi CÓ model cũ ---
        if (comparison is not null)
        {
            checks.Add(new("không thụt lùi so với model đang chạy",
                comparison.MacroF1Delta >= -MacroF1Tolerance,
                string.Format(CultureInfo.InvariantCulture, "macro F1 {0:+0.###;-0.###;0}", comparison.MacroF1Delta)));

            checks.Add(new("không làm hỏng thứ đang chạy được",
                comparison.Regressions.Count <= MaxRegressions,
                comparison.Regressions.Count == 0
                    ? "không có ví dụ nào từ đúng thành sai"
                    : $"{comparison.Regressions.Count} ví dụ từ đúng thành sai (mức cho phép {MaxRegressions}): " +
                      string.Join("; ", comparison.Regressions.Take(3))));
        }
        else
        {
            checks.Add(new("so với model đang chạy", true, "chưa có model nào đang chạy — đây là model đầu tiên"));
        }

        // --- 6 ---
        if (goldenMetrics is not null)
        {
            checks.Add(new("bộ vàng", goldenMetrics.Accuracy >= MinimumGoldenAccuracy,
                string.Format(CultureInfo.InvariantCulture, "{0}/{1} câu đúng{2}",
                    goldenMetrics.Correct, goldenMetrics.Total,
                    goldenMetrics.Wrong.Count == 0 ? "" : " — hỏng: " + string.Join("; ", goldenMetrics.Wrong))));
        }

        return new DeploymentDecision(checks.All(c => c.Passed), checks);
    }
}
