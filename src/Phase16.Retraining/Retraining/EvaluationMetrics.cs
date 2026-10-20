using System.Globalization;
using Ai.Phase16.Agent;

namespace Ai.Phase16.Retraining;

/// <summary>Số liệu của MỘT nhãn. Precision và recall tách riêng vì chúng hỏng theo hai kiểu khác nhau.</summary>
public sealed record LabelMetrics(Intent Label, int Support, int TruePositive, int FalsePositive, int FalseNegative)
{
    /// <summary>Trong những lần model nói nhãn này, bao nhiêu phần là đúng.</summary>
    public double Precision => TruePositive + FalsePositive == 0
        ? 0
        : (double)TruePositive / (TruePositive + FalsePositive);

    /// <summary>Trong những lần đáng lẽ phải là nhãn này, model bắt được bao nhiêu phần.</summary>
    public double Recall => TruePositive + FalseNegative == 0
        ? 0
        : (double)TruePositive / (TruePositive + FalseNegative);

    public double F1 => Precision + Recall == 0 ? 0 : 2 * Precision * Recall / (Precision + Recall);

    /// <summary>Model chưa bao giờ nói nhãn này — dấu hiệu nó đã bỏ hẳn một ý định.</summary>
    public bool NeverPredicted => TruePositive + FalsePositive == 0;

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "{0,-16} P {1:F2}  R {2:F2}  F1 {3:F2}  ({4} mẫu)", Label, Precision, Recall, F1, Support);
}

/// <summary>
/// KẾT QUẢ ĐÁNH GIÁ một model trên một tập ví dụ có nhãn.
///
/// VÌ SAO KHÔNG CHỈ DÙNG ACCURACY?
///
/// Vì accuracy là trung bình, và trung bình che mất đúng thứ đáng sợ nhất:
/// model bỏ hẳn một ý định. Bảy ý định, tập dữ liệu lệch, một model không bao
/// giờ trả lời <c>SearchInternet</c> vẫn có thể đạt accuracy cao — nó chỉ sai
/// ở một nhóm nhỏ. Nhưng với người dùng thì đó không phải "sai 8%", mà là
/// "chức năng tra Internet đã chết".
///
/// <see cref="MacroF1"/> tính F1 cho từng nhãn rồi lấy trung bình KHÔNG theo
/// trọng số mẫu, nên một nhãn chết kéo tụt điểm đúng như nó đáng bị kéo. Đó là
/// con số mà <see cref="DeploymentGate"/> dùng để so hai model.
///
/// <see cref="Predictions"/> được giữ lại (theo danh tính chuẩn hoá của ví dụ)
/// vì so hai model cần biết CHÍNH XÁC ví dụ nào đổi kết quả — chênh lệch
/// accuracy bằng 0 vẫn có thể là "sửa được 3 câu, làm hỏng 3 câu khác".
/// </summary>
public sealed record EvaluationMetrics
{
    public required int Total { get; init; }

    public required int Correct { get; init; }

    public double Accuracy => Total == 0 ? 0 : (double)Correct / Total;

    public required IReadOnlyList<LabelMetrics> PerLabel { get; init; }

    /// <summary>Trung bình F1 của các nhãn CÓ MẶT trong tập đánh giá, không đánh trọng số theo số mẫu.</summary>
    public double MacroF1 => PerLabel.Count == 0 ? 0 : PerLabel.Average(l => l.F1);

    /// <summary>Cross entropy trung bình — nhạy hơn accuracy, thấy được model "đúng nhưng lưỡng lự".</summary>
    public required double MeanLoss { get; init; }

    /// <summary>Danh tính ví dụ -> nhãn model đoán. Dùng để so từng ví dụ giữa hai model.</summary>
    public required IReadOnlyDictionary<string, Intent> Predictions { get; init; }

    /// <summary>Danh tính những ví dụ model đoán sai.</summary>
    public required IReadOnlyList<string> Wrong { get; init; }

    /// <summary>
    /// Ví dụ mà KHÔNG MỘT TỪ NÀO của nó từng xuất hiện trong phần dữ liệu TRAIN.
    ///
    /// Con số này tách "model dở" khỏi "dữ liệu thiếu" — với một tập nhỏ thì đó
    /// là hai chuyện hoàn toàn khác nhau.
    ///
    /// Lưu ý chỗ dễ nhầm: từ vẫn CÓ trong từ điển (từ điển dựng từ cả tập dữ
    /// liệu, xem <see cref="ModelTrainer.Train"/>), nên vector đầu vào không
    /// rỗng. Nhưng những chiều đó chưa bao giờ được cập nhật gradient, nên
    /// trọng số của chúng vẫn y như lúc khởi tạo ngẫu nhiên. Model không "chưa
    /// biết" câu đó — nó đang trả lời bằng nhiễu.
    ///
    /// Đó là hệ quả trực tiếp của việc chỉ có 70 ví dụ: giữ lại 20% để đánh
    /// giá là lấy đi ví dụ DUY NHẤT chứa những từ ấy.
    /// </summary>
    public required IReadOnlyList<string> WithoutTrainedWords { get; init; }

    /// <summary>Trong những câu sai, bao nhiêu câu model vốn không có cơ sở nào để đoán đúng.</summary>
    public int WithoutTrainedWordsWrong =>
        Wrong.Intersect(WithoutTrainedWords, StringComparer.Ordinal).Count();

    /// <summary>
    /// Accuracy chỉ trên những câu có ít nhất một từ đã được train.
    ///
    /// Đây KHÔNG phải con số để báo cáo thay cho <see cref="Accuracy"/> — bỏ
    /// bớt câu khó rồi khoe điểm cao là tự lừa mình. Nó để trả lời đúng một câu
    /// hỏi: chênh lệch giữa hai con số này là phần do THIẾU DỮ LIỆU, không phải
    /// do model.
    /// </summary>
    public double AccuracyOnTrainedVocabulary
    {
        get
        {
            int measurable = Total - WithoutTrainedWords.Count;
            if (measurable <= 0) return 0;

            int correctAmongMeasurable = Correct - (WithoutTrainedWords.Count - WithoutTrainedWordsWrong);
            return (double)correctAmongMeasurable / measurable;
        }
    }

    /// <summary>Số nhãn khác nhau mà model thực sự dùng tới.</summary>
    public int DistinctPredictedLabels => Predictions.Values.Distinct().Count();

    /// <summary>Nhãn có trong dữ liệu nhưng model không bao giờ đoán tới.</summary>
    public IReadOnlyList<Intent> DeadLabels =>
        [.. PerLabel.Where(l => l.NeverPredicted).Select(l => l.Label)];

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "accuracy {0:P1} ({1}/{2}), macro F1 {3:F3}, loss {4:F4}",
            Accuracy, Correct, Total, MacroF1, MeanLoss);
}
