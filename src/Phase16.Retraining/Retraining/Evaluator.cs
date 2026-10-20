using Ai.Phase16.Agent;
using Ai.Phase16.Learning;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Retraining;

/// <summary>Một ví dụ dùng để đánh giá: câu, nhãn đúng, và danh tính chuẩn hoá để so giữa hai model.</summary>
public sealed record EvaluationExample(string Input, Intent Label)
{
    public string Key => $"{Label}|{Keywords.Canonical(Input)}";

    public static EvaluationExample From(DatasetExample example)
    {
        ArgumentNullException.ThrowIfNull(example);

        return new EvaluationExample(example.Input, example.Label);
    }
}

/// <summary>
/// BỘ ĐÁNH GIÁ — chấm một model trên một tập ví dụ có nhãn.
///
/// Tách hẳn khỏi khâu huấn luyện, và điều đó có lý do: bộ phận quyết định
/// "model này có đủ tốt không" không được là bộ phận vừa tạo ra nó. Trong
/// <see cref="ModelTrainer"/> có đủ thông tin để tự chấm mình — và cũng đủ
/// cám dỗ để chấm trên chính dữ liệu vừa học.
///
/// Bộ đánh giá ở đây không biết gì về quá trình train. Nó nhận một
/// <see cref="IntentClassifier"/> và một danh sách ví dụ, rồi trả số liệu.
/// Nhờ vậy nó chấm được cả model đang chạy lẫn model ứng viên bằng CÙNG một
/// cách, trên CÙNG một tập — điều kiện tối thiểu để hai con số so được với nhau.
/// </summary>
public static class Evaluator
{
    /// <param name="trainedWords">
    /// Từ khoá đã thực sự xuất hiện trong phần dữ liệu train. Truyền vào thì
    /// báo cáo tách được phần sai do THIẾU DỮ LIỆU ra khỏi phần sai do model.
    /// Bỏ trống cũng chạy, chỉ là mất phần chẩn đoán đó.
    /// </param>
    public static EvaluationMetrics Evaluate(IntentClassifier classifier,
        IReadOnlyList<EvaluationExample> examples, IReadOnlySet<string>? trainedWords = null)
    {
        ArgumentNullException.ThrowIfNull(classifier);
        ArgumentNullException.ThrowIfNull(examples);

        var predictions = new Dictionary<string, Intent>(StringComparer.Ordinal);
        var wrong = new List<string>();
        var withoutTrainedWords = new List<string>();

        var truePositive = new Dictionary<Intent, int>();
        var falsePositive = new Dictionary<Intent, int>();
        var falseNegative = new Dictionary<Intent, int>();
        var support = new Dictionary<Intent, int>();

        int correct = 0;
        double lossSum = 0;

        foreach (var example in examples)
        {
            var prediction = classifier.Classify(example.Input);

            predictions[example.Key] = prediction.Intent;
            support[example.Label] = support.GetValueOrDefault(example.Label) + 1;

            if (trainedWords is not null && !Keywords.Extract(example.Input).Any(trainedWords.Contains))
            {
                withoutTrainedWords.Add(example.Key);
            }

            // Cross entropy của nhãn đúng. Kẹp sàn để log(0) không thành -vô cùng
            // khi model gán xác suất 0 cho nhãn đúng — một trường hợp hoàn toàn
            // có thật, và nếu để nó làm loss thành vô cùng thì mọi so sánh loss
            // sau đó đều vô nghĩa.
            double probability = Math.Max(prediction.Probabilities[(int)example.Label], 1e-12);
            lossSum += -Math.Log(probability);

            if (prediction.Intent == example.Label)
            {
                correct++;
                truePositive[example.Label] = truePositive.GetValueOrDefault(example.Label) + 1;
            }
            else
            {
                wrong.Add(example.Key);
                falsePositive[prediction.Intent] = falsePositive.GetValueOrDefault(prediction.Intent) + 1;
                falseNegative[example.Label] = falseNegative.GetValueOrDefault(example.Label) + 1;
            }
        }

        // Chỉ tính cho những nhãn CÓ MẶT trong tập đánh giá. Nhãn không có mẫu
        // nào thì F1 của nó là 0/0 — đưa vào trung bình sẽ kéo macro F1 xuống
        // vì một lý do không liên quan gì tới chất lượng model.
        var labels = support.Keys.OrderBy(l => (int)l).ToList();

        var perLabel = labels
            .Select(label => new LabelMetrics(
                label,
                support[label],
                truePositive.GetValueOrDefault(label),
                falsePositive.GetValueOrDefault(label),
                falseNegative.GetValueOrDefault(label)))
            .ToList();

        return new EvaluationMetrics
        {
            Total = examples.Count,
            Correct = correct,
            PerLabel = perLabel,
            MeanLoss = examples.Count == 0 ? 0 : lossSum / examples.Count,
            Predictions = predictions,
            Wrong = wrong,
            WithoutTrainedWords = withoutTrainedWords,
        };
    }

    public static EvaluationMetrics Evaluate(IntentClassifier classifier,
        IEnumerable<DatasetExample> examples, IReadOnlySet<string>? trainedWords = null)
    {
        ArgumentNullException.ThrowIfNull(examples);

        return Evaluate(classifier, [.. examples.Select(EvaluationExample.From)], trainedWords);
    }

    /// <summary>Từ khoá đã thực sự xuất hiện trong một tập ví dụ.</summary>
    public static IReadOnlySet<string> WordsIn(IEnumerable<DatasetExample> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var example in examples)
        {
            words.UnionWith(Keywords.Extract(example.Input));
        }

        return words;
    }
}
