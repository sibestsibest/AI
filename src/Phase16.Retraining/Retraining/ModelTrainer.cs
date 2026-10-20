using Ai.Phase16.Agent;
using Ai.Phase16.Embeddings;
using Ai.Phase16.Learning;
using Ai.Phase16.Memory;
using Ai.Phase16.Training;

namespace Ai.Phase16.Retraining;

/// <summary>
/// HUẤN LUYỆN TẤT ĐỊNH — cùng dữ liệu, cùng cấu hình thì ra cùng trọng số.
///
/// Dùng lại nguyên bộ máy Phase 2–6 (bag-of-words -> tanh -> softmax + cross
/// entropy) và <c>IntentClassifier.Train</c> của Phase 11. Phần Phase 16 thêm
/// vào chỉ là những gì cần để kết quả TÁI LẬP ĐƯỢC — và hoá ra đó là chỗ có
/// nhiều bẫy nhất.
///
/// BA NGUỒN BẤT ĐỊNH ĐÃ BỊ BỊT:
///
///   1. Trọng số khởi tạo  -> <c>new Random(config.Seed)</c>, seed nằm trong
///                            cấu hình và được lưu cùng model.
///
///   2. Thứ tự từ điển     -> <see cref="BuildVocabulary"/> SẮP XẾP từ theo
///                            thứ tự Ordinal. Bản gốc của Phase 11 duyệt
///                            HashSet để dựng từ điển, mà thứ tự duyệt HashSet
///                            là chi tiết cài đặt. Chỉ số của từ quyết định vị
///                            trí trọng số, nên từ điển xếp khác là model khác.
///
///   3. Thứ tự ví dụ       -> <c>DatasetVersion.Create</c> của Phase 15 đã sắp
///                            ví dụ về thứ tự chuẩn. Cần, vì gradient được
///                            cộng dồn theo thứ tự và phép cộng số thực dấu
///                            phẩy động KHÔNG có tính kết hợp: (a+b)+c ≠
///                            a+(b+c) ở những chữ số cuối. Chênh lệch cỡ 1e-16
///                            ấy, qua 3000 epoch, đủ để hai model khác nhau
///                            thấy rõ.
/// </summary>
public static class ModelTrainer
{
    /// <summary>
    /// Dựng từ điển TẤT ĐỊNH từ một tập dữ liệu: gom mọi từ khoá rồi sắp Ordinal.
    ///
    /// Sắp bằng <see cref="StringComparer.Ordinal"/> chứ không theo văn hoá —
    /// so sánh theo văn hoá xếp "cộng" và "cong" khác nhau tuỳ locale, và khi
    /// đó cùng một tập dữ liệu lại ra hai model khác nhau ở hai máy.
    /// </summary>
    public static Vocabulary BuildVocabulary(DatasetVersion dataset)
    {
        ArgumentNullException.ThrowIfNull(dataset);

        return BuildVocabulary(dataset.Examples.Select(e => e.Input));
    }

    public static Vocabulary BuildVocabulary(IEnumerable<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);

        var words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (string text in texts)
        {
            words.UnionWith(Keywords.Extract(text));
        }

        return Vocabulary.FromWords(words.OrderBy(w => w, StringComparer.Ordinal));
    }

    /// <summary>
    /// Train một model mới từ tập train của một lần chia dữ liệu.
    ///
    /// TỪ ĐIỂN DỰNG TỪ TOÀN BỘ TẬP DỮ LIỆU, không chỉ từ phần train — và đây là
    /// một lựa chọn cần nói rõ, vì nó thoạt nhìn giống rò rỉ dữ liệu.
    ///
    /// Nó không phải rò rỉ: từ điển chỉ quyết định chiều của vector đầu vào,
    /// không mang theo NHÃN nào. Còn nếu dựng từ điển chỉ từ phần train thì mọi
    /// từ chỉ xuất hiện ở tập đánh giá sẽ bị bỏ qua, và những câu đó encode
    /// thành vector rỗng — model sai chúng vì lý do kỹ thuật chứ không phải vì
    /// nó dở, và số liệu đánh giá không còn đo đúng cái cần đo.
    /// </summary>
    public static IntentClassifier Train(DatasetVersion dataset, DataSplit split, TrainingConfig config,
        Action<int, double>? onEpoch = null)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentNullException.ThrowIfNull(split);
        ArgumentNullException.ThrowIfNull(config);

        config.Validate();

        if (split.Training.Count == 0)
        {
            throw new InvalidOperationException("tập train rỗng — không có gì để học");
        }

        var classifier = new IntentClassifier(
            BuildVocabulary(dataset),
            new Random(config.Seed),
            new StochasticGradientDescent(config.LearningRate),
            config.HiddenSize);

        classifier.Train(split.TrainingPairs, config.Epochs, onEpoch);

        return classifier;
    }
}
