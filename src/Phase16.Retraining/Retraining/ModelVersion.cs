using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Ai.Phase16.Agent;
using Ai.Phase16.Embeddings;
using Ai.Phase16.Training;

namespace Ai.Phase16.Retraining;

/// <summary>
/// MỘT PHIÊN BẢN MODEL — bất biến, dựng lại chạy được, truy được về dữ liệu.
///
/// Chứa TRỌNG SỐ THẬT, không phải chỉ siêu dữ liệu. Đây là khác biệt giữa quay
/// lui thật và "quay lui" kiểu train lại rồi hy vọng ra như cũ: train lại phụ
/// thuộc vào việc máy đó, phiên bản đó, thư viện đó còn cho ra đúng dãy số như
/// hôm trước. Có trọng số trong tay thì <see cref="Restore"/> dựng lại đúng
/// model cũ, không phụ thuộc gì cả.
///
/// Cũng chứa <see cref="VocabularyWords"/> theo ĐÚNG THỨ TỰ đã train, vì chỉ số
/// của từ quyết định vị trí trọng số. Lưu trọng số mà quên từ điển thì cả mảng
/// số kia vô nghĩa: cùng bộ trọng số, từ điển xếp khác đi là một model khác hẳn.
///
/// <see cref="DatasetFingerprint"/> và <see cref="Config"/> là hai nửa của lời
/// hứa tái lập — biết cả hai thì dựng lại được chính xác model này từ đầu.
/// </summary>
public sealed record ModelVersion
{
    public required int Version { get; init; }

    /// <summary>Trọng số phẳng, đúng bố cục của <c>NeuralNetwork.GetParameters</c>.</summary>
    public required IReadOnlyList<double> Weights { get; init; }

    /// <summary>Từ điển theo đúng thứ tự chỉ số đã dùng khi train.</summary>
    public required IReadOnlyList<string> VocabularyWords { get; init; }

    public required TrainingConfig Config { get; init; }

    /// <summary>Phiên bản tập dữ liệu (Phase 15) đã train ra model này.</summary>
    public required int DatasetVersion { get; init; }

    public required string DatasetFingerprint { get; init; }

    /// <summary>Số liệu đo trên tập đánh giá — không phải trên tập train.</summary>
    public required EvaluationMetrics Metrics { get; init; }

    /// <summary>Số liệu trên bộ vàng, nếu đã chấm.</summary>
    public EvaluationMetrics? GoldenMetrics { get; init; }

    public required DateTimeOffset TrainedAt { get; init; }

    public int? ParentVersion { get; init; }

    public string Note { get; init; } = "";

    public int ParameterCount => Weights.Count;

    public string Label => $"m{Version}";

    /// <summary>
    /// VÂN TAY TRỌNG SỐ — băm chính dãy số, không băm siêu dữ liệu.
    ///
    /// Nhờ vậy câu "cùng dữ liệu + cùng cấu hình thì ra cùng model" kiểm chứng
    /// được bằng một phép so chuỗi, thay vì phải so từng tham số một. Số thực
    /// được in bằng "R" (round-trip) và theo InvariantCulture, nên vân tay
    /// không đổi theo locale của máy.
    /// </summary>
    public string Fingerprint
    {
        get
        {
            var builder = new StringBuilder();
            foreach (double weight in Weights)
            {
                builder.Append(weight.ToString("R", CultureInfo.InvariantCulture)).Append(';');
            }

            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
            return Convert.ToHexString(hash)[..12].ToLowerInvariant();
        }
    }

    /// <summary>
    /// Dựng lại model chạy được từ bản lưu.
    ///
    /// Random ở đây chỉ để khởi tạo rồi bị ghi đè hoàn toàn bởi
    /// <c>ImportWeights</c> — nhưng vẫn truyền seed của cấu hình vào, để nếu
    /// có ngày bản lưu hỏng thì model dựng ra vẫn là một model tất định, chứ
    /// không phải một mớ trọng số ngẫu nhiên khác nhau mỗi lần đọc.
    /// </summary>
    public IntentClassifier Restore()
    {
        var classifier = new IntentClassifier(
            Vocabulary.FromWords(VocabularyWords),
            new Random(Config.Seed),
            new StochasticGradientDescent(Config.LearningRate),
            Config.HiddenSize);

        classifier.ImportWeights([.. Weights]);
        return classifier;
    }

    public override string ToString() =>
        $"{Label} (dữ liệu v{DatasetVersion}/{DatasetFingerprint}, {ParameterCount} tham số, " +
        $"{Metrics}, vân tay {Fingerprint})";
}
