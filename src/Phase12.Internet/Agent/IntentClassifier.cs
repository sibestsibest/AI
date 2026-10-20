using Ai.Phase12.Activations;
using Ai.Phase12.Embeddings;
using Ai.Phase12.LinAlg;
using Ai.Phase12.Memory;
using Ai.Phase12.Network;
using Ai.Phase12.Training;

namespace Ai.Phase12.Agent;

/// <summary>Ý định mà agent nhận biết được.</summary>
public enum Intent
{
    /// <summary>Tính toán: "2 + 3 * 4", "Total bằng bao nhiêu"</summary>
    Calculate,

    /// <summary>Ghi nhớ: "hãy nhớ là tôi dùng C#"</summary>
    RememberFact,

    /// <summary>Nhớ lại: "tôi dùng ngôn ngữ gì"</summary>
    RecallMemory,

    /// <summary>Tra cứu kiến thức: "backpropagation là gì"</summary>
    SearchKnowledge,

    /// <summary>Truy vấn dữ liệu: "có bao nhiêu đơn hàng"</summary>
    QueryDatabase,

    /// <summary>Chào hỏi, tán gẫu.</summary>
    SmallTalk,

    /// <summary>
    /// Tra Internet: "tìm trên mạng tin mới nhất về .NET 10" — PHASE 12.
    ///
    /// Vì sao phải là một ý định RIÊNG, không gộp vào SearchKnowledge? Vì hai
    /// việc này khác nhau ở chỗ quan trọng nhất: SearchKnowledge đọc kho tài
    /// liệu nội bộ đã được kiểm, còn SearchInternet lấy dữ liệu chưa kiểm
    /// chứng từ bên ngoài. Cùng một câu trả lời nhưng độ tin cậy khác nhau,
    /// nên đường đi phải khác nhau — và người dùng phải được biết agent vừa
    /// đi đường nào.
    ///
    /// Thêm vào CUỐI enum là có ý: <see cref="OneHot"/> dùng (int)intent làm
    /// chỉ số, nên chèn vào giữa sẽ làm lệch nhãn của mọi ý định phía sau.
    /// </summary>
    SearchInternet,
}

public sealed record IntentPrediction(Intent Intent, double Confidence, Vec Probabilities)
{
    /// <summary>Dưới ngưỡng này thì agent nên nói "tôi không chắc" thay vì đoán bừa.</summary>
    public const double MinimumConfidence = 0.5;

    public bool IsConfident => Confidence >= MinimumConfidence;
}

/// <summary>
/// Bộ phân loại ý định — vai trò "language model" bên trong agent.
///
/// Đây chính là mảnh ghép mà Phase 9 còn thiếu. Reasoning engine tính toán
/// chính xác nhưng không hiểu nổi câu hỏi diễn đạt kiểu khác; mạng neural
/// thì ngược lại — nó khái quát hoá từ ví dụ nhưng không tính đúng được.
/// Ghép lại: MẠNG NEURAL hiểu người dùng muốn gì, rồi giao việc cho CÔNG CỤ
/// KÝ HIỆU thực thi chính xác.
///
/// KIẾN TRÚC (dùng lại y nguyên Phase 2–6):
///     bag-of-words (V chiều) -> hidden 16 (tanh) -> 7 ý định (softmax)
///
/// Bag-of-words: mỗi từ trong từ điển là một chiều, giá trị 1 nếu câu có
/// chứa từ đó. Thô sơ — mất hoàn toàn thứ tự từ — nhưng với việc phân loại
/// ý định thì thường chỉ cần biết CÓ những từ nào là đủ.
/// </summary>
public sealed class IntentClassifier
{
    private static readonly Intent[] AllIntents = Enum.GetValues<Intent>();

    private readonly NeuralNetwork _network;
    private readonly SoftmaxCrossEntropy _loss = new();
    private readonly IOptimizer _optimizer;

    public Vocabulary Vocabulary { get; }

    public IntentClassifier(IEnumerable<string> trainingTexts, Random rng, IOptimizer? optimizer = null)
    {
        ArgumentNullException.ThrowIfNull(trainingTexts);
        ArgumentNullException.ThrowIfNull(rng);

        Vocabulary = new Vocabulary();
        foreach (string text in trainingTexts)
        {
            foreach (string word in Keywords.Extract(text))
            {
                Vocabulary.Add(word);
            }
        }

        _optimizer = optimizer ?? new StochasticGradientDescent(0.3);
        _network = new NeuralNetwork(
            new Layer(Vocabulary.Count, 16, new Tanh(), rng),
            new Layer(16, AllIntents.Length, new Identity(), rng));
    }

    public int ParameterCount => _network.ParameterCount;

    /// <summary>
    /// Câu -> vector bag-of-words. Từ chưa từng gặp bị BỎ QUA hoàn toàn —
    /// đây là giới hạn thật: model chỉ nhận ra những từ nó đã thấy lúc train.
    /// </summary>
    public Vec Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var vector = new Vec(Vocabulary.Count);
        foreach (string word in Keywords.Extract(text))
        {
            if (Vocabulary.Contains(word))
            {
                vector[Vocabulary.IndexOf(word)] = 1;
            }
        }

        return vector;
    }

    /// <summary>Số từ trong câu mà model thực sự nhận ra.</summary>
    public int KnownWordCount(string text) =>
        Keywords.Extract(text).Count(Vocabulary.Contains);

    public IntentPrediction Classify(string text)
    {
        var probabilities = Softmax.Apply(_network.Forward(Encode(text)));

        int best = 0;
        for (int i = 1; i < probabilities.Length; i++)
        {
            if (probabilities[i] > probabilities[best]) best = i;
        }

        // Không nhận ra từ nào thì mọi dự đoán đều vô nghĩa — hạ độ tự tin về 0.
        double confidence = KnownWordCount(text) == 0 ? 0 : probabilities[best];

        return new IntentPrediction(AllIntents[best], confidence, probabilities);
    }

    public void Train(IReadOnlyList<(string Text, Intent Label)> examples, int epochs, Action<int, double>? onEpoch = null)
    {
        ArgumentNullException.ThrowIfNull(examples);
        if (examples.Count == 0) throw new ArgumentException("Cần ít nhất 1 ví dụ.", nameof(examples));

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            _network.ZeroGradients();
            double totalLoss = 0;

            foreach (var (text, label) in examples)
            {
                var input = Encode(text);
                var target = OneHot(label);

                var logits = _network.Forward(input);
                totalLoss += _loss.Compute(logits, target);
                _network.Backward(_loss.Gradient(logits, target));
            }

            var gradients = _network.GetGradients();
            for (int i = 0; i < gradients.Length; i++) gradients[i] /= examples.Count;

            var parameters = _network.GetParameters();
            _optimizer.Step(parameters, gradients);
            _network.SetParameters(parameters);

            onEpoch?.Invoke(epoch, totalLoss / examples.Count);
        }
    }

    public double Accuracy(IReadOnlyList<(string Text, Intent Label)> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        int correct = examples.Count(e => Classify(e.Text).Intent == e.Label);
        return (double)correct / examples.Count;
    }

    public static Vec OneHot(Intent intent)
    {
        var v = new Vec(AllIntents.Length);
        v[(int)intent] = 1;
        return v;
    }

    public static Intent IntentAt(int index) => AllIntents[index];
}
