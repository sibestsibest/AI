using Ai.Phase5.Activations;
using Ai.Phase5.Training;

namespace Ai.Phase5.Embeddings;

/// <summary>
/// Một mẫu huấn luyện: từ trung tâm, từ ngữ cảnh, và nhãn
/// (1 = thật sự xuất hiện cùng nhau, 0 = cặp âm lấy ngẫu nhiên).
/// </summary>
public readonly record struct WordPair(int Center, int Context, double Label);

/// <summary>
/// Skip-gram với negative sampling (word2vec, Mikolov 2013) — rút gọn tối đa.
///
/// GIẢ THIẾT NỀN TẢNG (phân bố ngữ nghĩa, Firth 1957):
///     "Bạn nhận ra một từ qua những từ đi cùng nó."
///
/// MÔ HÌNH:
///     P(b là ngữ cảnh của a) = sigmoid( center[a] · context[b] )
///
/// VÌ SAO PHẢI CÓ HAI BẢNG EMBEDDING?
/// Đây là chi tiết dễ bỏ sót nhất, và bỏ sót là model học sai hẳn.
///
/// Nếu chỉ dùng MỘT bảng, ta đo "cat và dog có hay đi cạnh nhau không".
/// Trong ngữ liệu, cat và dog KHÔNG BAO GIỜ đứng cùng câu, nên cặp
/// (cat, dog) bị lấy làm cặp âm và model tích cực ĐẨY chúng ra xa —
/// ngược hẳn điều ta muốn.
///
/// Với HAI bảng, cặp âm (cat, dog) chỉ đẩy center[cat] khỏi context[dog],
/// không đụng tới center[dog]. Trong khi đó cả center[cat] lẫn center[dog]
/// đều buộc phải có tích vô hướng cao với context[pet], context[animal],
/// context[fur], context[tail]. Muốn vậy chúng phải nằm cùng một vùng
/// không gian — và thế là chúng GIỐNG nhau.
///
/// Nói cách khác: một bảng học quan hệ "đi cạnh nhau" (syntagmatic),
/// hai bảng học quan hệ "thay thế được cho nhau" (paradigmatic). Cái thứ hai
/// mới là cái ta gọi là tương đồng ngữ nghĩa.
///
/// GRADIENT (tự dẫn được, nhờ sigmoid + BCE triệt tiêu nhau — xem Phase 4):
///     p = sigmoid( c · x )      c = center[a] , x = context[b]
///     δ = p − y
///     ∂L/∂c = δ · x
///     ∂L/∂x = δ · c
/// </summary>
public sealed class SkipGramModel
{
    private static readonly Sigmoid SigmoidFunction = new();

    private readonly IOptimizer _optimizer;

    /// <summary>Vector "từ trung tâm" — đây là embedding dùng để đo similarity.</summary>
    public EmbeddingTable Center { get; }

    /// <summary>Vector "từ ngữ cảnh" — chỉ dùng trong lúc huấn luyện rồi bỏ đi.</summary>
    public EmbeddingTable Context { get; }

    public SkipGramModel(Vocabulary vocabulary, int dimensions, Random rng, IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        ArgumentNullException.ThrowIfNull(rng);
        ArgumentNullException.ThrowIfNull(optimizer);

        Center = new EmbeddingTable(vocabulary, dimensions, rng);
        Context = new EmbeddingTable(vocabulary, dimensions, rng);
        _optimizer = optimizer;
    }

    /// <summary>Xác suất mô hình gán cho việc hai từ đi cùng nhau.</summary>
    public double Probability(int center, int context) =>
        SigmoidFunction.Forward(Center.Lookup(center).Dot(Context.Lookup(context)));

    /// <summary>Chạy một epoch, trả về loss trung bình đo TRƯỚC khi cập nhật.</summary>
    public double RunEpoch(IReadOnlyList<WordPair> pairs)
    {
        ArgumentNullException.ThrowIfNull(pairs);
        if (pairs.Count == 0) throw new ArgumentException("Cần ít nhất 1 cặp.", nameof(pairs));

        Center.ZeroGradients();
        Context.ZeroGradients();

        double totalLoss = 0;

        foreach (var pair in pairs)
        {
            var c = Center.Lookup(pair.Center);
            var x = Context.Lookup(pair.Context);

            // Forward
            double probability = SigmoidFunction.Forward(c.Dot(x));

            // Loss (binary cross entropy, chặn log(0))
            double clamped = Math.Clamp(probability, 1e-12, 1 - 1e-12);
            totalLoss += -(pair.Label * Math.Log(clamped) + (1 - pair.Label) * Math.Log(1 - clamped));

            // Backward
            double delta = probability - pair.Label;
            Center.AccumulateGradient(pair.Center, x.Scale(delta));
            Context.AccumulateGradient(pair.Context, c.Scale(delta));
        }

        ApplyGradients(Center, pairs.Count);
        ApplyGradients(Context, pairs.Count);

        return totalLoss / pairs.Count;
    }

    public void Train(IReadOnlyList<WordPair> pairs, int epochs, Action<int, double>? onEpoch = null)
    {
        if (epochs < 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double loss = RunEpoch(pairs);
            onEpoch?.Invoke(epoch, loss);
        }
    }

    private void ApplyGradients(EmbeddingTable table, int sampleCount)
    {
        var gradients = table.GetGradients();
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] /= sampleCount;
        }

        var parameters = table.GetParameters();
        _optimizer.Step(parameters, gradients);
        table.SetParameters(parameters);
    }
}
