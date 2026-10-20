using Ai.Phase6.Activations;
using Ai.Phase6.Embeddings;
using Ai.Phase6.LinAlg;
using Ai.Phase6.Network;
using Ai.Phase6.Training;

namespace Ai.Phase6.Language;

/// <summary>Một mẫu huấn luyện: ngữ cảnh (các token trước) -> token tiếp theo.</summary>
public readonly record struct NextTokenSample(int[] Context, int Target);

/// <summary>
/// Mô hình ngôn ngữ neural, kiến trúc Bengio 2003 rút gọn.
///
///   ngữ cảnh: ["I", "like"]
///        │
///        ├─ embedding("I")    -> vector d chiều
///        └─ embedding("like") -> vector d chiều
///        │
///        ▼ nối lại thành vector (contextSize · d) chiều
///   [Layer ẩn, Tanh]
///        ▼
///   [Layer ra, Identity]  -> LOGITS, V số (một số cho mỗi từ trong từ điển)
///        ▼
///   softmax               -> PHÂN BỐ XÁC SUẤT trên toàn bộ từ điển
///
/// Đây chính xác là bộ khung của GPT, bỏ đi attention (Phase 7–8) và thu nhỏ
/// mọi con số. "Dự đoán từ tiếp theo" là toàn bộ mục tiêu huấn luyện —
/// không hơn.
/// </summary>
public sealed class LanguageModel
{
    private readonly SoftmaxCrossEntropy _loss = new();
    private readonly IOptimizer _optimizer;

    public Tokenizer Tokenizer { get; }

    public EmbeddingTable Embeddings { get; }

    public NeuralNetwork Network { get; }

    public int ContextSize { get; }

    public int VocabularySize => Tokenizer.Vocabulary.Count;

    public LanguageModel(
        Tokenizer tokenizer,
        int contextSize,
        int embeddingDimensions,
        int hiddenSize,
        Random rng,
        IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(rng);
        ArgumentNullException.ThrowIfNull(optimizer);
        if (contextSize < 1) throw new ArgumentOutOfRangeException(nameof(contextSize));

        Tokenizer = tokenizer;
        ContextSize = contextSize;
        _optimizer = optimizer;

        Embeddings = new EmbeddingTable(tokenizer.Vocabulary, embeddingDimensions, rng);

        int inputSize = contextSize * embeddingDimensions;
        Network = new NeuralNetwork(
            new Layer(inputSize, hiddenSize, new Tanh(), rng),
            new Layer(hiddenSize, tokenizer.Vocabulary.Count, new Identity(), rng));
    }

    public int ParameterCount => Embeddings.Count * Embeddings.Dimensions + Network.ParameterCount;

    /// <summary>Nối các vector embedding của ngữ cảnh thành một vector duy nhất.</summary>
    public Vec EncodeContext(int[] context)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (context.Length != ContextSize)
        {
            throw new ArgumentException(
                $"Ngữ cảnh phải có {ContextSize} token nhưng nhận {context.Length}.", nameof(context));
        }

        int d = Embeddings.Dimensions;
        var input = new Vec(ContextSize * d);

        for (int position = 0; position < ContextSize; position++)
        {
            var embedding = Embeddings.Lookup(context[position]);
            for (int i = 0; i < d; i++)
            {
                input[position * d + i] = embedding[i];
            }
        }

        return input;
    }

    /// <summary>Logits thô — một số cho mỗi từ trong từ điển. Chưa phải xác suất.</summary>
    public Vec Logits(int[] context) => Network.Forward(EncodeContext(context));

    /// <summary>Phân bố xác suất của từ tiếp theo.</summary>
    public Vec Predict(int[] context) => Softmax.Apply(Logits(context));

    /// <summary>Phân bố xác suất của từ tiếp theo, cho ngữ cảnh viết bằng chữ.</summary>
    public Vec Predict(string contextText)
    {
        var tokens = Tokenizer.Encode(contextText);

        if (tokens.Length < ContextSize)
        {
            // Đệm <s> ở đầu nếu ngữ cảnh ngắn hơn cửa sổ.
            var padded = new int[ContextSize];
            int pad = ContextSize - tokens.Length;
            for (int i = 0; i < pad; i++) padded[i] = Tokenizer.StartId;
            Array.Copy(tokens, 0, padded, pad, tokens.Length);
            return Predict(padded);
        }

        return Predict(tokens[^ContextSize..]);
    }

    /// <summary>Các từ có xác suất cao nhất, kèm xác suất.</summary>
    public (string Word, double Probability)[] TopPredictions(string contextText, int take = 5)
    {
        var probabilities = Predict(contextText);

        return Enumerable.Range(0, probabilities.Length)
            .Select(i => (Word: Tokenizer.Vocabulary.WordAt(i), Probability: probabilities[i]))
            .OrderByDescending(x => x.Probability)
            .Take(take)
            .ToArray();
    }

    /// <summary>
    /// Sinh văn bản: dự đoán một từ, thêm vào ngữ cảnh, lặp lại.
    /// Đây đúng là cách mọi LLM sinh chữ — một token mỗi lần, tự nối tiếp
    /// đầu ra của chính mình.
    /// </summary>
    public string Generate(string prompt, int maxTokens = 10, double temperature = 1.0, Random? rng = null)
    {
        var tokens = new List<int>();
        for (int i = 0; i < ContextSize; i++) tokens.Add(Tokenizer.StartId);
        tokens.AddRange(Tokenizer.Encode(prompt));

        var generated = new List<string>();

        for (int step = 0; step < maxTokens; step++)
        {
            var context = tokens.TakeLast(ContextSize).ToArray();
            var probabilities = Softmax.Apply(Logits(context), temperature);

            int next = rng is null ? ArgMax(probabilities) : Sample(probabilities, rng);
            if (next == Tokenizer.EndId) break;

            tokens.Add(next);
            generated.Add(Tokenizer.Vocabulary.WordAt(next));
        }

        return string.Join(' ', generated);
    }

    /// <summary>Sinh các mẫu "ngữ cảnh -> từ tiếp theo" từ một tập câu.</summary>
    public NextTokenSample[] BuildSamples(IEnumerable<string> sentences)
    {
        ArgumentNullException.ThrowIfNull(sentences);

        var samples = new List<NextTokenSample>();

        foreach (string sentence in sentences)
        {
            var tokens = Tokenizer.EncodeWithBoundaries(sentence, ContextSize);

            for (int i = ContextSize; i < tokens.Length; i++)
            {
                samples.Add(new NextTokenSample(tokens[(i - ContextSize)..i], tokens[i]));
            }
        }

        return [.. samples];
    }

    // ------------------------------------------------------------------
    // Huấn luyện
    // ------------------------------------------------------------------

    /// <summary>
    /// Một epoch. Chuỗi chain rule giờ dài thêm một mắt xích so với Phase 3:
    ///
    ///   loss -> logits -> layer ra -> layer ẩn -> VECTOR EMBEDDING -> bảng embedding
    ///
    /// Mắt xích cuối là điểm mới: <c>Network.Backward</c> trả về ∂L/∂x, và ta
    /// cắt vector đó thành từng đoạn d chiều rồi cộng vào gradient của đúng
    /// token đã tra cứu. Embedding được học chính bằng gradient descent, không
    /// khác gì trọng số thường.
    /// </summary>
    public double RunEpoch(IReadOnlyList<NextTokenSample> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count == 0) throw new ArgumentException("Cần ít nhất 1 mẫu.", nameof(samples));

        Network.ZeroGradients();
        Embeddings.ZeroGradients();

        double totalLoss = 0;
        int d = Embeddings.Dimensions;

        foreach (var sample in samples)
        {
            var target = OneHot(sample.Target);

            // Forward
            var logits = Logits(sample.Context);
            totalLoss += _loss.Compute(logits, target);

            // Backward qua mạng: ∂L/∂logits = p − y
            var gradientWrtInput = Network.Backward(_loss.Gradient(logits, target));

            // Backward tiếp vào bảng embedding
            for (int position = 0; position < ContextSize; position++)
            {
                var slice = new Vec(d);
                for (int i = 0; i < d; i++)
                {
                    slice[i] = gradientWrtInput[position * d + i];
                }

                Embeddings.AccumulateGradient(sample.Context[position], slice);
            }
        }

        ApplyNetworkGradients(samples.Count);
        ApplyEmbeddingGradients(samples.Count);

        return totalLoss / samples.Count;
    }

    public void Train(IReadOnlyList<NextTokenSample> samples, int epochs, Action<int, double>? onEpoch = null)
    {
        if (epochs < 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double loss = RunEpoch(samples);
            onEpoch?.Invoke(epoch, loss);
        }
    }

    public double AverageLoss(IReadOnlyList<NextTokenSample> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);

        double total = 0;
        foreach (var sample in samples)
        {
            total += _loss.Compute(Logits(sample.Context), OneHot(sample.Target));
        }

        return total / samples.Count;
    }

    public Vec OneHot(int tokenId)
    {
        var v = new Vec(VocabularySize);
        v[tokenId] = 1;
        return v;
    }

    private void ApplyNetworkGradients(int sampleCount)
    {
        var gradients = Network.GetGradients();
        for (int i = 0; i < gradients.Length; i++) gradients[i] /= sampleCount;

        var parameters = Network.GetParameters();
        _optimizer.Step(parameters, gradients);
        Network.SetParameters(parameters);
    }

    private void ApplyEmbeddingGradients(int sampleCount)
    {
        var gradients = Embeddings.GetGradients();
        for (int i = 0; i < gradients.Length; i++) gradients[i] /= sampleCount;

        var parameters = Embeddings.GetParameters();
        _optimizer.Step(parameters, gradients);
        Embeddings.SetParameters(parameters);
    }

    private static int ArgMax(Vec v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best]) best = i;
        }

        return best;
    }

    private static int Sample(Vec probabilities, Random rng)
    {
        double roll = rng.NextDouble();
        double cumulative = 0;

        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (roll <= cumulative) return i;
        }

        return probabilities.Length - 1;
    }
}
