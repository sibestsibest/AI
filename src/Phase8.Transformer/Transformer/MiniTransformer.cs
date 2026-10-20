using Ai.Phase8.Activations;
using Ai.Phase8.Embeddings;
using Ai.Phase8.Language;
using Ai.Phase8.LinAlg;
using Ai.Phase8.Training;

namespace Ai.Phase8.Transformer;

/// <summary>
/// Mini Transformer hoàn chỉnh — đúng chuỗi kiến trúc của đề bài:
///
///     token ids
///        ▼
///     Embedding                    tra bảng, mỗi token -> vector d chiều
///        ▼
///     + Positional Encoding        thêm thông tin thứ tự
///        ▼
///   ┌─ TransformerBlock ────────────────────────────────┐
///   │   LayerNorm -> Self Attention -> + residual       │
///   │   LayerNorm -> Feed Forward   -> + residual       │
///   └───────────────────────────────────────────────────┘
///        ▼  (có thể xếp chồng nhiều khối)
///     LayerNorm cuối
///        ▼
///     Output head (d -> V)          logits
///        ▼
///     softmax                       phân bố xác suất từ tiếp theo
///
/// Đây chính là GPT, thu nhỏ: bỏ multi-head, chỉ 1–2 khối, mọi con số bé lại.
/// Không thiếu thành phần kiến trúc nào.
/// </summary>
public sealed class MiniTransformer
{
    private readonly SoftmaxCrossEntropy _loss = new();
    private readonly IOptimizer _optimizer;
    private readonly Mat _positionalEncoding;
    private readonly TransformerBlock[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly Dense _outputHead;

    private int[]? _lastTokens;

    public Tokenizer Tokenizer { get; }

    public EmbeddingTable Embeddings { get; }

    public int ModelDimensions { get; }

    public int MaxSequenceLength { get; }

    public IReadOnlyList<TransformerBlock> Blocks => _blocks;

    public MiniTransformer(
        Tokenizer tokenizer,
        int modelDimensions,
        int headDimensions,
        int feedForwardSize,
        int blockCount,
        int maxSequenceLength,
        Random rng,
        IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(rng);
        ArgumentNullException.ThrowIfNull(optimizer);

        Tokenizer = tokenizer;
        ModelDimensions = modelDimensions;
        MaxSequenceLength = maxSequenceLength;
        _optimizer = optimizer;

        Embeddings = new EmbeddingTable(tokenizer.Vocabulary, modelDimensions, rng);
        _positionalEncoding = PositionalEncoding.Build(maxSequenceLength, modelDimensions);

        _blocks = new TransformerBlock[blockCount];
        for (int i = 0; i < blockCount; i++)
        {
            _blocks[i] = new TransformerBlock($"block{i}", modelDimensions, headDimensions, feedForwardSize, rng);
        }

        _finalNorm = new LayerNorm("ln_f", modelDimensions);
        _outputHead = new Dense("head", modelDimensions, tokenizer.Vocabulary.Count, new Identity(), rng);
    }

    public int VocabularySize => Tokenizer.Vocabulary.Count;

    public IEnumerable<Param> ModuleParameters() =>
    [
        .. _blocks.SelectMany(b => b.Parameters()),
        .. _finalNorm.Parameters(),
        .. _outputHead.Parameters(),
    ];

    public int ParameterCount =>
        Embeddings.Count * Embeddings.Dimensions + ModuleParameters().Sum(p => p.Count);

    // ------------------------------------------------------------------
    // Forward
    // ------------------------------------------------------------------

    /// <summary>Logits cho từ tiếp theo, dựa trên vị trí CUỐI của chuỗi.</summary>
    public Vec Forward(int[] tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Length == 0) throw new ArgumentException("Chuỗi rỗng.", nameof(tokens));
        if (tokens.Length > MaxSequenceLength)
        {
            throw new ArgumentException($"Chuỗi dài {tokens.Length} vượt quá {MaxSequenceLength}.", nameof(tokens));
        }

        _lastTokens = tokens;

        // 1. Embedding: mỗi token thành một hàng.
        var x = new Mat(tokens.Length, ModelDimensions);
        for (int i = 0; i < tokens.Length; i++)
        {
            x.SetRow(i, Embeddings.Lookup(tokens[i]));
        }

        // 2. Cộng mã vị trí.
        x = PositionalEncoding.Add(x, _positionalEncoding);

        // 3. Các khối transformer.
        foreach (var block in _blocks)
        {
            x = block.Forward(x);
        }

        // 4. Chuẩn hoá cuối.
        x = _finalNorm.Forward(x);

        // 5. Chỉ vị trí CUỐI mới dùng để dự đoán từ tiếp theo.
        var lastRow = Mat.FromRowVectors(x.Row(x.Rows - 1));
        return _outputHead.Forward(lastRow).Row(0);
    }

    public Vec Predict(int[] tokens) => Softmax.Apply(Forward(tokens));

    public Vec Predict(string contextText)
    {
        var tokens = new List<int> { Tokenizer.StartId };
        tokens.AddRange(Tokenizer.Encode(contextText));
        return Predict([.. tokens]);
    }

    public (string Word, double Probability)[] TopPredictions(string contextText, int take = 4)
    {
        var probabilities = Predict(contextText);

        return Enumerable.Range(0, probabilities.Length)
            .Select(i => (Word: Tokenizer.Vocabulary.WordAt(i), Probability: probabilities[i]))
            .OrderByDescending(x => x.Probability)
            .Take(take)
            .ToArray();
    }

    /// <summary>Trọng số attention của khối đầu tiên ở lần forward gần nhất.</summary>
    public Mat? LastAttentionWeights => _blocks[0].Attention.LastAttentionWeights;

    // ------------------------------------------------------------------
    // Backward
    // ------------------------------------------------------------------

    /// <summary>
    /// Đẩy gradient ngược qua toàn bộ kiến trúc, về tận bảng embedding.
    /// </summary>
    public void Backward(Vec gradientWrtLogits)
    {
        ArgumentNullException.ThrowIfNull(gradientWrtLogits);

        if (_lastTokens is null)
        {
            throw new InvalidOperationException("Phải gọi Forward trước Backward.");
        }

        int n = _lastTokens.Length;

        // 5'. Qua output head (chỉ hàng cuối có gradient).
        var dLast = _outputHead.Backward(Mat.FromRowVectors(gradientWrtLogits));

        var dx = new Mat(n, ModelDimensions);
        dx.SetRow(n - 1, dLast.Row(0));

        // 4'. Qua LayerNorm cuối.
        dx = _finalNorm.Backward(dx);

        // 3'. Qua các khối, theo thứ tự NGƯỢC.
        for (int i = _blocks.Length - 1; i >= 0; i--)
        {
            dx = _blocks[i].Backward(dx);
        }

        // 2'. Mã vị trí là hằng số -> gradient đi thẳng qua, không đổi.

        // 1'. Vào bảng embedding. Token lặp lại thì gradient CỘNG DỒN.
        for (int i = 0; i < n; i++)
        {
            Embeddings.AccumulateGradient(_lastTokens[i], dx.Row(i));
        }
    }

    public void ZeroGradients()
    {
        Embeddings.ZeroGradients();
        foreach (var p in ModuleParameters()) p.ZeroGrad();
    }

    // ------------------------------------------------------------------
    // Huấn luyện
    // ------------------------------------------------------------------

    public double RunEpoch(IReadOnlyList<NextTokenSample> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count == 0) throw new ArgumentException("Cần ít nhất 1 mẫu.", nameof(samples));

        ZeroGradients();
        double totalLoss = 0;

        foreach (var sample in samples)
        {
            var target = OneHot(sample.Target);
            var logits = Forward(sample.Context);

            totalLoss += _loss.Compute(logits, target);
            Backward(_loss.Gradient(logits, target));
        }

        ApplyGradients(samples.Count);
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
            total += _loss.Compute(Forward(sample.Context), OneHot(sample.Target));
        }

        return total / samples.Count;
    }

    /// <summary>
    /// Mẫu huấn luyện: mọi tiền tố của câu -> token kế tiếp.
    /// "&lt;s&gt; I like cats &lt;/s&gt;" cho ra 4 mẫu, mỗi mẫu là một tiền tố dài dần.
    /// </summary>
    public NextTokenSample[] BuildSamples(IEnumerable<string> sentences)
    {
        ArgumentNullException.ThrowIfNull(sentences);

        var samples = new List<NextTokenSample>();

        foreach (string sentence in sentences)
        {
            var tokens = new List<int> { Tokenizer.StartId };
            tokens.AddRange(Tokenizer.Encode(sentence));
            tokens.Add(Tokenizer.EndId);

            for (int i = 1; i < tokens.Count; i++)
            {
                samples.Add(new NextTokenSample(tokens.Take(i).ToArray(), tokens[i]));
            }
        }

        return [.. samples];
    }

    public string Generate(string prompt, int maxTokens = 8, double temperature = 1.0, Random? rng = null)
    {
        var tokens = new List<int> { Tokenizer.StartId };
        tokens.AddRange(Tokenizer.Encode(prompt));

        var generated = new List<string>();

        for (int step = 0; step < maxTokens && tokens.Count < MaxSequenceLength; step++)
        {
            var probabilities = Softmax.Apply(Forward([.. tokens]), temperature);

            int next = rng is null ? ArgMax(probabilities) : Sample(probabilities, rng);
            if (next == Tokenizer.EndId) break;

            tokens.Add(next);
            generated.Add(Tokenizer.Vocabulary.WordAt(next));
        }

        return string.Join(' ', generated);
    }

    public Vec OneHot(int tokenId)
    {
        var v = new Vec(VocabularySize);
        v[tokenId] = 1;
        return v;
    }

    // ------------------------------------------------------------------
    // Tham số dạng mảng phẳng (cho optimizer và gradient check)
    // ------------------------------------------------------------------

    public double[] GetAllParameters()
    {
        var flat = new List<double>(Embeddings.GetParameters());

        foreach (var p in ModuleParameters())
        {
            for (int r = 0; r < p.Value.Rows; r++)
            {
                for (int c = 0; c < p.Value.Cols; c++) flat.Add(p.Value[r, c]);
            }
        }

        return [.. flat];
    }

    public void SetAllParameters(double[] parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);

        int embeddingCount = Embeddings.Count * Embeddings.Dimensions;
        Embeddings.SetParameters(parameters[..embeddingCount]);

        int k = embeddingCount;
        foreach (var p in ModuleParameters())
        {
            for (int r = 0; r < p.Value.Rows; r++)
            {
                for (int c = 0; c < p.Value.Cols; c++) p.Value[r, c] = parameters[k++];
            }
        }
    }

    public double[] GetAllGradients()
    {
        var flat = new List<double>(Embeddings.GetGradients());

        foreach (var p in ModuleParameters())
        {
            for (int r = 0; r < p.Grad.Rows; r++)
            {
                for (int c = 0; c < p.Grad.Cols; c++) flat.Add(p.Grad[r, c]);
            }
        }

        return [.. flat];
    }

    private void ApplyGradients(int sampleCount)
    {
        var gradients = GetAllGradients();
        for (int i = 0; i < gradients.Length; i++) gradients[i] /= sampleCount;

        var parameters = GetAllParameters();
        _optimizer.Step(parameters, gradients);
        SetAllParameters(parameters);
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
