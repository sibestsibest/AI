using System.Diagnostics;
using Ai.Phase8.Language;
using Ai.Phase8.LinAlg;
using Ai.Phase8.Training;
using Ai.Phase8.Transformer;

namespace Ai.Phase8.Demos;

public static class TransformerDemo
{
    public static MiniTransformer Build(string[] corpus, int seed = 42) =>
        new(
            new Tokenizer(corpus),
            modelDimensions: 16,
            headDimensions: 16,
            feedForwardSize: 32,
            blockCount: 2,
            maxSequenceLength: 8,
            new Random(seed),
            new StochasticGradientDescent(0.1));

    public static MiniTransformer Train(string[] corpus, int epochs = 1500, int seed = 42)
    {
        var model = Build(corpus, seed);
        model.Train(model.BuildSamples(corpus), epochs);
        return model;
    }

    public static void Run()
    {
        ShowPositionalEncoding();
        ShowLayerNorm();
        ShowArchitecture();
        var model = ShowTraining();
        ShowAttentionPattern(model);
        ShowGeneration(model);
    }

    private static void ShowPositionalEncoding()
    {
        Console.WriteLine("=== DEMO 1: POSITIONAL ENCODING ===\n");

        Console.WriteLine("  Attention không biết thứ tự: nó chỉ tính tích vô hướng giữa mọi cặp.");
        Console.WriteLine("  Với attention thuần, \"mèo đuổi chuột\" và \"chuột đuổi mèo\" là một.\n");

        var encoding = PositionalEncoding.Build(maxLength: 6, dimensions: 8);
        Console.WriteLine("  PE(pos, 2i) = sin(pos / 10000^(2i/d)) , PE(pos, 2i+1) = cos(...)\n");
        Console.WriteLine($"  {"vị trí",7} | các chiều 0..7");
        Console.WriteLine("  " + new string('-', 70));

        for (int pos = 0; pos < encoding.Rows; pos++)
        {
            Console.WriteLine($"  {pos,7} | {encoding.Row(pos)}");
        }

        Console.WriteLine("\n  Cột 0–1 đổi nhanh (phân biệt vị trí liền kề), cột 6–7 đổi rất chậm");
        Console.WriteLine("  (phân biệt đầu câu với cuối câu). Mỗi vị trí có một dấu vân tay riêng.");

        Console.WriteLine("\n  Cosine similarity giữa các vị trí (gần nhau -> giống nhau hơn):");
        for (int pos = 1; pos < encoding.Rows; pos++)
        {
            double similarity = Embeddings.Similarity.Cosine(encoding.Row(0), encoding.Row(pos));
            Console.WriteLine($"    cos(vị trí 0, vị trí {pos}) = {similarity,8:F4}");
        }
    }

    private static void ShowLayerNorm()
    {
        Console.WriteLine("\n\n=== DEMO 2: LAYER NORMALIZATION ===\n");

        var norm = new LayerNorm("demo", 4);
        var x = Mat.FromRows(
            [1, 2, 3, 4],
            [100, 200, 300, 400],
            [-5, 0, 5, 10]);

        Console.WriteLine("  Đầu vào (ba token có thang đo rất khác nhau):");
        Console.WriteLine(x);

        var y = norm.Forward(x);
        Console.WriteLine("\n  Sau LayerNorm (γ = 1, β = 0):");
        Console.WriteLine(y);

        Console.WriteLine("\n  Kiểm tra từng hàng:");
        for (int i = 0; i < y.Rows; i++)
        {
            var row = y.Row(i);
            double mean = row.Sum() / row.Length;
            double variance = 0;
            for (int j = 0; j < row.Length; j++) variance += (row[j] - mean) * (row[j] - mean);
            variance /= row.Length;

            Console.WriteLine($"    hàng {i}: trung bình = {mean,10:F6} , phương sai = {variance,10:F6}");
        }

        Console.WriteLine("\n  Ba token có thang đo chênh nhau 100 lần đều được kéo về cùng một");
        Console.WriteLine("  thang đo. Nhờ vậy độ lớn không trôi dần qua các layer.");
    }

    private static void ShowArchitecture()
    {
        Console.WriteLine("\n\n=== DEMO 3: KIẾN TRÚC ===\n");

        var model = Build(TextCorpus.Basic);

        Console.WriteLine("     token ids");
        Console.WriteLine("        ▼");
        Console.WriteLine($"     Embedding                  {model.Embeddings.Count} × {model.ModelDimensions} = " +
                          $"{model.Embeddings.Count * model.ModelDimensions} tham số");
        Console.WriteLine("        ▼");
        Console.WriteLine("     + Positional Encoding      0 tham số (bảng sin/cos cố định)");
        Console.WriteLine("        ▼");

        for (int i = 0; i < model.Blocks.Count; i++)
        {
            var block = model.Blocks[i];
            int feedForward = block.FeedForwardIn.Parameters().Sum(p => p.Count)
                              + block.FeedForwardOut.Parameters().Sum(p => p.Count);

            Console.WriteLine($"   ── TransformerBlock {i} ──");
            Console.WriteLine($"        LayerNorm        {block.Norm1.Parameters().Sum(p => p.Count),5} tham số");
            Console.WriteLine($"        Self Attention   {block.Attention.Parameters().Sum(p => p.Count),5} tham số  (Wq, Wk, Wv, Wo)");
            Console.WriteLine($"        + residual");
            Console.WriteLine($"        LayerNorm        {block.Norm2.Parameters().Sum(p => p.Count),5} tham số");
            Console.WriteLine($"        Feed Forward     {feedForward,5} tham số  ({model.ModelDimensions} -> 32 -> {model.ModelDimensions}, ReLU)");
            Console.WriteLine($"        + residual");
            Console.WriteLine("        ▼");
        }

        Console.WriteLine("     LayerNorm cuối");
        Console.WriteLine("        ▼");
        Console.WriteLine($"     Output head                {model.ModelDimensions} -> {model.VocabularySize} (logits)");
        Console.WriteLine("        ▼");
        Console.WriteLine("     softmax                    phân bố xác suất");
        Console.WriteLine($"\n  TỔNG: {model.ParameterCount} tham số.");

        // Gradient check TRƯỚC khi train — quy trình chuẩn từ Phase 3.
        Console.WriteLine("\n  --- Gradient check toàn kiến trúc ---");
        var samples = model.BuildSamples(TextCorpus.Basic).Take(3).ToArray();

        var stopwatch = Stopwatch.StartNew();
        var check = TransformerGradientCheck.Check(model, samples);
        stopwatch.Stop();

        Console.WriteLine($"    Tham số kiểm tra   : {check.Analytic.Length}");
        Console.WriteLine($"    Lệch lớn nhất      : {check.MaxAbsoluteError:E3} (tại #{check.WorstIndex})");
        Console.WriteLine($"    backprop = {check.Analytic[check.WorstIndex]:E6} , số = {check.Numerical[check.WorstIndex]:E6}");
        Console.WriteLine($"    Thời gian          : {stopwatch.Elapsed.TotalSeconds:F2} giây");
        Console.WriteLine($"    Kết quả            : {(check.Passed ? "ĐẠT" : "KHÔNG ĐẠT")}");
        Console.WriteLine("\n  Backprop qua attention (Jacobian softmax, ba nhánh gradient của X) và");
        Console.WriteLine("  qua LayerNorm (hai số hạng hiệu chỉnh vì μ, σ phụ thuộc x) đều đúng.");
    }

    private static MiniTransformer ShowTraining()
    {
        Console.WriteLine("\n\n=== DEMO 4: HUẤN LUYỆN ===\n");

        var model = Build(TextCorpus.Weighted);
        var samples = model.BuildSamples(TextCorpus.Weighted);

        Console.WriteLine($"  Ngữ liệu: {TextCorpus.Weighted.Length} câu -> {samples.Length} mẫu");
        Console.WriteLine($"  \"I like cats\" ×3, \"I like dogs\" ×2, \"I like programming\" ×1\n");

        var stopwatch = Stopwatch.StartNew();

        Console.WriteLine($"{"Epoch",7} | {"Loss",11} | {"Perplexity",11} | \"I like\" -> cats / dogs / programming");
        Console.WriteLine(new string('-', 88));

        model.Train(samples, 1000, (epoch, loss) =>
        {
            if (epoch != 1 && epoch % 200 != 0) return;

            var p = model.Predict("I like");
            var vocabulary = model.Tokenizer.Vocabulary;

            Console.WriteLine($"{epoch,7} | {loss,11:F7} | {SoftmaxCrossEntropy.Perplexity(loss),11:F4} | " +
                              $"{p[vocabulary.IndexOf("cats")]:F4} / {p[vocabulary.IndexOf("dogs")]:F4} / " +
                              $"{p[vocabulary.IndexOf("programming")]:F4}");
        });

        stopwatch.Stop();
        Console.WriteLine($"\n  1000 epoch trong {stopwatch.Elapsed.TotalSeconds:F1} giây.");

        var final = model.Predict("I like");
        var v = model.Tokenizer.Vocabulary;

        Console.WriteLine($"\n  {"từ",-14} {"transformer",12} {"tần suất thật",14}");
        Console.WriteLine("  " + new string('-', 44));
        foreach (var (word, truth) in new[] { ("cats", 0.5), ("dogs", 1.0 / 3), ("programming", 1.0 / 6) })
        {
            Console.WriteLine($"  {word,-14} {final[v.IndexOf(word)],12:F4} {truth,14:F4}");
        }

        return model;
    }

    private static void ShowAttentionPattern(MiniTransformer model)
    {
        Console.WriteLine("\n\n=== DEMO 5: MODEL ĐANG CHÚ Ý VÀO ĐÂU? ===\n");

        var tokens = new List<int> { model.Tokenizer.StartId };
        tokens.AddRange(model.Tokenizer.Encode("I like"));

        model.Forward([.. tokens]);
        var weights = model.LastAttentionWeights!;
        var labels = tokens.Select(model.Tokenizer.Vocabulary.WordAt).ToArray();

        Console.WriteLine("  Trọng số attention của khối 0, khi xử lý \"<s> I like\":\n");
        Console.WriteLine($"  {"token",-8}" + string.Concat(labels.Select(l => $"{l,9}")));
        Console.WriteLine("  " + new string('-', 8 + 9 * labels.Length));

        for (int i = 0; i < labels.Length; i++)
        {
            Console.Write($"  {labels[i],-8}");
            for (int j = 0; j < labels.Length; j++)
            {
                Console.Write($"{weights[i, j],9:F4}");
            }

            Console.WriteLine();
        }

        Console.WriteLine("\n  Tam giác trên bằng 0 nhờ causal mask — token không nhìn được tương lai.");
        Console.WriteLine("  Hàng cuối (\"like\") là hàng quyết định dự đoán từ tiếp theo.");
    }

    private static void ShowGeneration(MiniTransformer model)
    {
        Console.WriteLine("\n\n=== DEMO 6: SINH VĂN BẢN ===\n");

        Console.WriteLine("  Greedy:");
        foreach (string prompt in new[] { "I", "I like", "I love" })
        {
            Console.WriteLine($"    \"{prompt}\" -> \"{prompt} {model.Generate(prompt)}\"");
        }

        Console.WriteLine("\n  Lấy mẫu (T = 1.0), 8 lần với prompt \"I\":");
        var rng = new Random(7);
        var counts = new Dictionary<string, int>();

        for (int i = 0; i < 8; i++)
        {
            string sentence = "I " + model.Generate("I", maxTokens: 6, temperature: 1.0, rng);
            counts[sentence] = counts.GetValueOrDefault(sentence) + 1;
            Console.WriteLine($"    \"{sentence}\"");
        }

        Console.WriteLine("\n  Transformer đã học xong: tokenize -> embedding -> vị trí -> attention");
        Console.WriteLine("  -> FFN -> layer norm -> logits -> softmax -> sinh chữ.");
        Console.WriteLine("  Cùng một kiến trúc với GPT, chỉ khác quy mô.");
    }
}
