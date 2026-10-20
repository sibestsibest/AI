using Ai.Phase6.Embeddings;
using Ai.Phase6.Language;
using Ai.Phase6.LinAlg;
using Ai.Phase6.Training;

namespace Ai.Phase6.Demos;

public static class LanguageModelDemo
{
    public static LanguageModel Build(string[] corpus, int seed = 42)
    {
        var tokenizer = new Tokenizer(corpus);
        return new LanguageModel(
            tokenizer,
            contextSize: 2,
            embeddingDimensions: 12,
            hiddenSize: 24,
            new Random(seed),
            new StochasticGradientDescent(0.5));
    }

    public static LanguageModel Train(string[] corpus, int epochs = 4000, int seed = 42)
    {
        var model = Build(corpus, seed);
        model.Train(model.BuildSamples(corpus), epochs);
        return model;
    }

    public static void Run()
    {
        ShowTokenization();
        ShowLogitsAndSoftmax();
        var model = ShowTraining();
        ShowPredictions(model);
        ShowWeightedCorpus();
        ShowGeneration(model);
    }

    private static void ShowTokenization()
    {
        Console.WriteLine("=== DEMO 1: TOKENIZE ===\n");

        var tokenizer = new Tokenizer(TextCorpus.Basic);

        Console.WriteLine($"  Ngữ liệu ({TextCorpus.Basic.Length} câu):");
        foreach (string sentence in TextCorpus.Basic)
        {
            Console.WriteLine($"    \"{sentence}\"");
        }

        Console.WriteLine($"\n  Từ điển ({tokenizer.Vocabulary.Count} token):");
        for (int i = 0; i < tokenizer.Vocabulary.Count; i++)
        {
            Console.WriteLine($"    {i} -> {tokenizer.Vocabulary.WordAt(i)}");
        }

        Console.WriteLine("\n  Mã hoá:");
        foreach (string sentence in TextCorpus.Basic.Take(2))
        {
            var ids = tokenizer.Encode(sentence);
            Console.WriteLine($"    \"{sentence}\" -> [{string.Join(", ", ids)}] -> \"{tokenizer.Decode(ids)}\"");
        }

        Console.WriteLine("\n  Bọc token biên (contextSize = 2):");
        var bounded = tokenizer.EncodeWithBoundaries("I like cats", contextSize: 2);
        Console.WriteLine($"    [{string.Join(", ", bounded)}] = \"{tokenizer.Decode(bounded)}\"");

        Console.WriteLine("\n  Mẫu huấn luyện sinh ra (ngữ cảnh -> từ tiếp theo):");
        var model = Build(TextCorpus.Basic);
        foreach (var sample in model.BuildSamples(TextCorpus.Basic).Take(8))
        {
            Console.WriteLine($"    [{model.Tokenizer.Decode(sample.Context),-18}] -> " +
                              $"{model.Tokenizer.Vocabulary.WordAt(sample.Target)}");
        }
    }

    private static void ShowLogitsAndSoftmax()
    {
        Console.WriteLine("\n\n=== DEMO 2: LOGITS -> SOFTMAX -> XÁC SUẤT ===\n");

        var logits = Vec.Of(1, 2, 3);
        var probabilities = Softmax.Apply(logits);

        Console.WriteLine($"  logits = {logits}");
        Console.WriteLine($"  e^1 = {Math.Exp(1):F4} , e^2 = {Math.Exp(2):F4} , e^3 = {Math.Exp(3):F4}");
        Console.WriteLine($"  tổng = {Math.Exp(1) + Math.Exp(2) + Math.Exp(3):F4}");
        Console.WriteLine($"  softmax = {probabilities}");
        Console.WriteLine($"  tổng xác suất = {probabilities.Sum():F10}  (luôn bằng 1)");
        Console.WriteLine("\n  Logits chênh nhau 1 và 2 đơn vị, nhưng xác suất chênh 7.4 lần —");
        Console.WriteLine("  hàm e^x khuếch đại khoảng cách. Đó là ý nghĩa của 'soft-MAX'.");

        Console.WriteLine("\n  Cross entropy khi đáp án đúng là phần tử thứ 3:");
        var loss = new SoftmaxCrossEntropy();
        Console.WriteLine($"    L = −log(p₃) = −log({probabilities[2]:F4}) = " +
                          $"{loss.Compute(logits, Vec.Of(0, 0, 1)):F6}");
        Console.WriteLine($"    ∂L/∂logits = p − y = {loss.Gradient(logits, Vec.Of(0, 0, 1))}");
        Console.WriteLine("    -> phần tử đúng bị đẩy LÊN (gradient âm), các phần tử sai bị đẩy XUỐNG.");

        Console.WriteLine("\n  Loss thay đổi theo mức tự tin vào đáp án đúng:");
        Console.WriteLine($"  {"p(đúng)",10} {"Loss",10} {"Perplexity",12}");
        foreach (double p in new[] { 0.9, 0.5, 0.333, 0.1, 0.01 })
        {
            double l = -Math.Log(p);
            Console.WriteLine($"  {p,10:F3} {l,10:F4} {SoftmaxCrossEntropy.Perplexity(l),12:F2}");
        }

        Console.WriteLine("\n  Perplexity = 'model đang lưỡng lự giữa bao nhiêu phương án'.");
        Console.WriteLine("  Nhiệt độ khi sinh văn bản (cùng logits [1,2,3]):");
        foreach (double t in new[] { 0.1, 0.5, 1.0, 2.0, 10.0 })
        {
            Console.WriteLine($"    T = {t,4}  ->  {Softmax.Apply(logits, t)}");
        }
    }

    private static LanguageModel ShowTraining()
    {
        Console.WriteLine("\n\n=== DEMO 3: HUẤN LUYỆN MÔ HÌNH NGÔN NGỮ ===\n");

        var model = Build(TextCorpus.Basic);
        var samples = model.BuildSamples(TextCorpus.Basic);

        Console.WriteLine($"  Từ điển        : {model.VocabularySize} token");
        Console.WriteLine($"  Cửa sổ ngữ cảnh: {model.ContextSize} token");
        Console.WriteLine($"  Embedding      : {model.Embeddings.Dimensions} chiều");
        Console.WriteLine($"  Mẫu huấn luyện : {samples.Length}");
        Console.WriteLine($"  Tổng tham số   : {model.ParameterCount}");
        Console.WriteLine($"    embedding: {model.Embeddings.Count} × {model.Embeddings.Dimensions} = " +
                          $"{model.Embeddings.Count * model.Embeddings.Dimensions}");
        Console.WriteLine($"    mạng     : {model.Network.ParameterCount}\n");

        Console.WriteLine($"{"Epoch",7} | {"Loss",11} | {"Perplexity",11} | \"I like\" -> cats / dogs / programming");
        Console.WriteLine(new string('-', 88));

        model.Train(samples, 4000, (epoch, loss) =>
        {
            if (epoch != 1 && epoch % 500 != 0) return;

            var p = model.Predict("I like");
            var vocabulary = model.Tokenizer.Vocabulary;

            Console.WriteLine($"{epoch,7} | {loss,11:F7} | {SoftmaxCrossEntropy.Perplexity(loss),11:F4} | " +
                              $"{p[vocabulary.IndexOf("cats")]:F4} / {p[vocabulary.IndexOf("dogs")]:F4} / " +
                              $"{p[vocabulary.IndexOf("programming")]:F4}");
        });

        return model;
    }

    private static void ShowPredictions(LanguageModel model)
    {
        Console.WriteLine("\n\n=== DEMO 4: DỰ ĐOÁN TỪ TIẾP THEO ===\n");

        foreach (string context in new[] { "I like", "I love", "<s> I", "like cats" })
        {
            Console.WriteLine($"  Ngữ cảnh: \"{context}\"");
            foreach (var (word, probability) in model.TopPredictions(context, take: 4))
            {
                string bar = new('█', (int)Math.Round(probability * 40));
                Console.WriteLine($"    {word,-14} {probability,8:F4}  {bar}");
            }

            Console.WriteLine();
        }

        Console.WriteLine("  Phân bố THẬT trong ngữ liệu:");
        Console.WriteLine("    sau \"I like\": cats 1/3, dogs 1/3, programming 1/3  -> model phải ra ≈ 0.333 mỗi từ");
        Console.WriteLine("    sau \"I love\": cats 1/2, programming 1/2            -> model phải ra ≈ 0.500 mỗi từ");
        Console.WriteLine("\n  Model không 'chọn một từ' — nó học đúng PHÂN BỐ tần suất của ngữ liệu.");
    }

    private static void ShowWeightedCorpus()
    {
        Console.WriteLine("\n\n=== DEMO 5: MODEL HỌC TẦN SUẤT, KHÔNG CHỈ HỌC TẬP HỢP ===\n");

        var model = Train(TextCorpus.Weighted);
        var vocabulary = model.Tokenizer.Vocabulary;
        var p = model.Predict("I like");

        Console.WriteLine("  Ngữ liệu có \"I like cats\" ×3, \"I like dogs\" ×2, \"I like programming\" ×1\n");
        Console.WriteLine($"  {"từ",-14} {"model dự đoán",14} {"tần suất thật",14}");
        Console.WriteLine("  " + new string('-', 46));

        (string Word, double Truth)[] expectations =
        [
            ("cats", 3.0 / 6),
            ("dogs", 2.0 / 6),
            ("programming", 1.0 / 6),
        ];

        foreach (var (word, truth) in expectations)
        {
            Console.WriteLine($"  {word,-14} {p[vocabulary.IndexOf(word)],14:F4} {truth,14:F4}");
        }

        Console.WriteLine("\n  Đây là điều một LLM thật sự làm: học phân bố xác suất của ngôn ngữ.");
    }

    private static void ShowGeneration(LanguageModel model)
    {
        Console.WriteLine("\n\n=== DEMO 6: SINH VĂN BẢN ===\n");

        Console.WriteLine("  Greedy (luôn chọn từ có xác suất cao nhất):");
        foreach (string prompt in new[] { "I", "I like", "I love" })
        {
            Console.WriteLine($"    \"{prompt}\" -> \"{prompt} {model.Generate(prompt, maxTokens: 6)}\"");
        }

        Console.WriteLine("\n  Lấy mẫu theo xác suất (T = 1.0), 6 lần với prompt \"I\":");
        var rng = new Random(7);
        for (int i = 0; i < 6; i++)
        {
            Console.WriteLine($"    \"I {model.Generate("I", maxTokens: 6, temperature: 1.0, rng)}\"");
        }

        Console.WriteLine("\n  Greedy luôn ra cùng một câu; lấy mẫu thì đa dạng. Mọi LLM đều");
        Console.WriteLine("  hoạt động thế này: dự đoán một token, nối vào, lặp lại.");

        Console.WriteLine("\n  Embedding học được (phụ phẩm của việc dự đoán từ tiếp theo):");
        foreach (string word in new[] { "like", "cats" })
        {
            var neighbours = Similarity.NearestNeighbours(model.Embeddings, word, take: 2);
            Console.WriteLine($"    {word,-6} ~ {string.Join(", ", neighbours.Select(n => $"{n.Word} ({n.Score:F3})"))}");
        }

        Console.WriteLine("\n  \"like\" và \"love\" gần nhau vì chúng xuất hiện ở cùng vị trí ngữ pháp —");
        Console.WriteLine("  không ai dạy model điều đó, nó là hệ quả của mục tiêu dự đoán từ.");
    }
}
