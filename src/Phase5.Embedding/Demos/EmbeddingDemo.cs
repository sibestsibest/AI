using Ai.Phase5.Embeddings;
using Ai.Phase5.LinAlg;
using Ai.Phase5.Training;

namespace Ai.Phase5.Demos;

public static class EmbeddingDemo
{
    private static readonly string[] Focus = ["cat", "dog", "car", "apple"];

    public static EmbeddingTable TrainEmbeddings(int seed = 42, int dimensions = 8, int epochs = 3000)
    {
        var vocabulary = Corpus.BuildVocabulary();
        var rng = new Random(seed);
        var model = new SkipGramModel(vocabulary, dimensions, rng, new StochasticGradientDescent(0.5));
        var pairs = Corpus.BuildPairs(vocabulary, rng);

        model.Train(pairs, epochs);

        return model.Center;
    }

    public static void Run()
    {
        ShowOneHotProblem();
        ShowCosineByHand();
        var table = ShowTraining();
        ShowSimilarities(table);
        ShowNeighbours(table);
    }

    /// <summary>Vì sao one-hot không đủ.</summary>
    private static void ShowOneHotProblem()
    {
        Console.WriteLine("=== DEMO 1: VẤN ĐỀ CỦA ONE-HOT ===\n");

        var vocabulary = Corpus.BuildVocabulary();
        Console.WriteLine($"  Từ điển có {vocabulary.Count} từ: {string.Join(", ", vocabulary.Words)}\n");

        Console.WriteLine("  Biểu diễn one-hot (mỗi từ một chiều riêng):");
        foreach (string word in Focus)
        {
            var oneHot = Similarity.OneHot(vocabulary.IndexOf(word), vocabulary.Count);
            Console.WriteLine($"    {word,-6} -> {oneHot}");
        }

        Console.WriteLine("\n  Cosine similarity giữa các vector one-hot:");
        for (int i = 0; i < Focus.Length; i++)
        {
            for (int j = i + 1; j < Focus.Length; j++)
            {
                var a = Similarity.OneHot(vocabulary.IndexOf(Focus[i]), vocabulary.Count);
                var b = Similarity.OneHot(vocabulary.IndexOf(Focus[j]), vocabulary.Count);
                Console.WriteLine($"    cos({Focus[i]}, {Focus[j]}) = {Similarity.Cosine(a, b):F4}");
            }
        }

        Console.WriteLine("\n  TẤT CẢ đều bằng 0. One-hot nói được 'cat khác dog' nhưng KHÔNG nói");
        Console.WriteLine("  được 'cat gần dog hơn car'. Mọi từ cách đều mọi từ.");
        Console.WriteLine($"  Thêm nữa: cần {vocabulary.Count} chiều cho {vocabulary.Count} từ — từ điển thật có 50.000 từ.");
    }

    /// <summary>Cosine similarity tính tay.</summary>
    private static void ShowCosineByHand()
    {
        Console.WriteLine("\n\n=== DEMO 2: COSINE SIMILARITY TÍNH TAY ===\n");

        (Vec A, Vec B, string Note)[] cases =
        [
            (Vec.Of(1, 0), Vec.Of(1, 0), "cùng hướng hoàn toàn"),
            (Vec.Of(3, 4), Vec.Of(6, 8), "cùng hướng, khác độ dài"),
            (Vec.Of(1, 0), Vec.Of(1, 1), "lệch 45°"),
            (Vec.Of(1, 0), Vec.Of(0, 1), "vuông góc"),
            (Vec.Of(1, 0), Vec.Of(-1, 0), "ngược hướng"),
        ];

        Console.WriteLine($"  {"a",-10} {"b",-10} {"a·b",8} {"‖a‖",8} {"‖b‖",8} {"cosine",9}   ghi chú");
        Console.WriteLine("  " + new string('-', 76));

        foreach (var (a, b, note) in cases)
        {
            Console.WriteLine($"  {a,-10} {b,-10} {a.Dot(b),8:F2} {a.Norm(),8:F4} {b.Norm(),8:F4} " +
                              $"{Similarity.Cosine(a, b),9:F4}   {note}");
        }

        Console.WriteLine("\n  Hàng 2: [3,4]·[6,8] = 18+32 = 50 ; ‖a‖=5 ; ‖b‖=10 ; 50/(5·10) = 1.0");
        Console.WriteLine("  Độ dài khác nhau nhưng cùng hướng -> similarity vẫn bằng 1.");
    }

    private static EmbeddingTable ShowTraining()
    {
        Console.WriteLine("\n\n=== DEMO 3: HỌC EMBEDDING TỪ NGỮ CẢNH ===\n");

        var vocabulary = Corpus.BuildVocabulary();
        var rng = new Random(42);
        var model = new SkipGramModel(vocabulary, dimensions: 8, rng, new StochasticGradientDescent(0.5));
        var table = model.Center;
        var pairs = Corpus.BuildPairs(vocabulary, rng);

        Console.WriteLine($"  Ngữ liệu: {Corpus.Sentences.Length} dòng, {vocabulary.Count} từ, {table.Dimensions} chiều");
        Console.WriteLine($"  Cặp huấn luyện: {pairs.Count(p => p.Label == 1)} cặp dương, " +
                          $"{pairs.Count(p => p.Label == 0)} cặp âm");
        Console.WriteLine($"  Tham số cần học: 2 bảng × {vocabulary.Count} từ × {table.Dimensions} chiều = " +
                          $"{2 * vocabulary.Count * table.Dimensions}");
        Console.WriteLine("  (chỉ bảng 'center' được giữ lại; bảng 'context' bỏ đi sau khi train)\n");

        Console.WriteLine("  Trước khi train (vector ngẫu nhiên):");
        Console.WriteLine($"    cos(cat, dog) = {Similarity.Cosine(table.Lookup("cat"), table.Lookup("dog")):F4}");
        Console.WriteLine($"    cos(cat, car) = {Similarity.Cosine(table.Lookup("cat"), table.Lookup("car")):F4}\n");

        Console.WriteLine($"{"Epoch",7} | {"Loss",12} | {"cos(cat,dog)",13} | {"cos(cat,car)",13}");
        Console.WriteLine(new string('-', 56));

        model.Train(pairs, 3000, (epoch, loss) =>
        {
            if (epoch != 1 && epoch % 500 != 0) return;

            Console.WriteLine($"{epoch,7} | {loss,12:F8} | " +
                              $"{Similarity.Cosine(table.Lookup("cat"), table.Lookup("dog")),13:F6} | " +
                              $"{Similarity.Cosine(table.Lookup("cat"), table.Lookup("car")),13:F6}");
        });

        Console.WriteLine("\n  Vector học được của 4 từ chính:");
        foreach (string word in Focus)
        {
            Console.WriteLine($"    {word,-6} -> {table.Lookup(word)}");
        }

        return table;
    }

    private static void ShowSimilarities(EmbeddingTable table)
    {
        Console.WriteLine("\n\n=== DEMO 4: MA TRẬN SIMILARITY ===\n");

        Console.WriteLine($"  {"",-8}" + string.Concat(Focus.Select(w => $"{w,9}")));
        Console.WriteLine("  " + new string('-', 8 + 9 * Focus.Length));

        foreach (string row in Focus)
        {
            Console.Write($"  {row,-8}");
            foreach (string col in Focus)
            {
                Console.Write($"{Similarity.Cosine(table.Lookup(row), table.Lookup(col)),9:F4}");
            }

            Console.WriteLine();
        }

        double catDog = Similarity.Cosine(table.Lookup("cat"), table.Lookup("dog"));
        double catCar = Similarity.Cosine(table.Lookup("cat"), table.Lookup("car"));

        Console.WriteLine($"\n  cos(cat, dog) = {catDog:F4}   <- cùng ngữ cảnh: pet, animal, fur, tail");
        Console.WriteLine($"  cos(cat, car) = {catCar:F4}   <- không chung ngữ cảnh nào");
        Console.WriteLine($"\n  Model chưa bao giờ được dạy 'mèo và chó đều là động vật'.");
        Console.WriteLine("  Nó chỉ thấy các từ này xuất hiện cạnh cùng một nhóm từ khác.");
    }

    private static void ShowNeighbours(EmbeddingTable table)
    {
        Console.WriteLine("\n\n=== DEMO 5: TỪ GẦN NHẤT ===\n");

        foreach (string word in Focus)
        {
            var neighbours = Similarity.NearestNeighbours(table, word, take: 3);
            string list = string.Join(", ", neighbours.Select(n => $"{n.Word} ({n.Score:F3})"));
            Console.WriteLine($"  {word,-6} ~  {list}");
        }

        Console.WriteLine("\n  Đây chính là cơ chế đằng sau tìm kiếm ngữ nghĩa và RAG:");
        Console.WriteLine("  nhúng câu hỏi thành vector, rồi tìm các vector gần nhất.");
    }
}
