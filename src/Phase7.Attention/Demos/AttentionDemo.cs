using Ai.Phase7.Attention;
using Ai.Phase7.LinAlg;
using Ai.Phase7.Training;

namespace Ai.Phase7.Demos;

public static class AttentionDemo
{
    /// <summary>Ví dụ 3 token dùng xuyên suốt demo và unit test.</summary>
    public static (Mat Q, Mat K, Mat V) HandExample() =>
    (
        Mat.FromRows([1, 0], [0, 1], [1, 1]),        // Q
        Mat.FromRows([1, 0], [0, 1], [1, 1]),        // K
        Mat.FromRows([10, 0], [0, 10], [5, 5])       // V
    );

    public static void Run()
    {
        ShowHandCalculation();
        ShowScalingMatters();
        ShowSelfAttentionOnWords();
        ShowCausalMask();
        ShowLearnableHead();
    }

    private static void ShowHandCalculation()
    {
        Console.WriteLine("=== DEMO 1: ATTENTION VỚI 3 TOKEN, TÍNH TAY ===\n");

        var (q, k, v) = HandExample();

        Console.WriteLine("  Q (mỗi hàng = query của một token):");
        Console.WriteLine(q);
        Console.WriteLine("  K (mỗi hàng = key của một token):");
        Console.WriteLine(k);
        Console.WriteLine("  V (mỗi hàng = value của một token):");
        Console.WriteLine(v);

        Console.WriteLine("\n  --- BƯỚC 1: Q · Kᵀ (chấm điểm mọi cặp) ---");
        var raw = q.Multiply(k.Transpose());
        Console.WriteLine(raw);
        Console.WriteLine("    token 0 (q=[1,0]) · k₀=[1,0] = 1 ; · k₁=[0,1] = 0 ; · k₂=[1,1] = 1");
        Console.WriteLine("    token 2 (q=[1,1]) · k₀=[1,0] = 1 ; · k₁=[0,1] = 1 ; · k₂=[1,1] = 2");

        Console.WriteLine($"\n  --- BƯỚC 2: chia cho √d = √2 = {Math.Sqrt(2):F6} ---");
        var result = ScaledDotProductAttention.Compute(q, k, v);
        Console.WriteLine(result.Scores);

        Console.WriteLine("\n  --- BƯỚC 3: softmax từng hàng ---");
        Console.WriteLine(result.Weights);
        Console.WriteLine("    Hàng 0: exp([0.7071, 0, 0.7071]) = [2.0281, 1.0000, 2.0281], tổng = 5.0562");
        Console.WriteLine("            -> [2.0281/5.0562, 1/5.0562, 2.0281/5.0562] = [0.4011, 0.1978, 0.4011]");

        for (int i = 0; i < result.Weights.Rows; i++)
        {
            Console.WriteLine($"    Tổng hàng {i} = {result.Weights.Row(i).Sum():F10}");
        }

        Console.WriteLine("\n  --- BƯỚC 4: nhân với V (trộn nội dung theo trọng số) ---");
        Console.WriteLine(result.Output);
        Console.WriteLine("    Hàng 0 = 0.4011·[10,0] + 0.1978·[0,10] + 0.4011·[5,5]");
        Console.WriteLine("           = [4.0111, 0] + [0, 1.9778] + [2.0056, 2.0056]");
        Console.WriteLine("           = [6.0167, 3.9833]");

        Console.WriteLine("\n  Token 0 và token 2 có query giống nhau về hướng nên nhận nhiều chú ý;");
        Console.WriteLine("  token 1 vuông góc với query của token 0 nên chỉ nhận 19.8%.");
    }

    private static void ShowScalingMatters()
    {
        Console.WriteLine("\n\n=== DEMO 2: VÌ SAO PHẢI CHIA CHO √d ===\n");

        var rng = new Random(42);

        Console.WriteLine($"  {"d",5} | {"trọng số lớn nhất (có chia √d)",32} | {"(không chia)",22}");
        Console.WriteLine("  " + new string('-', 66));

        foreach (int d in new[] { 2, 8, 32, 128, 512 })
        {
            var q = RandomMatrix(4, d, rng);
            var k = RandomMatrix(4, d, rng);
            var v = RandomMatrix(4, 2, rng);

            var scaled = ScaledDotProductAttention.Compute(q, k, v);
            double maxScaled = MaxWeight(scaled.Weights);

            // Mô phỏng phiên bản KHÔNG chia: nhân ngược lại √d rồi softmax lại.
            var unscaledWeights = new Mat(4, 4);
            for (int i = 0; i < 4; i++)
            {
                unscaledWeights.SetRow(i, Softmax.Apply(scaled.Scores.Row(i).Scale(Math.Sqrt(d))));
            }

            Console.WriteLine($"  {d,5} | {maxScaled,32:F4} | {MaxWeight(unscaledWeights),22:F4}");
        }

        Console.WriteLine("\n  Không chia √d: d càng lớn thì trọng số càng dồn hết vào MỘT token");
        Console.WriteLine("  (tiến về 1.0000). Softmax bão hoà -> gradient của các token khác");
        Console.WriteLine("  bằng 0 -> mô hình ngừng học. Chia cho √d giữ phân bố mềm.");
    }

    private static void ShowSelfAttentionOnWords()
    {
        Console.WriteLine("\n\n=== DEMO 3: SELF-ATTENTION TRÊN MỘT CÂU ===\n");

        // Embedding dựng tay: "cat" và "kitten" gần nhau, "car" khác hẳn.
        string[] tokens = ["cat", "kitten", "car", "engine"];
        var x = Mat.FromRows(
            [1.0, 0.1, 0.0],     // cat
            [0.9, 0.2, 0.0],     // kitten  <- gần cat
            [0.0, 0.1, 1.0],     // car
            [0.1, 0.0, 0.9]);    // engine  <- gần car

        var result = ScaledDotProductAttention.SelfAttention(x);

        Console.WriteLine("  Self-attention: Q = K = V = X (mỗi token tự sinh cả ba vai trò)\n");
        Console.WriteLine($"  {"token",-9}" + string.Concat(tokens.Select(t => $"{t,9}")));
        Console.WriteLine("  " + new string('-', 9 + 9 * tokens.Length));

        for (int i = 0; i < tokens.Length; i++)
        {
            Console.Write($"  {tokens[i],-9}");
            for (int j = 0; j < tokens.Length; j++)
            {
                Console.Write($"{result.Weights[i, j],9:F4}");
            }

            Console.WriteLine();
        }

        Console.WriteLine("\n  Đọc bảng: hàng i cho biết token i phân bổ chú ý cho các token nào.");
        Console.WriteLine($"  'cat' chú ý 'kitten' {result.Weights[0, 1]:P1} nhưng chỉ chú ý 'car' {result.Weights[0, 2]:P1}.");
        Console.WriteLine("  Không ai lập trình điều đó — nó sinh ra từ tích vô hướng của embedding.");
    }

    private static void ShowCausalMask()
    {
        Console.WriteLine("\n\n=== DEMO 4: CAUSAL MASK (che tương lai) ===\n");

        string[] tokens = ["I", "like", "cats", "</s>"];
        var x = Mat.FromRows(
            [1.0, 0.2],
            [0.3, 1.0],
            [0.8, 0.5],
            [0.1, 0.1]);

        var masked = ScaledDotProductAttention.SelfAttention(x, causalMask: true);

        Console.WriteLine($"  {"token",-8}" + string.Concat(tokens.Select(t => $"{t,9}")));
        Console.WriteLine("  " + new string('-', 8 + 9 * tokens.Length));

        for (int i = 0; i < tokens.Length; i++)
        {
            Console.Write($"  {tokens[i],-8}");
            for (int j = 0; j < tokens.Length; j++)
            {
                Console.Write($"{masked.Weights[i, j],9:F4}");
            }

            Console.WriteLine();
        }

        Console.WriteLine("\n  Tam giác trên toàn 0: token không được nhìn về phía sau nó.");
        Console.WriteLine("  Cài bằng cách đặt score = −∞ trước softmax, vì e^(−∞) = 0.");
        Console.WriteLine("\n  Bắt buộc phải có khi huấn luyện mô hình sinh văn bản: nếu token");
        Console.WriteLine("  'like' được nhìn thấy 'cats', nó chỉ việc chép đáp án và mô hình");
        Console.WriteLine("  không học được gì.");
    }

    private static void ShowLearnableHead()
    {
        Console.WriteLine("\n\n=== DEMO 5: ĐẦU ATTENTION CÓ THAM SỐ HỌC ĐƯỢC ===\n");

        var head = new AttentionHead(modelDimensions: 3, headDimensions: 2, new Random(42));
        var x = Mat.FromRows([1.0, 0.1, 0.0], [0.9, 0.2, 0.0], [0.0, 0.1, 1.0]);

        Console.WriteLine($"  Model {head.ModelDimensions} chiều -> head {head.HeadDimensions} chiều, " +
                          $"{head.ParameterCount} tham số (3 ma trận {head.ModelDimensions}×{head.HeadDimensions})\n");

        Console.WriteLine("  Wq:");
        Console.WriteLine(head.Wq);
        Console.WriteLine("  Wk:");
        Console.WriteLine(head.Wk);
        Console.WriteLine("  Wv:");
        Console.WriteLine(head.Wv);

        Console.WriteLine("\n  Q = X·Wq, K = X·Wk, V = X·Wv rồi mới attention:");
        var result = head.Forward(x);
        Console.WriteLine("  Trọng số attention:");
        Console.WriteLine(result.Weights);
        Console.WriteLine("  Đầu ra:");
        Console.WriteLine(result.Output);

        Console.WriteLine("\n  Ba ma trận này chưa được train nên chưa có ý nghĩa gì.");
        Console.WriteLine("  Phase 8 sẽ đặt đầu attention vào một transformer hoàn chỉnh và học chúng.");
    }

    private static Mat RandomMatrix(int rows, int cols, Random rng)
    {
        var m = new Mat(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                m[r, c] = rng.NextDouble() * 2 - 1;
            }
        }

        return m;
    }

    private static double MaxWeight(Mat weights)
    {
        double max = 0;
        for (int r = 0; r < weights.Rows; r++)
        {
            for (int c = 0; c < weights.Cols; c++)
            {
                max = Math.Max(max, weights[r, c]);
            }
        }

        return max;
    }
}
