using Ai.Phase11.Agent;
using Ai.Phase11.Memory;

namespace Ai.Phase11.Demos;

public static class AgentDemo
{
    public static void Run()
    {
        var classifier = ShowClassifierTraining();
        ShowArchitecture();
        var agent = ShowConversation();
        ShowDecisionTrace(agent);
        ShowHybridAdvantage(agent);
        ShowLimits(classifier);
    }

    private static IntentClassifier ShowClassifierTraining()
    {
        Console.WriteLine("=== DEMO 1: HUẤN LUYỆN BỘ PHÂN LOẠI Ý ĐỊNH ===\n");

        var classifier = new IntentClassifier(IntentTrainingData.Texts, new Random(42));

        Console.WriteLine($"  Ví dụ có nhãn : {IntentTrainingData.Examples.Length}");
        Console.WriteLine($"  Từ điển       : {classifier.Vocabulary.Count} từ");
        Console.WriteLine($"  Kiến trúc     : bag-of-words({classifier.Vocabulary.Count}) -> 16 tanh -> 6 softmax");
        Console.WriteLine($"  Tham số       : {classifier.ParameterCount}\n");

        Console.WriteLine($"{"Epoch",7} | {"Loss",11} | accuracy");
        Console.WriteLine(new string('-', 35));

        classifier.Train(IntentTrainingData.Examples, 3000, (epoch, loss) =>
        {
            if (epoch != 1 && epoch % 500 != 0) return;

            Console.WriteLine($"{epoch,7} | {loss,11:F7} | {classifier.Accuracy(IntentTrainingData.Examples),8:P0}");
        });

        Console.WriteLine("\n  Đây là mạng neural của Phase 2–6 dùng lại nguyên vẹn: layer, tanh,");
        Console.WriteLine("  softmax, cross entropy, backpropagation. Chỉ khác bài toán.");

        Console.WriteLine("\n  Thử với câu CHƯA từng thấy trong tập train:");
        string[] unseen =
        [
            "nhớ giúp tôi rằng tôi thích Gundam",
            "tính 2 nhân 8",
            "attention là gì",
            "có bao nhiêu khách hàng",
            "chào",
        ];

        foreach (string text in unseen)
        {
            var prediction = classifier.Classify(text);
            Console.WriteLine($"    \"{text,-38}\" -> {prediction.Intent,-16} ({prediction.Confidence:P0})");
        }

        return classifier;
    }

    private static void ShowArchitecture()
    {
        Console.WriteLine("\n\n=== DEMO 2: KIẾN TRÚC AGENT ===\n");
        Console.WriteLine("                        User");
        Console.WriteLine("                          ↓");
        Console.WriteLine("                    AI Controller          <- phân loại ý định (neural, Phase 2-6)");
        Console.WriteLine("                          ↓");
        Console.WriteLine("                  ┌───────┴───────┐");
        Console.WriteLine("                  ↓               ↓");
        Console.WriteLine("               Memory         Reasoning     <- Phase 10 / Phase 9");
        Console.WriteLine("                  ↓               ↓");
        Console.WriteLine("                  └───────┬───────┘");
        Console.WriteLine("                          ↓");
        Console.WriteLine("                        Tools");
        Console.WriteLine("                  ┌───────┼───────┐");
        Console.WriteLine("                  ↓       ↓       ↓");
        Console.WriteLine("             Calculator Search  Database");
        Console.WriteLine("                          ↓");
        Console.WriteLine("                       Result");
        Console.WriteLine("                          ↓");
        Console.WriteLine("                    AI Response");

        var agent = AgentFactory.Build(epochs: 1);
        Console.WriteLine("\n  Công cụ đã đăng ký:");
        foreach (var tool in agent.Tools)
        {
            Console.WriteLine($"    {tool.Name,-12} {tool.Description}");
        }
    }

    private static AiController ShowConversation()
    {
        Console.WriteLine("\n\n=== DEMO 3: HỘI THOẠI ===\n");

        var clock = new ManualClock(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));
        var agent = AgentFactory.Build(clock: clock);

        string[] conversation =
        [
            "xin chào",
            "hãy nhớ rằng tôi dùng C# và dotnet",
            "ghi nhớ là tôi làm việc với Azure",
            "hãy nhớ tôi thích Gundam",
            "tôi dùng ngôn ngữ nào",
            "bạn nhớ gì về tôi",
            "tính 2 + 3 * 4",
            "C bằng bao nhiêu",
            "backpropagation là gì",
            "có bao nhiêu đơn hàng",
            "doanh thu tổng cộng là bao nhiêu",
            "đơn hàng lớn nhất là đơn nào",
            "bạn làm được những gì",
        ];

        foreach (string input in conversation)
        {
            var response = agent.Handle(input);

            Console.WriteLine($"  NGƯỜI DÙNG: {input}");
            Console.WriteLine($"  AGENT     : {response.Answer}");
            Console.WriteLine($"              [{response.Intent}, tự tin {response.Confidence:P0}" +
                              $"{(response.ToolUsed is null ? "" : $", công cụ: {response.ToolUsed}")}]");
            Console.WriteLine();

            clock.Advance(TimeSpan.FromMinutes(2));
        }

        return agent;
    }

    private static void ShowDecisionTrace(AiController agent)
    {
        Console.WriteLine("\n=== DEMO 4: AGENT RA QUYẾT ĐỊNH NHƯ THẾ NÀO ===\n");

        foreach (string input in new[] { "tính 2 + 3 * 4", "tôi dùng ngôn ngữ nào", "xin chào" })
        {
            var response = agent.Handle(input);

            Console.WriteLine($"  \"{input}\"");
            foreach (var step in response.Decisions)
            {
                Console.WriteLine($"     {step.Question,-38} -> {step.Decision}");
            }

            Console.WriteLine($"     {"KẾT QUẢ",-38} -> {response.Answer}");

            if (response.Detail is not null)
            {
                Console.WriteLine($"     chi tiết: {response.Detail}");
            }

            Console.WriteLine();
        }
    }

    private static void ShowHybridAdvantage(AiController agent)
    {
        Console.WriteLine("\n=== DEMO 5: VÌ SAO PHẢI GHÉP NEURAL VỚI KÝ HIỆU ===\n");

        Console.WriteLine("  Agent tính toán CHÍNH XÁC, vì nó giao việc cho reasoning engine:");
        foreach (string question in new[] { "tính 2 + 3 * 4", "tính 100 / 8", "tính C - A" })
        {
            Console.WriteLine($"    \"{question,-18}\" -> {agent.Handle(question).Answer}");
        }

        Console.WriteLine("\n  Mạng neural đơn độc sẽ nội suy ra 13.97 thay vì 14, và càng sai hơn");
        Console.WriteLine("  với số chưa từng gặp. Nó không tính — nó khớp mẫu.");

        Console.WriteLine("\n  Ngược lại, reasoning engine đơn độc không hiểu nổi cách diễn đạt mới.");
        Console.WriteLine("  Cả ba câu dưới đây đều được mạng neural quy về cùng một ý định:");
        foreach (string phrasing in new[] { "có bao nhiêu đơn hàng", "đếm số đơn", "how many orders are there" })
        {
            var response = agent.Handle(phrasing);
            Console.WriteLine($"    \"{phrasing,-34}\" -> {response.Intent} -> {response.Answer}");
        }

        Console.WriteLine("\n  NEURAL lo phần mơ hồ. KÝ HIỆU lo phần chính xác. Đó là toàn bộ ý tưởng.");

        Console.WriteLine("\n  Agent cũng nhớ được qua nhiều lượt — thứ mà context window không làm được:");
        Console.WriteLine($"    Bộ nhớ dài hạn hiện có {agent.Memory.LongTerm.Count} mẩu:");
        foreach (var fact in agent.Memory.LongTerm.Facts)
        {
            Console.WriteLine($"      • {fact.Content}");
        }
    }

    private static void ShowLimits(IntentClassifier classifier)
    {
        Console.WriteLine("\n\n=== DEMO 6: AGENT BIẾT KHI NÀO NÓ KHÔNG BIẾT ===\n");

        var agent = AgentFactory.Build();

        string[] hard =
        [
            "zxcvbnm qwerty",
            "làm thế nào để nấu phở",
            "tính 5 chia 0",
            "Z bằng bao nhiêu",
        ];

        foreach (string input in hard)
        {
            var response = agent.Handle(input);
            Console.WriteLine($"  \"{input}\"");
            Console.WriteLine($"     nhận ra {classifier.KnownWordCount(input)} từ quen, " +
                              $"ý định {response.Intent} ({response.Confidence:P0})");
            Console.WriteLine($"     -> {response.Answer}\n");
        }

        Console.WriteLine("  Không nhận ra từ nào -> độ tự tin về 0 -> agent nói thẳng là không chắc,");
        Console.WriteLine("  thay vì gọi bừa một công cụ. Đây là hành vi cố ý thiết kế: thà nói");
        Console.WriteLine("  \"tôi không biết\" còn hơn đưa một đáp án bịa ra một cách tự tin.");
    }
}
