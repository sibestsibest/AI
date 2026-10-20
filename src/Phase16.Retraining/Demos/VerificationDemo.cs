using Ai.Phase16.Agent;
using Ai.Phase16.Grounding;
using Ai.Phase16.Internet;
using Ai.Phase16.Memory;
using Ai.Phase16.Verification;

namespace Ai.Phase16.Demos;

/// <summary>Demo riêng cho Phase 14 — Verification + Answer Generation.</summary>
public static class VerificationDemo
{
    public static async Task RunAsync()
    {
        ShowPipeline();
        await ShowGroundedAnswerAsync();
        await ShowInsufficientAsync();
        ShowContradiction();
        await ShowConflictedAnswerAsync();
        ShowValidator();
        await ShowFactVsInferenceAsync();
    }

    private static ManualClock Clock() => new(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));

    private static void ShowPipeline()
    {
        Console.WriteLine("\n\n=== DEMO 13: ĐƯỜNG ỐNG TRẢ LỜI CÓ KIỂM CHỨNG ===\n");

        Console.WriteLine("     Câu hỏi");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Retrieve + Collect Evidence     EvidenceCollector (Phase 13, 5 kênh)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Reason                          ReasoningEngine (Phase 9)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Verify                          ContradictionDetector + ConfidenceEstimator");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Generate Answer                 AnswerGenerator (theo khuôn, từ căn cứ)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Validate                        AnswerValidator (chốt chặn cuối)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     GroundedAnswer");

        Console.WriteLine("\n  Bốn kết luận có thể ra:");
        Console.WriteLine("    Answered      căn cứ đủ và nhất quán -> trả lời");
        Console.WriteLine("    Uncertain     căn cứ yếu             -> trả lời KÈM nói rõ là chưa chắc");
        Console.WriteLine("    Conflicted    căn cứ nói ngược nhau  -> trình bày CẢ HAI, không chọn bên");
        Console.WriteLine("    Insufficient  không đủ căn cứ        -> NÓI THẲNG là không biết");
    }

    private static async Task ShowGroundedAnswerAsync()
    {
        Console.WriteLine("\n\n=== DEMO 14: TRẢ LỜI CÓ CƠ SỞ ===\n");

        var clock = Clock();
        var agent = AgentFactory.Build(clock: clock, internet: InternetStackFactory.BuildOffline(clock));

        var answer = await agent.AnswerAsync("SSRF là gì");

        Console.WriteLine($"  CÂU HỎI: {answer.Question}\n");
        Console.WriteLine("  " + answer.Text.Replace("\n", "\n  "));

        Console.WriteLine($"\n  Kết luận      : {answer.Verdict}");
        Console.WriteLine($"  Độ tự tin     : {answer.Confidence:F3} ({answer.Band})");
        Console.WriteLine($"  Sự thật / suy ra: {answer.Summary.FactCount} / {answer.Summary.InferenceCount}");
        Console.WriteLine($"  Kiểm tra cuối : {answer.Validation}");

        Console.WriteLine("\n  Tóm tắt suy luận — QUYẾT ĐỊNH và THAM CHIẾU, không phải dòng suy nghĩ:");
        foreach (string decision in answer.Summary.Decisions)
        {
            Console.WriteLine($"    • {decision}");
        }

        Console.WriteLine($"    căn cứ đã dẫn: {string.Join(", ", answer.Summary.EvidenceIds)}");
    }

    private static async Task ShowInsufficientAsync()
    {
        Console.WriteLine("\n\n=== DEMO 15: KHÔNG ĐỦ CĂN CỨ THÌ KHÔNG TRẢ LỜI ===\n");

        var clock = Clock();
        var agent = AgentFactory.Build(clock: clock, internet: InternetStackFactory.BuildOffline(clock));

        string[] questions =
        [
            "thủ đô của nước Wakanda là gì",
            "zzzqqq khongtontai wxyz",
            "giá bitcoin hiện tại",
        ];

        foreach (string question in questions)
        {
            var answer = await agent.AnswerAsync(question);

            Console.WriteLine($"  \"{question}\"");
            Console.WriteLine($"     -> {answer.Verdict}, tự tin {answer.Confidence:F3}");
            Console.WriteLine($"     {answer.Text.Split('\n')[0]}");
            Console.WriteLine();
        }

        Console.WriteLine("  Đây là nhánh quan trọng nhất của Phase 14. Một hệ thống KHÔNG có nhánh này");
        Console.WriteLine("  sẽ luôn trả lời một điều gì đó — và điều nó nói khi không biết gì chính là");
        Console.WriteLine("  điều nó bịa ra.");
    }

    private static void ShowContradiction()
    {
        Console.WriteLine("\n\n=== DEMO 16: PHÁT HIỆN MÂU THUẪN ===\n");

        var detector = new ContradictionDetector();

        var pairs = new (string Label, Evidence A, Evidence B)[]
        {
            ("phủ định lệch nhau",
                Knowledge("SSRF", "SSRF là lỗ hổng cho phép máy chủ gửi request tới địa chỉ nội bộ"),
                Web("SSRF không phải lỗ hổng cho phép máy chủ gửi request tới địa chỉ nội bộ")),

            ("lệch số",
                Knowledge("Timeout", "Giới hạn thời gian mặc định của bộ tải trang là 10 giây"),
                Web("Giới hạn thời gian mặc định của bộ tải trang là 30 giây")),

            ("cặp từ đối nghịch",
                Knowledge("Cache", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu"),
                Web("Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu")),

            ("KHÁC chủ đề — không phải mâu thuẫn",
                Knowledge("Softmax", "Softmax biến logits thành phân bố xác suất"),
                Knowledge("Overfitting", "Overfitting là khi model không học được quy luật")),
        };

        foreach (var (label, a, b) in pairs)
        {
            var contradiction = detector.Compare(a, b);

            Console.WriteLine($"  {label}");

            if (contradiction is null)
            {
                Console.WriteLine("     -> KHÔNG báo mâu thuẫn (đúng: hai câu không cùng chủ đề)\n");
                continue;
            }

            Console.WriteLine($"     -> {contradiction.Kind}, nặng {contradiction.Severity:F2}");
            Console.WriteLine($"        {contradiction.Explanation}\n");
        }

        Console.WriteLine("  Điều kiện tiên quyết cho cả ba dấu hiệu: hai mẩu phải THỰC SỰ cùng chủ đề.");
        Console.WriteLine("  Bỏ điều kiện đó thì mọi câu chứa chữ \"không\" đều mâu thuẫn với mọi câu");
        Console.WriteLine("  không chứa nó — báo động giả kiểu đó còn tệ hơn không phát hiện gì.");
    }

    private static async Task ShowConflictedAnswerAsync()
    {
        Console.WriteLine("\n\n=== DEMO 17: CĂN CỨ NÓI NGƯỢC NHAU -> KHÔNG CHỌN BÊN ===\n");

        var clock = Clock();

        // Kho kiến thức và web nói ngược nhau về cùng một chuyện.
        var knowledge = new KnowledgeStore(clock);
        knowledge.Add("Cache", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu");

        var web = new StaticSearchProvider("mini-web")
            .Add("Cache", "https://blog.example.com/cache",
                "Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu");

        var collector = new EvidenceCollector(
            knowledge,
            internet: new SearchService(web, clock: clock,
                options: new SearchServiceOptions { FetchPages = false }),
            clock: clock);

        var answer = await new AnswerPipeline(collector, clock: clock)
            .AnswerAsync("bật cache ảnh hưởng tốc độ phản hồi thế nào");

        Console.WriteLine($"  CÂU HỎI: {answer.Question}\n");
        Console.WriteLine("  " + answer.Text.Replace("\n", "\n  "));

        Console.WriteLine($"\n  Kết luận  : {answer.Verdict}");
        Console.WriteLine($"  Độ tự tin : {answer.Confidence:F3} (bị trừ vì mâu thuẫn)");
        Console.WriteLine($"  Mâu thuẫn : {answer.Contradictions.Count}");

        Console.WriteLine("\n  Chọn một phía rồi im lặng về phía kia là hành vi tệ nhất: người đọc nhận");
        Console.WriteLine("  một câu trả lời trông dứt khoát, và mất luôn thông tin quan trọng nhất —");
        Console.WriteLine("  rằng chính các nguồn cũng không thống nhất.");
    }

    private static void ShowValidator()
    {
        Console.WriteLine("\n\n=== DEMO 18: CHỐT CHẶN CUỐI — BẮT CÂU KHÔNG CÓ CĂN CỨ ===\n");

        var validator = new AnswerValidator();

        var evidence = new[]
        {
            Knowledge("SSRF", "SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ"),
        };

        var claims = new[]
        {
            new Claim
            {
                Id = "c1",
                Text = "SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ",
                Kind = ClaimKind.Fact,
                EvidenceIds = [evidence[0].Id],
                Support = 1.0,
            },
        };

        Console.WriteLine("  (a) Câu trả lời dựng ĐÚNG từ căn cứ:\n");
        string good = "SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ.";
        var okReport = validator.Validate(good, claims, evidence);
        Console.WriteLine($"      \"{good}\"");
        Console.WriteLine($"      -> {okReport}\n");

        Console.WriteLine("  (b) Câu trả lời có một câu BỊA THÊM (căn cứ không hề nói):\n");
        string bad = good + " Ngoài ra, mọi tường lửa thương mại đều đã chặn hoàn toàn kiểu tấn công này từ năm 2019.";
        var badReport = validator.Validate(bad, claims, evidence);
        Console.WriteLine($"      \"…đều đã chặn hoàn toàn kiểu tấn công này từ năm 2019.\"");
        Console.WriteLine($"      -> {badReport}");

        foreach (var unsupported in badReport.UnsupportedClaims)
        {
            Console.WriteLine($"         {unsupported}");
        }

        Console.WriteLine("\n  Validator được viết ĐỘC LẬP với bộ sinh câu trả lời — nó chỉ nhận văn bản");
        Console.WriteLine("  cùng danh sách căn cứ rồi tự kiểm. Nhờ vậy nó bắt được cả một bộ sinh viết");
        Console.WriteLine("  sau này, kể cả khi đó là một model đã train (Phase 16).");
    }

    private static async Task ShowFactVsInferenceAsync()
    {
        Console.WriteLine("\n\n=== DEMO 19: SỰ THẬT vs SUY RA ===\n");

        var clock = Clock();
        var agent = AgentFactory.Build(clock: clock, internet: InternetStackFactory.BuildOffline(clock));

        var arithmetic = await agent.AnswerAsync("2 + 3 * 4");

        Console.WriteLine($"  \"2 + 3 * 4\"  -> {arithmetic.Verdict}, tự tin {arithmetic.Confidence:F3}");
        foreach (var claim in arithmetic.Claims)
        {
            Console.WriteLine($"     [{claim.Kind}] {claim.Text}");
        }

        var conceptual = await agent.AnswerAsync("softmax là gì");

        Console.WriteLine($"\n  \"softmax là gì\" -> {conceptual.Verdict}, tự tin {conceptual.Confidence:F3}");
        foreach (var claim in conceptual.Claims)
        {
            Console.WriteLine($"     [{claim.Kind}] {claim.Text[..Math.Min(70, claim.Text.Length)]}…");
        }

        Console.WriteLine("\n  Suy ra có thể SAI dù mọi căn cứ đều đúng — chỉ cần bước ghép sai. Nên nó");
        Console.WriteLine("  được gắn nhãn khác, và bị trừ độ tự tin khi chiếm quá nửa câu trả lời.");
    }

    // ------------------------------------------------------------------

    private static Evidence Knowledge(string topic, string content) =>
        Evidence.FromKnowledge(new KnowledgeEntry
        {
            Id = $"k-{topic}",
            Topic = topic,
            Content = content,
            TopicKeywords = Keywords.Extract(topic),
            AllKeywords = Keywords.Extract($"{topic} {content}"),
            AddedAt = default,
        }) with { Reliability = 0.95, Relevance = 1.0 };

    private static Evidence Web(string content) =>
        Evidence.FromWeb(
            UntrustedText.FromWeb(content),
            Source.FromInternet("Trang web", new Uri("https://blog.example.com/x"), default, 0.6))
        with { Reliability = 0.6, Relevance = 1.0 };
}
