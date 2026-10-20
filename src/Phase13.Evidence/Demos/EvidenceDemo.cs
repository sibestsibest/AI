using Ai.Phase13.Agent;
using Ai.Phase13.Grounding;
using Ai.Phase13.Internet;
using Ai.Phase13.Memory;

namespace Ai.Phase13.Demos;

/// <summary>Demo riêng cho Phase 13 — Knowledge + Memory + Context + Evidence.</summary>
public static class EvidenceDemo
{
    public static async Task RunAsync()
    {
        ShowThreeStores();
        var agent = await ShowEvidenceCollectionAsync();
        await ShowVerifiedBeatsWebAsync();
        await ShowContextAsync();
        ShowKnowledgeRule();
        await ShowProvenanceAsync(agent);
    }

    private static ManualClock Clock() => new(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));

    private static void ShowThreeStores()
    {
        Console.WriteLine("\n\n=== DEMO 7: BA CHỖ CHỨA THÔNG TIN, KHÁC NHAU Ở ĐÂU ===\n");

        Console.WriteLine("                         Knowledge      Memory        Context");
        Console.WriteLine("  ----------------------------------------------------------------");
        Console.WriteLine("  Nói về                 thế giới       người dùng    lượt vừa rồi");
        Console.WriteLine("  Ghi vào bằng           người biên tập Store()       mỗi lượt nói");
        Console.WriteLine("  Tồn tại                đến khi sửa    đến khi quên  hết phiên là mất");
        Console.WriteLine("  Tự phai đi             không          CÓ, theo giá trị  không (cắt cứng)");
        Console.WriteLine("  Mức tin cậy            Verified       UserProvided  UserProvided");
        Console.WriteLine("  Thành kiến thức được?  đã là rồi      được          được");

        Console.WriteLine("\n  Và kênh thứ tư — Internet — là kênh duy nhất KHÔNG được:");
        Console.WriteLine("    Internet               thế giới       (không kiểm)  Untrusted   -> KHÔNG BAO GIỜ");
    }

    private static async Task<AiController> ShowEvidenceCollectionAsync()
    {
        Console.WriteLine("\n\n=== DEMO 8: THU THẬP CĂN CỨ TỪ MỌI KÊNH ===\n");

        var clock = Clock();
        var agent = AgentFactory.Build(clock: clock, internet: InternetStackFactory.BuildOffline(clock));

        await agent.HandleAsync("hãy nhớ rằng tôi đang nghiên cứu về SSRF");
        clock.Advance(TimeSpan.FromMinutes(5));

        var set = await agent.CollectEvidenceAsync("SSRF là gì");

        Console.WriteLine($"  Câu hỏi: \"{set.Question}\"");
        Console.WriteLine($"  Gom được {set.Items.Count} mẩu căn cứ từ {set.DistinctChannelCount} kênh " +
                          $"(đã gộp {set.DuplicatesRemoved} mẩu trùng)\n");

        Console.WriteLine($"  {"điểm",6} | {"kênh",-12} | {"tin cậy",7} | căn cứ");
        Console.WriteLine("  " + new string('-', 96));

        foreach (var item in set.Items)
        {
            Console.WriteLine($"  {item.Score,6:F3} | {item.Kind,-12} | {item.Reliability,7:F2} | {item.Preview(52)}");
            Console.WriteLine($"  {"",6} | {"",-12} | {"",7} | ↳ {item.Source.Cite()}");
        }

        Console.WriteLine($"\n  Có nguồn đã kiểm chứng : {set.HasVerifiedEvidence}");
        Console.WriteLine($"  Phải cảnh báo chưa kiểm: {set.RequiresDisclosure}");
        Console.WriteLine($"  Độ tin cậy cao nhất    : {set.BestReliability:F2}");

        return agent;
    }

    private static async Task ShowVerifiedBeatsWebAsync()
    {
        Console.WriteLine("\n\n=== DEMO 9: NGUỒN ĐÃ KIỂM THẮNG NGUỒN WEB ===\n");

        Console.WriteLine("  Cùng một câu hỏi, hai cấu hình khác nhau:\n");

        var clock = Clock();

        // (a) chỉ có Internet, không có kho kiến thức
        var webOnly = new EvidenceCollector(
            knowledge: null,
            internet: InternetStackFactory.BuildOffline(clock).Service,
            clock: clock);

        var a = await webOnly.CollectAsync("prompt injection là gì");

        Console.WriteLine("  (a) CHỈ có Internet:");
        Console.WriteLine($"      {a.Items.Count} mẩu, kênh: {string.Join(", ", a.Items.Select(i => i.Kind).Distinct())}");
        Console.WriteLine($"      phải cảnh báo chưa kiểm chứng: {a.RequiresDisclosure}");
        Console.WriteLine($"      mẩu đầu: {a.Best?.Preview(70)}");

        // (b) có cả kho kiến thức nội bộ — kho này CŨNG có tài liệu về prompt injection
        var both = new EvidenceCollector(
            KnowledgeSeed.BuildStore(clock),
            internet: InternetStackFactory.BuildOffline(clock).Service,
            clock: clock);

        var b = await both.CollectAsync("prompt injection là gì");

        Console.WriteLine("\n  (b) Có CẢ kho kiến thức nội bộ:");
        Console.WriteLine($"      {b.Items.Count} mẩu, kênh: {string.Join(", ", b.Items.Select(i => i.Kind).Distinct())}");
        Console.WriteLine($"      phải cảnh báo chưa kiểm chứng: {b.RequiresDisclosure}");
        Console.WriteLine($"      mẩu đầu: {b.Best?.Preview(70)}");
        Console.WriteLine($"      ↳ {b.Best?.Source.Cite()}");

        Console.WriteLine("\n  Cùng một câu hỏi, cùng một Internet. Khác biệt duy nhất là có kho đã");
        Console.WriteLine("  kiểm hay không — và nó đổi cả thứ tự căn cứ lẫn việc có phải cảnh báo.");
        Console.WriteLine("  Đây là điều Phase 13 thêm vào: không phải lấy được NHIỀU thông tin hơn,");
        Console.WriteLine("  mà biết thông tin nào ĐÁNG TIN hơn, và nói ra được vì sao.");
    }

    private static async Task ShowContextAsync()
    {
        Console.WriteLine("\n\n=== DEMO 10: NGỮ CẢNH HỘI THOẠI ===\n");

        var clock = Clock();
        var context = new ConversationContext(clock, maxTurns: 3);

        context.Record("softmax hoạt động thế nào", "Softmax biến logits thành phân bố xác suất.");
        context.Record("còn attention thì sao", "Attention tính softmax(QKᵀ/√d)V.");

        Console.WriteLine("  Bản ghi hội thoại:");
        Console.WriteLine("    " + context.Transcript().Replace("\n", "\n    "));

        Console.WriteLine("\n  Câu hỏi quá ngắn được ngữ cảnh bổ sung:");
        Console.WriteLine($"    \"còn nó\"        -> \"{context.Expand("còn nó")}\"");

        Console.WriteLine("\n  Nhưng câu hỏi ĐÃ RÕ thì để nguyên, không nhồi ngữ cảnh vào:");
        Console.WriteLine($"    \"overfitting là gì\" -> \"{context.Expand("overfitting là gì")}\"");

        Console.WriteLine("\n  Cửa sổ trượt cắt CỨNG theo số lượt (maxTurns = 3):");
        context.Record("embedding là gì", "...");
        context.Record("transformer là gì", "...");

        Console.WriteLine($"    đã đi qua {context.TotalTurns} lượt, còn giữ {context.Count}");
        Console.WriteLine($"    lượt còn lại: {string.Join(", ", context.Turns.Select(t => $"#{t.Index}"))}");
        Console.WriteLine("\n    Lượt #1 bị đẩy ra, bất kể nó quan trọng đến đâu. Đó đúng là hành vi");
        Console.WriteLine("    của context window — và là lý do memory phải tồn tại.");

        await Task.CompletedTask;
    }

    private static void ShowKnowledgeRule()
    {
        Console.WriteLine("\n\n=== DEMO 11: LUẬT CỨNG — WEB KHÔNG THÀNH KIẾN THỨC ===\n");

        var clock = Clock();
        var store = new KnowledgeStore(clock);

        Console.WriteLine($"  Kho kiến thức ban đầu: {store.Count} mục\n");

        var candidates = new (string Label, Evidence Evidence)[]
        {
            ("từ Internet",
                Evidence.FromWeb(
                    UntrustedText.FromWeb("Một trang web nào đó khẳng định điều gì đó rất thuyết phục."),
                    Source.FromInternet("Trang web", new Uri("https://example.com/x"), clock.Now, 0.9))),

            ("từ Internet (.gov, uy tín cao)",
                Evidence.FromWeb(
                    UntrustedText.FromWeb("Kể cả một trang .gov cũng không được tự động thành kiến thức."),
                    Source.FromInternet("NIST", new Uri("https://nist.gov/x"), clock.Now, 1.0))),

            ("người dùng tự nói",
                Evidence.FromMemory(new MemoryFact
                {
                    Id = "m1",
                    Content = "tôi dùng C# và dotnet",
                    Keywords = Memory.Keywords.Extract("tôi dùng C# và dotnet"),
                    Importance = 0.7,
                    CreatedAt = clock.Now,
                    LastAccessedAt = clock.Now,
                })),

            ("suy luận ký hiệu",
                Evidence.FromReasoning("C = A + B", "30")),
        };

        foreach (var (label, evidence) in candidates)
        {
            var result = store.TryPromote(evidence, topic: label);

            Console.WriteLine($"  {(result.Accepted ? "NHẬN " : "CHẶN ")} {label,-32} {result.Reason}");
        }

        Console.WriteLine($"\n  Kho kiến thức sau cùng: {store.Count} mục");
        Console.WriteLine("\n  Vì sao luật này quan trọng đến thế? Vì nếu bỏ nó, đường đi sẽ là:");
        Console.WriteLine("    trang web bất kỳ -> kiến thức lâu dài -> dữ liệu huấn luyện (Phase 15)");
        Console.WriteLine("    -> trọng số model");
        Console.WriteLine("  Tức là bất kỳ ai đăng được một trang web cũng dạy được model.");
    }

    private static async Task ShowProvenanceAsync(AiController agent)
    {
        Console.WriteLine("\n\n=== DEMO 12: MỌI KHẲNG ĐỊNH TRUY ĐƯỢC VỀ NGUỒN ===\n");

        var response = await agent.HandleAsync("tìm trên mạng thông tin về SSRF");

        Console.WriteLine($"  NGƯỜI DÙNG: tìm trên mạng thông tin về SSRF");
        Console.WriteLine($"  AGENT     : {response.Answer.Replace("\n", "\n              ")}\n");

        Console.WriteLine($"  Lượt số               : {response.TurnIndex}");
        Console.WriteLine($"  Dữ liệu chưa kiểm     : {response.UsedUnverifiedInternetData}");
        Console.WriteLine($"  Số mẩu căn cứ         : {response.Evidence?.Items.Count}");
        Console.WriteLine($"  Số kênh               : {response.Evidence?.DistinctChannelCount}");

        Console.WriteLine("\n  Dấu vết chấm điểm:");
        Console.WriteLine("    " + (response.Detail ?? "").Replace("\n", "\n    "));

        Console.WriteLine("\n  Bộ nhớ và kho kiến thức KHÔNG hề tăng lên sau khi tra mạng:");
        Console.WriteLine($"    ký ức    : {agent.Memory.LongTerm.Count} mẩu (chỉ gồm điều bạn tự bảo tôi nhớ)");
        Console.WriteLine($"    kiến thức: {agent.Knowledge?.Count} mục (đúng bằng lúc khởi tạo)");
        Console.WriteLine($"    mục nào có nguồn gốc từ web? " +
                          $"{agent.Knowledge?.Entries.Count(e => e.Origin.Contains("Internet"))}");
    }
}
