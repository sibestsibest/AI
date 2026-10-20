using Ai.Phase10.Memory;

namespace Ai.Phase10.Demos;

public static class MemoryDemo
{
    public static void Run()
    {
        ShowStoreAndRetrieve();
        ShowSelectiveStorage();
        ShowRanking();
        ShowShortVersusLongTerm();
        ShowForgetting();
        ShowComparisonTable();
    }

    private static MemoryStore BuildStore(ManualClock clock) =>
        new(clock, shortTermCapacity: 5);

    private static void ShowStoreAndRetrieve()
    {
        Console.WriteLine("=== DEMO 1: STORE & RETRIEVE ===\n");

        var clock = new ManualClock(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));
        var store = BuildStore(clock);

        (string Content, double Importance)[] inputs =
        [
            ("User likes Gundam", 0.6),
            ("User uses C#", 0.9),
            ("User works with Azure", 0.8),
        ];

        Console.WriteLine("  Nạp ký ức:");
        foreach (var (content, importance) in inputs)
        {
            var result = store.Store(content, importance);
            Console.WriteLine($"    \"{content}\" (importance {importance:F1}) -> {result.Outcome}");
            clock.Advance(TimeSpan.FromMinutes(5));
        }

        Console.WriteLine($"\n  Bộ nhớ dài hạn: {store.LongTerm.Count} mẩu");
        Console.WriteLine($"  Bộ nhớ ngắn hạn: {store.ShortTerm.Count}/{store.ShortTerm.Capacity} mẩu\n");

        foreach (string query in new[] { "Gundam", "C#", "cloud Azure", "cooking" })
        {
            var fact = store.Retrieve(query);
            Console.WriteLine($"    Retrieve(\"{query}\") -> {(fact is null ? "KHÔNG NHỚ GÌ LIÊN QUAN" : $"\"{fact.Content}\"")}");
        }

        Console.WriteLine("\n  \"cooking\" không khớp từ khoá nào -> store trả về null thay vì bịa.");
    }

    private static void ShowSelectiveStorage()
    {
        Console.WriteLine("\n\n=== DEMO 2: KHÔNG LƯU MỌI THỨ MỘT CÁCH MÙ QUÁNG ===\n");

        var clock = new ManualClock(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));
        var store = BuildStore(clock);

        (string Content, double Importance)[] inputs =
        [
            ("User uses C#", 0.9),
            ("ok", 0.9),
            ("User uses C#", 0.9),
            ("User mentioned the weather is nice", 0.1),
            ("uses C# User", 0.9),
            ("User works with Azure", 0.8),
        ];

        Console.WriteLine($"  {"đầu vào",-44} | {"kết quả",-11} | lý do");
        Console.WriteLine("  " + new string('-', 96));

        foreach (var (content, importance) in inputs)
        {
            var result = store.Store(content, importance);
            Console.WriteLine($"  {$"\"{content}\" ({importance:F1})",-44} | {result.Outcome,-11} | {result.Reason}");
        }

        Console.WriteLine($"\n  Đã nhận 6 đầu vào, chỉ lưu {store.LongTerm.Count} ký ức:");
        foreach (var fact in store.LongTerm.Facts)
        {
            Console.WriteLine($"    {fact}  (importance {fact.Importance:F2}, nhắc {fact.AccessCount} lần)");
        }

        Console.WriteLine("\n  Chú ý dòng \"uses C# User\": đảo thứ tự từ nhưng vẫn là cùng một ý,");
        Console.WriteLine("  nên bị nhận ra là trùng và đem đi CỦNG CỐ thay vì tạo bản sao.");
    }

    private static void ShowRanking()
    {
        Console.WriteLine("\n\n=== DEMO 3: RANK — VÌ SAO KÝ ỨC NÀY ĐƯỢC CHỌN? ===\n");

        var clock = new ManualClock(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));
        var store = BuildStore(clock);

        store.Store("User uses C# and dotnet every day", 0.9);
        store.Store("User works with Azure cloud", 0.8);
        store.Store("User likes Gundam models", 0.6);
        store.Store("User once tried Python", 0.35);

        // Giả lập: C# được dùng nhiều lần, Python thì bị bỏ quên từ lâu.
        for (int i = 0; i < 6; i++) store.Retrieve("C#");
        clock.Advance(TimeSpan.FromDays(10));

        const string query = "programming language";
        Console.WriteLine($"  Câu hỏi: \"{query}\" -> từ khoá: programming, language\n");

        foreach (var scored in store.Rank("C# python programming"))
        {
            Console.WriteLine($"  {scored.Score,6:F3}  {scored.Fact.Content}");
            Console.WriteLine($"          {scored.Explain()}");
        }

        Console.WriteLine("\n  \"C#\" thắng không chỉ vì khớp từ khoá, mà còn vì quan trọng hơn,");
        Console.WriteLine("  được dùng gần đây hơn và dùng nhiều lần hơn.");
        Console.WriteLine("  Ký ức không khớp từ khoá nào bị chấm 0 — không bao giờ lôi ra bừa.");
    }

    private static void ShowShortVersusLongTerm()
    {
        Console.WriteLine("\n\n=== DEMO 4: NGẮN HẠN vs DÀI HẠN ===\n");

        var clock = new ManualClock(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));
        var store = new MemoryStore(clock, shortTermCapacity: 3);

        string[] conversation =
        [
            "User uses C#",
            "User works with Azure",
            "User likes Gundam",
            "User lives in Hanoi",
            "User prefers dark theme",
        ];

        Console.WriteLine("  Bộ nhớ ngắn hạn sức chứa 3. Nạp lần lượt 5 câu:\n");

        foreach (string line in conversation)
        {
            store.Store(line, 0.7);
            var window = store.ShortTerm.Items.Select(f => f.Content.Replace("User ", ""));
            Console.WriteLine($"    + \"{line}\"");
            Console.WriteLine($"      cửa sổ ngắn hạn: [{string.Join(" | ", window)}]");
        }

        Console.WriteLine($"\n  Ngắn hạn giữ {store.ShortTerm.Count} mẩu mới nhất.");
        Console.WriteLine($"  Dài hạn vẫn giữ đủ {store.LongTerm.Count} mẩu — vì mọi mẩu đều đủ quan trọng.");
        Console.WriteLine($"\n    Retrieve(\"C#\") -> \"{store.Retrieve("C#")?.Content}\"");
        Console.WriteLine("    -> \"User uses C#\" đã trôi khỏi cửa sổ ngắn hạn nhưng vẫn nhớ được.");
        Console.WriteLine("\n  Đây là điều một LLM thuần KHÔNG làm được: cái gì trôi khỏi context");
        Console.WriteLine("  window là mất hẳn. Phải có bộ nhớ dài hạn bên ngoài thì mới nhớ lâu.");
    }

    private static void ShowForgetting()
    {
        Console.WriteLine("\n\n=== DEMO 5: FORGET ===\n");

        var clock = new ManualClock(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));
        var store = new MemoryStore(clock, shortTermCapacity: 10);

        store.Store("User uses C# every day", 0.95);
        store.Store("User works with Azure", 0.80);
        store.Store("User likes Gundam", 0.60);
        store.Store("User tried a new cafe once", 0.35);
        store.Store("User mentioned a random link", 0.30);

        Console.WriteLine("  Điểm giữ lại ngay sau khi lưu (mọi thứ đều còn tươi):");
        foreach (var fact in store.LongTerm.Facts)
        {
            Console.WriteLine($"    {store.RetentionScore(fact),6:F3}  {fact.Content}");
        }

        Console.WriteLine("\n  --- 30 ngày trôi qua, người dùng chỉ nhắc lại C# và Azure ---\n");
        clock.Advance(TimeSpan.FromDays(30));
        store.Retrieve("C#");
        store.Retrieve("Azure");

        Console.WriteLine("  Điểm giữ lại lúc này:");
        foreach (var fact in store.LongTerm.Facts.OrderByDescending(store.RetentionScore))
        {
            Console.WriteLine($"    {store.RetentionScore(fact),6:F3}  {fact.Content}");
        }

        var forgotten = store.ForgetWeak(threshold: 0.25);

        Console.WriteLine($"\n  ForgetWeak(0.25) đã quên {forgotten.Count} mẩu:");
        foreach (var fact in forgotten)
        {
            Console.WriteLine($"    - {fact.Content}");
        }

        Console.WriteLine($"\n  Còn lại {store.LongTerm.Count} mẩu:");
        foreach (var fact in store.LongTerm.Facts)
        {
            Console.WriteLine($"    + {fact.Content}");
        }

        Console.WriteLine("\n  Quên KHÔNG phải là hỏng hóc — nó là tính năng. Bộ nhớ giữ lại mọi thứ");
        Console.WriteLine("  sẽ ngập rác, và rác làm hỏng khâu truy xuất. Quên đúng thứ vô giá trị");
        Console.WriteLine("  chính là cách giữ cho phần còn lại dùng được.");

        Console.WriteLine("\n  Quên có chủ đích (khi người dùng yêu cầu xoá):");
        var target = store.LongTerm.Facts.First(f => f.Content.Contains("Gundam"));
        Console.WriteLine($"    Forget(\"{target.Id}\") -> {store.Forget(target.Id)}");
        Console.WriteLine($"    Retrieve(\"Gundam\") -> {store.Retrieve("Gundam")?.Content ?? "KHÔNG CÒN NHỚ"}");
    }

    private static void ShowComparisonTable()
    {
        Console.WriteLine("\n\n=== DEMO 6: MODEL WEIGHTS vs MEMORY vs CONTEXT vs DATABASE ===\n");

        (string Aspect, string Weights, string Memory, string Context, string Database)[] rows =
        [
            ("Nội dung", "mẫu hình thống kê", "sự kiện về người dùng", "cuộc hội thoại hiện tại", "dữ liệu nghiệp vụ"),
            ("Ghi vào bằng", "huấn luyện (chậm)", "Store() (tức thì)", "nối vào prompt", "INSERT"),
            ("Tồn tại", "vĩnh viễn đến khi train lại", "đến khi bị quên", "hết phiên là mất", "đến khi DELETE"),
            ("Dung lượng", "cố định theo kiến trúc", "lớn, có chọn lọc", "hữu hạn & nhỏ", "rất lớn"),
            ("Truy xuất", "ngầm, qua forward pass", "Search + Rank", "đọc thẳng", "truy vấn chính xác"),
            ("Tự phai đi", "không", "CÓ", "không (bị cắt cứng)", "không"),
            ("Ví dụ", "'sau I like thường là danh từ'", "'User dùng C#'", "3 câu vừa chat", "bảng Orders"),
        ];

        Console.WriteLine($"  {"",-16} {"Model weights",-32} {"Memory",-26} {"Context",-26} Database");
        Console.WriteLine("  " + new string('-', 128));

        foreach (var (aspect, weights, memory, context, database) in rows)
        {
            Console.WriteLine($"  {aspect,-16} {weights,-32} {memory,-26} {context,-26} {database}");
        }

        Console.WriteLine("\n  Bốn thứ này KHÔNG thay thế được cho nhau:");
        Console.WriteLine("    • Weights biết ngôn ngữ hoạt động ra sao, nhưng không biết BẠN là ai.");
        Console.WriteLine("    • Context biết bạn vừa nói gì, nhưng quên sạch khi đóng phiên.");
        Console.WriteLine("    • Database biết chính xác nhưng phải hỏi đúng câu truy vấn mới ra.");
        Console.WriteLine("    • Memory là thứ duy nhất tích luỹ được hiểu biết về bạn qua thời gian,");
        Console.WriteLine("      tự bỏ bớt cái vô giá trị, và trả lời được câu hỏi mơ hồ.");
    }
}
