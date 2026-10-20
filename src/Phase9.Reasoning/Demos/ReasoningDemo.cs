using Ai.Phase9.Reasoning;

namespace Ai.Phase9.Demos;

public static class ReasoningDemo
{
    public static void Run()
    {
        ShowBasicExample();
        ShowAllOperators();
        ShowChainedReasoning();
        ShowNaturalLanguage();
        ShowLimits();
    }

    private static void ShowBasicExample()
    {
        Console.WriteLine("=== DEMO 1: VÍ DỤ CỦA ĐỀ BÀI ===\n");

        var engine = new ReasoningEngine();

        Console.WriteLine("  Nạp sự thật:");
        foreach (string statement in new[] { "A = 10", "B = 20", "C = A + B" })
        {
            var fact = engine.Learn(statement);
            Console.WriteLine($"    {statement,-14} -> lưu: {fact.Name} := {fact.Value}");
        }

        Console.WriteLine("\n  Câu hỏi: \"C bằng bao nhiêu?\"\n");
        var answer = engine.Ask("C bằng bao nhiêu?");

        Console.WriteLine("  Quy trình:");
        Console.WriteLine($"    1. Parse question   -> \"{QuestionNormalizer.Normalize("C bằng bao nhiêu?")}\"");
        Console.WriteLine($"    2. Find relevant    -> {string.Join(", ", answer.UsedFacts.Select(f => f.Name))}");
        Console.WriteLine("    3. Apply rule & calculate:");

        foreach (var step in answer.Steps)
        {
            Console.WriteLine($"         {step.Description}");
        }

        Console.WriteLine($"\n  TRẢ LỜI: C = {answer.Text}");
        Console.WriteLine("\n  Không có con số nào được học. Engine suy ra bằng phép thay thế");
        Console.WriteLine("  và số học — đúng tuyệt đối, và mọi bước đều truy vết được.");
    }

    private static void ShowAllOperators()
    {
        Console.WriteLine("\n\n=== DEMO 2: CÁC LUẬT ĐƯỢC HỖ TRỢ ===\n");

        var engine = new ReasoningEngine();
        engine.LearnAll("A = 10", "B = 20");

        Console.WriteLine("  Đã biết: A = 10 , B = 20\n");
        Console.WriteLine($"  {"câu hỏi",-14} | {"kết quả",10} | giải thích");
        Console.WriteLine("  " + new string('-', 56));

        foreach (string question in new[] { "A + B", "A - B", "A * B", "A / B", "A = B", "A > B", "A < B" })
        {
            var answer = engine.Ask(question);
            string explanation = answer.IsComparison
                ? $"so sánh 10 và 20"
                : $"tính số học";

            Console.WriteLine($"  {question,-14} | {answer.Text,10} | {explanation}");
        }

        Console.WriteLine("\n  Có cả thứ tự ưu tiên và dấu ngoặc:");
        foreach (string question in new[] { "2 + 3 * 4", "( 2 + 3 ) * 4", "A + B * 2", "( A + B ) / 3" })
        {
            Console.WriteLine($"    {question,-16} = {engine.Ask(question).Text}");
        }
    }

    private static void ShowChainedReasoning()
    {
        Console.WriteLine("\n\n=== DEMO 3: SUY LUẬN NHIỀU BƯỚC ===\n");

        var engine = new ReasoningEngine();
        engine.LearnAll(
            "Price = 250",
            "Quantity = 4",
            "Subtotal = Price * Quantity",
            "TaxRate = 10",
            "Tax = Subtotal * TaxRate / 100",
            "Total = Subtotal + Tax");

        Console.WriteLine("  Đã biết:");
        foreach (var fact in engine.Knowledge.Facts)
        {
            Console.WriteLine($"    {fact.Source}");
        }

        Console.WriteLine("\n  Câu hỏi: \"Total bằng bao nhiêu?\"\n");
        var answer = engine.Ask("Total bằng bao nhiêu?");

        foreach (var step in answer.Steps)
        {
            Console.WriteLine($"    {step.Description}");
        }

        Console.WriteLine($"\n  TRẢ LỜI: {answer.Text}");
        Console.WriteLine("\n  Engine tự lần ngược chuỗi phụ thuộc 3 tầng mà không ai chỉ đường.");

        Console.WriteLine("\n  So sánh cũng suy luận bắc cầu được:");
        Console.WriteLine($"    \"Total > Subtotal\"  -> {engine.Ask("Total > Subtotal").Text}");
        Console.WriteLine($"    \"Tax < Subtotal\"    -> {engine.Ask("Tax < Subtotal").Text}");

        Console.WriteLine("\n  Đổi MỘT sự thật, mọi kết quả phụ thuộc cập nhật ngay, không cần train lại:");
        engine.Learn("Quantity = 10");
        Console.WriteLine($"    Quantity = 10  ->  Total = {engine.Ask("Total").Text}");
    }

    private static void ShowNaturalLanguage()
    {
        Console.WriteLine("\n\n=== DEMO 4: CÂU HỎI BẰNG NGÔN NGỮ TỰ NHIÊN ===\n");

        var engine = new ReasoningEngine();
        engine.LearnAll("A = 10", "B = 20", "C = A + B");

        string[] questions =
        [
            "C bằng bao nhiêu?",
            "Cho biết giá trị của C",
            "Hãy tính A + B",
            "A có lớn hơn B không?",
            "B có lớn hơn A không?",
            "what is C",
            "A plus B",
            "C divided by 3",
        ];

        Console.WriteLine($"  {"câu hỏi gốc",-34} | {"sau khi gọt",-16} | kết quả");
        Console.WriteLine("  " + new string('-', 72));

        foreach (string question in questions)
        {
            string normalized = QuestionNormalizer.Normalize(question);
            Console.WriteLine($"  {question,-34} | {normalized,-16} | {engine.Ask(question).Text}");
        }
    }

    private static void ShowLimits()
    {
        Console.WriteLine("\n\n=== DEMO 5: ENGINE TỪ CHỐI KHI KHÔNG BIẾT ===\n");

        var engine = new ReasoningEngine();
        engine.LearnAll("A = 10", "B = 0", "Loop1 = Loop2 + 1", "Loop2 = Loop1 + 1");

        (string Question, string Why)[] cases =
        [
            ("Z bằng bao nhiêu?", "biến chưa được định nghĩa"),
            ("A / B", "chia cho 0"),
            ("Loop1", "hai sự thật tham chiếu vòng tròn"),
            ("A + + B", "cú pháp sai"),
        ];

        foreach (var (question, why) in cases)
        {
            try
            {
                Console.WriteLine($"  \"{question}\" -> {engine.Ask(question).Text}");
            }
            catch (Exception ex) when (ex is ReasoningException or ParseException)
            {
                Console.WriteLine($"  \"{question,-20}\" -> TỪ CHỐI ({why})");
                Console.WriteLine($"      {ex.Message}");
            }
        }

        Console.WriteLine("\n  Đây là khác biệt lớn nhất so với mạng neural: engine BIẾT khi nào");
        Console.WriteLine("  nó không biết, và nói thẳng. Mạng neural luôn trả về MỘT con số,");
        Console.WriteLine("  kể cả khi con số đó hoàn toàn bịa — nó không có khái niệm 'không biết'.");
    }
}
