using Ai.Phase11.Memory;
using Ai.Phase11.Reasoning;
using Ai.Phase11.Tools;

namespace Ai.Phase11.Agent;

/// <summary>Một bước trong quá trình quyết định — để agent giải trình được.</summary>
public sealed record DecisionStep(string Question, string Decision);

public sealed record AgentResponse(
    string Answer,
    Intent Intent,
    double Confidence,
    string? ToolUsed,
    IReadOnlyList<DecisionStep> Decisions,
    string? Detail = null);

/// <summary>
/// AI CONTROLLER — bộ não điều phối.
///
///                     User
///                       ↓
///                 AI Controller
///                       ↓
///               ┌───────┴───────┐
///               ↓               ↓
///            Memory         Reasoning
///               ↓               ↓
///               └───────┬───────┘
///                       ↓
///                     Tools
///               ┌───────┼───────┐
///               ↓       ↓       ↓
///          Calculator Search Database
///                       ↓
///                    Result
///                       ↓
///                  AI Response
///
/// LUỒNG QUYẾT ĐỊNH (đúng như đề bài):
///
///     1. Trả lời được bằng kiến thức sẵn có không?
///            KHÔNG ↓
///     2. Có cần Memory không?
///            KHÔNG ↓
///     3. Có cần Tool không?
///            CÓ    ↓
///     4. Gọi Tool -> nhận kết quả -> Reasoning -> Answer
///
/// Điểm mấu chốt của toàn bộ Phase 11: controller KHÔNG cố tự làm mọi thứ.
/// Mạng neural chỉ làm đúng một việc nó giỏi — đoán xem người dùng muốn gì
/// (bài toán mơ hồ, nhiều cách diễn đạt). Phần thực thi giao cho công cụ
/// chính xác. Đây là lý do agent trả lời được "2+3*4" bằng 14 chứ không
/// phải 13.97.
/// </summary>
public sealed class AiController
{
    private readonly IntentClassifier _classifier;
    private readonly MemoryStore _memory;
    private readonly CalculatorTool _calculator;
    private readonly SearchTool _search;
    private readonly DatabaseTool _database;

    public AiController(
        IntentClassifier classifier,
        MemoryStore memory,
        CalculatorTool calculator,
        SearchTool search,
        DatabaseTool database)
    {
        ArgumentNullException.ThrowIfNull(classifier);
        ArgumentNullException.ThrowIfNull(memory);
        ArgumentNullException.ThrowIfNull(calculator);
        ArgumentNullException.ThrowIfNull(search);
        ArgumentNullException.ThrowIfNull(database);

        _classifier = classifier;
        _memory = memory;
        _calculator = calculator;
        _search = search;
        _database = database;
    }

    public MemoryStore Memory => _memory;

    public ReasoningEngine Reasoning => _calculator.Engine;

    public IReadOnlyList<ITool> Tools => [_calculator, _search, _database];

    public AgentResponse Handle(string input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var decisions = new List<DecisionStep>();
        var prediction = _classifier.Classify(input);

        decisions.Add(new DecisionStep(
            "Người dùng muốn gì?",
            $"{prediction.Intent} (độ tự tin {prediction.Confidence:P0}, " +
            $"nhận ra {_classifier.KnownWordCount(input)} từ quen)"));

        // --- BƯỚC 1: trả lời được ngay bằng kiến thức sẵn có? ---
        if (prediction.Intent == Intent.SmallTalk && prediction.IsConfident)
        {
            decisions.Add(new DecisionStep("Trả lời được bằng kiến thức sẵn có?", "CÓ — chào hỏi"));

            return new AgentResponse(SmallTalkReply(input), prediction.Intent, prediction.Confidence,
                null, decisions);
        }

        decisions.Add(new DecisionStep("Trả lời được bằng kiến thức sẵn có?", "KHÔNG"));

        // Không nhận ra ý định thì nói thẳng, không đoán bừa.
        if (!prediction.IsConfident)
        {
            decisions.Add(new DecisionStep("Có đủ tự tin để hành động?",
                $"KHÔNG — {prediction.Confidence:P0} dưới ngưỡng {IntentPrediction.MinimumConfidence:P0}"));

            return new AgentResponse(
                "Tôi không chắc bạn muốn gì. Tôi làm được: tính toán, ghi nhớ thông tin, " +
                "nhớ lại điều bạn đã nói, tra cứu khái niệm, và truy vấn dữ liệu đơn hàng.",
                prediction.Intent, prediction.Confidence, null, decisions);
        }

        // --- BƯỚC 2: có cần Memory không? ---
        if (prediction.Intent is Intent.RememberFact or Intent.RecallMemory)
        {
            decisions.Add(new DecisionStep("Có cần Memory?", "CÓ"));
            return HandleMemory(input, prediction, decisions);
        }

        decisions.Add(new DecisionStep("Có cần Memory?", "KHÔNG"));

        // --- BƯỚC 3 & 4: cần Tool -> gọi -> suy luận -> trả lời ---
        var tool = SelectTool(prediction.Intent);
        decisions.Add(new DecisionStep("Có cần Tool?", $"CÓ — dùng '{tool.Name}'"));

        var result = tool.Execute(input);
        decisions.Add(new DecisionStep("Tool trả về gì?",
            result.Success ? $"\"{result.Output}\"" : $"THẤT BẠI — {result.Output}"));

        if (!result.Success)
        {
            return new AgentResponse(
                $"Tôi đã thử dùng {tool.Name} nhưng không xong: {result.Output}",
                prediction.Intent, prediction.Confidence, tool.Name, decisions, result.Detail);
        }

        decisions.Add(new DecisionStep("Diễn giải kết quả", "ghép thành câu trả lời"));

        return new AgentResponse(Compose(prediction.Intent, result),
            prediction.Intent, prediction.Confidence, tool.Name, decisions, result.Detail);
    }

    private ITool SelectTool(Intent intent) => intent switch
    {
        Intent.Calculate => _calculator,
        Intent.SearchKnowledge => _search,
        Intent.QueryDatabase => _database,
        _ => _search,
    };

    private AgentResponse HandleMemory(string input, IntentPrediction prediction, List<DecisionStep> decisions)
    {
        if (prediction.Intent == Intent.RememberFact)
        {
            string content = StripRememberPrefix(input);
            var stored = _memory.Store(content, importance: 0.7);

            decisions.Add(new DecisionStep("Có đáng lưu không?", $"{stored.Outcome} — {stored.Reason}"));

            string answer = stored.Outcome switch
            {
                StoreOutcome.Stored => $"Đã ghi nhớ: \"{content}\".",
                StoreOutcome.Reinforced => $"Tôi đã biết điều này rồi, và vừa củng cố lại: \"{content}\".",
                _ => $"Tôi không lưu câu này ({stored.Reason}).",
            };

            return new AgentResponse(answer, prediction.Intent, prediction.Confidence,
                "memory", decisions, stored.Reason);
        }

        var hits = _memory.Search(input, take: 3);
        decisions.Add(new DecisionStep("Memory có gì liên quan?",
            hits.Count == 0 ? "không có" : $"{hits.Count} mẩu, điểm cao nhất {hits[0].Score:F3}"));

        if (hits.Count == 0)
        {
            return new AgentResponse(
                "Tôi không nhớ gì liên quan đến câu hỏi này. Bạn có thể bảo tôi ghi nhớ.",
                prediction.Intent, prediction.Confidence, "memory", decisions);
        }

        string recalled = string.Join("; ", hits.Select(h => h.Fact.Content));
        return new AgentResponse($"Tôi nhớ: {recalled}.", prediction.Intent, prediction.Confidence,
            "memory", decisions, hits[0].Explain());
    }

    private static string Compose(Intent intent, ToolResult result) => intent switch
    {
        Intent.Calculate => $"Kết quả: {result.Output}",
        Intent.SearchKnowledge => result.Output,
        Intent.QueryDatabase => $"Theo dữ liệu: {result.Output}",
        _ => result.Output,
    };

    private static string SmallTalkReply(string input)
    {
        var words = Keywords.Extract(input);

        if (words.Contains("tạm") || words.Contains("biệt") || words.Contains("bye"))
        {
            return "Tạm biệt.";
        }

        if (words.Contains("ai") || words.Contains("gì") || words.Contains("làm")
            || words.Contains("do") || words.Contains("can"))
        {
            return "Tôi là một agent nhỏ ghép từ 4 phần: bộ phân loại ý định (mạng neural), " +
                   "bộ nhớ, bộ suy luận ký hiệu, và ba công cụ (máy tính, tra cứu, cơ sở dữ liệu).";
        }

        return "Chào bạn. Tôi có thể tính toán, ghi nhớ, tra cứu và truy vấn dữ liệu.";
    }

    /// <summary>Bỏ phần "hãy nhớ rằng..." để chỉ lưu lại nội dung thật.</summary>
    private static string StripRememberPrefix(string input)
    {
        string[] prefixes =
        [
            "hãy nhớ rằng", "hãy nhớ là", "hãy nhớ", "ghi nhớ rằng", "ghi nhớ là", "ghi nhớ",
            "nhớ giúp tôi", "nhớ rằng", "nhớ là", "lưu lại", "hãy lưu",
            "please remember that", "please remember", "remember that", "remember", "save this",
        ];

        string text = input.Trim();

        foreach (string prefix in prefixes.OrderByDescending(p => p.Length))
        {
            if (text.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return text[prefix.Length..].Trim(' ', ':', ',', '.');
            }
        }

        return text;
    }
}
