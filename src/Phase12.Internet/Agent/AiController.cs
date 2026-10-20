using Ai.Phase12.Internet;
using Ai.Phase12.Memory;
using Ai.Phase12.Reasoning;
using Ai.Phase12.Tools;

namespace Ai.Phase12.Agent;

/// <summary>Một bước trong quá trình quyết định — để agent giải trình được.</summary>
public sealed record DecisionStep(string Question, string Decision);

public sealed record AgentResponse(
    string Answer,
    Intent Intent,
    double Confidence,
    string? ToolUsed,
    IReadOnlyList<DecisionStep> Decisions,
    string? Detail = null)
{
    /// <summary>
    /// Câu trả lời này có dựa trên nội dung Internet chưa kiểm chứng hay không.
    ///
    /// Đây là thông tin người dùng CÓ QUYỀN biết: "2 + 3 * 4 = 14" và "giá
    /// bitcoin hôm nay là X" có độ chắc chắn hoàn toàn khác nhau, dù cả hai
    /// đều được agent nói ra bằng cùng một giọng.
    /// </summary>
    public bool UsedUnverifiedInternetData { get; init; }

    /// <summary>Báo cáo lượt tra Internet, nếu có — Phase 13 sẽ dựng evidence từ đây.</summary>
    public Internet.InternetSearchReport? InternetReport { get; init; }
}

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
/// PHASE 12 mở thêm một nhánh: nếu câu hỏi cần thông tin BÊN NGOÀI thì agent
/// ra Internet. Nhánh này bất đồng bộ, và nó là nhánh DUY NHẤT trả về dữ liệu
/// chưa kiểm chứng — nên nó được gắn cờ riêng trong AgentResponse.
///
/// LUỒNG QUYẾT ĐỊNH:
///
///     1. Trả lời được bằng kiến thức sẵn có không?
///            KHÔNG ↓
///     2. Có cần Memory không?
///            KHÔNG ↓
///     3. Có cần Internet không?            <- PHASE 12
///            KHÔNG ↓
///     4. Có cần Tool không?
///            CÓ    ↓
///     5. Gọi Tool -> nhận kết quả -> Reasoning -> Answer
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
    private readonly InternetSearchTool? _internet;

    public AiController(
        IntentClassifier classifier,
        MemoryStore memory,
        CalculatorTool calculator,
        SearchTool search,
        DatabaseTool database,
        InternetSearchTool? internet = null)
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

        // Internet là TUỲ CHỌN. Agent phải chạy được đầy đủ khi không có mạng
        // — và khi không có, nó nói thẳng là không tra được, chứ không lặng lẽ
        // đi tra kho tài liệu nội bộ rồi trả lời như thể đã lên mạng.
        _internet = internet;
    }

    public MemoryStore Memory => _memory;

    public ReasoningEngine Reasoning => _calculator.Engine;

    public IReadOnlyList<ITool> Tools => [_calculator, _search, _database];

    /// <summary>Công cụ bất đồng bộ đã đăng ký (Phase 12: internet).</summary>
    public IReadOnlyList<IAsyncTool> AsyncTools => _internet is null ? [] : [_internet];

    public bool HasInternetAccess => _internet is not null;

    public async Task<AgentResponse> HandleAsync(string input, CancellationToken cancellationToken = default)
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
                "nhớ lại điều bạn đã nói, tra cứu khái niệm, truy vấn dữ liệu đơn hàng" +
                (HasInternetAccess ? ", và tra cứu trên Internet." : "."),
                prediction.Intent, prediction.Confidence, null, decisions);
        }

        // --- BƯỚC 2: có cần Memory không? ---
        if (prediction.Intent is Intent.RememberFact or Intent.RecallMemory)
        {
            decisions.Add(new DecisionStep("Có cần Memory?", "CÓ"));
            return HandleMemory(input, prediction, decisions);
        }

        decisions.Add(new DecisionStep("Có cần Memory?", "KHÔNG"));

        // --- BƯỚC 3: có cần Internet không? (PHASE 12) ---
        if (prediction.Intent == Intent.SearchInternet)
        {
            decisions.Add(new DecisionStep("Có cần Internet?", "CÓ"));
            return await HandleInternetAsync(input, prediction, decisions, cancellationToken).ConfigureAwait(false);
        }

        decisions.Add(new DecisionStep("Có cần Internet?", "KHÔNG"));

        // --- BƯỚC 4 & 5: cần Tool -> gọi -> suy luận -> trả lời ---
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

    /// <summary>
    /// Nhánh Internet.
    ///
    /// Ba điều nhánh này phải làm khác hẳn mọi nhánh còn lại:
    ///
    ///   1. NÓI RÕ đây là dữ liệu chưa kiểm chứng — trong câu trả lời, không
    ///      phải chỉ trong metadata.
    ///   2. KHÔNG ghi gì vào memory. Nội dung Internet không được tự động trở
    ///      thành ký ức lâu dài; đó là việc của Phase 13 sau khi qua evidence.
    ///   3. Giữ nguyên <see cref="InternetSearchReport"/> trong phản hồi, để
    ///      Phase 13–14 dựng được evidence và truy được nguồn.
    /// </summary>
    private async Task<AgentResponse> HandleInternetAsync(
        string input, IntentPrediction prediction, List<DecisionStep> decisions, CancellationToken cancellationToken)
    {
        if (_internet is null)
        {
            decisions.Add(new DecisionStep("Có công cụ Internet?", "KHÔNG — chưa được cấu hình"));

            return new AgentResponse(
                "Tôi chưa được cấp quyền truy cập Internet, nên không tra được thông tin mới. " +
                "Tôi chỉ trả lời được từ kho tài liệu nội bộ và những gì bạn đã bảo tôi ghi nhớ.",
                prediction.Intent, prediction.Confidence, null, decisions);
        }

        decisions.Add(new DecisionStep("Có công cụ Internet?", $"CÓ — dùng '{_internet.Name}'"));

        var result = await _internet.ExecuteAsync(input, cancellationToken).ConfigureAwait(false);
        var report = _internet.LastReport;

        if (!result.Success)
        {
            decisions.Add(new DecisionStep("Internet trả về gì?", $"THẤT BẠI — {result.Output}"));

            return new AgentResponse(
                $"Tôi đã thử tra Internet nhưng không được: {result.Output}. " +
                "Tôi không có thông tin nào để trả lời câu này.",
                prediction.Intent, prediction.Confidence, _internet.Name, decisions, result.Detail)
            {
                InternetReport = report,
            };
        }

        decisions.Add(new DecisionStep("Internet trả về gì?",
            $"{report?.Sources.Count ?? 0} nguồn, {report?.Documents.Count ?? 0} trang đã tải"));

        decisions.Add(new DecisionStep("Có được coi là sự thật?",
            "KHÔNG — dữ liệu ngoài, chưa kiểm chứng (kiểm chứng là Phase 14)"));

        return new AgentResponse(result.Output, prediction.Intent, prediction.Confidence,
            _internet.Name, decisions, result.Detail)
        {
            UsedUnverifiedInternetData = true,
            InternetReport = report,
        };
    }

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

    private string SmallTalkReply(string input)
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
                   "bộ nhớ, bộ suy luận ký hiệu, và các công cụ (máy tính, tra cứu, cơ sở dữ liệu" +
                   (HasInternetAccess ? ", Internet)." : ").");
        }

        return "Chào bạn. Tôi có thể tính toán, ghi nhớ, tra cứu và truy vấn dữ liệu" +
               (HasInternetAccess ? ", và tra cứu trên Internet." : ".");
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
