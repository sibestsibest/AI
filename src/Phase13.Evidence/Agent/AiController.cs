using Ai.Phase13.Grounding;
using Ai.Phase13.Internet;
using Ai.Phase13.Memory;
using Ai.Phase13.Reasoning;
using Ai.Phase13.Tools;

namespace Ai.Phase13.Agent;

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

    /// <summary>Báo cáo lượt tra Internet, nếu có.</summary>
    public Internet.InternetSearchReport? InternetReport { get; init; }

    /// <summary>
    /// Căn cứ đã dùng để dựng câu trả lời (PHASE 13).
    ///
    /// Đây là thứ phân biệt một câu trả lời CÓ CƠ SỞ với một câu trả lời chỉ
    /// nghe có vẻ đúng: mỗi khẳng định truy được về một mẩu evidence, mỗi mẩu
    /// evidence truy được về một nguồn. Phase 14 sẽ kiểm chứng dựa trên đúng
    /// tập này.
    /// </summary>
    public Grounding.EvidenceSet? Evidence { get; init; }

    /// <summary>Số thứ tự lượt hội thoại của câu trả lời này.</summary>
    public int TurnIndex { get; init; }
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
/// Phase 12 mở thêm một nhánh: nếu câu hỏi cần thông tin BÊN NGOÀI thì agent
/// ra Internet. Nhánh này bất đồng bộ, và nó là nhánh DUY NHẤT trả về dữ liệu
/// chưa kiểm chứng — nên nó được gắn cờ riêng trong AgentResponse.
///
/// LUỒNG QUYẾT ĐỊNH:
///
///     1. Trả lời được bằng kiến thức sẵn có không?
///            KHÔNG ↓
///     2. Có cần Memory không?
///            KHÔNG ↓
///     3. Có cần Internet không?            <- Phase 12
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
    private readonly KnowledgeStore? _knowledge;
    private readonly ConversationContext? _context;
    private readonly IEvidenceCollector? _evidence;

    public AiController(
        IntentClassifier classifier,
        MemoryStore memory,
        CalculatorTool calculator,
        SearchTool search,
        DatabaseTool database,
        InternetSearchTool? internet = null,
        KnowledgeStore? knowledge = null,
        ConversationContext? context = null,
        IEvidenceCollector? evidence = null)
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

        // PHASE 13 — cũng đều là tuỳ chọn, vì mọi hành vi của Phase 11 và 12
        // phải còn nguyên khi chúng vắng mặt.
        _knowledge = knowledge;
        _context = context;
        _evidence = evidence;
    }

    public MemoryStore Memory => _memory;

    /// <summary>Kho kiến thức nội bộ (PHASE 13).</summary>
    public KnowledgeStore? Knowledge => _knowledge;

    /// <summary>Ngữ cảnh hội thoại (PHASE 13).</summary>
    public ConversationContext? Context => _context;

    public bool HasEvidenceCollector => _evidence is not null;

    public ReasoningEngine Reasoning => _calculator.Engine;

    public IReadOnlyList<ITool> Tools => [_calculator, _search, _database];

    /// <summary>Công cụ bất đồng bộ đã đăng ký (Phase 12: internet).</summary>
    public IReadOnlyList<IAsyncTool> AsyncTools => _internet is null ? [] : [_internet];

    public bool HasInternetAccess => _internet is not null;

    /// <summary>
    /// Xử lý một lượt của người dùng.
    ///
    /// PHASE 13 bọc thêm một việc quanh luồng cũ: GHI LẠI LƯỢT NÓI. Thứ tự
    /// quan trọng — lượt được ghi SAU khi đã trả lời, không phải trước. Nếu
    /// ghi trước, chính câu hỏi đang xử lý sẽ nằm trong ngữ cảnh của bước thu
    /// thập evidence, và agent sẽ lấy câu hỏi của người dùng làm căn cứ để
    /// trả lời câu hỏi đó — một kiểu lập luận vòng tròn.
    /// </summary>
    public async Task<AgentResponse> HandleAsync(string input, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);

        var response = await HandleCoreAsync(input, cancellationToken).ConfigureAwait(false);

        if (_context is null) return response;

        var turn = _context.Record(input, response.Answer, response.Intent, response.UsedUnverifiedInternetData);

        return response with { TurnIndex = turn.Index };
    }

    /// <summary>
    /// Thu thập căn cứ cho một câu hỏi mà KHÔNG trả lời nó.
    ///
    /// Đây là điểm nối Phase 13 -> Phase 14: Phase 14 sẽ gọi hàm này, rồi
    /// kiểm chứng và dựng câu trả lời từ tập căn cứ trả về.
    /// </summary>
    public async Task<EvidenceSet> CollectEvidenceAsync(string question, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(question);

        if (_evidence is null)
        {
            throw new InvalidOperationException(
                "Agent này chưa được cấu hình EvidenceCollector — xem AgentFactory.Build.");
        }

        return await _evidence.CollectAsync(question, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AgentResponse> HandleCoreAsync(string input, CancellationToken cancellationToken)
    {
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

        // --- BƯỚC 3: có cần Internet không? (Phase 12) ---
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
    /// Nhánh Internet — từ PHASE 13 thì nó đi qua EVIDENCE chứ không dùng thẳng
    /// kết quả tra web nữa.
    ///
    /// Khác biệt so với Phase 12 là khác biệt về BẢN CHẤT, không phải về cách
    /// trình bày. Ở Phase 12, hỏi "tìm trên mạng về SSRF" thì agent trả về đúng
    /// những gì web nói, kèm cảnh báo chưa kiểm chứng. Ở Phase 13, cùng câu hỏi
    /// đó đi thu thập căn cứ từ CẢ NĂM kênh — và kho nội bộ cũng có tài liệu về
    /// SSRF. Kết quả: hai kênh độc lập cùng xác nhận, kênh đã kiểm được xếp
    /// trước, và câu trả lời KHÔNG còn phải cảnh báo nữa.
    ///
    /// Đó chính là điều Phase 13 thêm vào: không phải "lấy được nhiều thông tin
    /// hơn", mà "biết thông tin nào đáng tin hơn, và vì sao".
    ///
    /// Ba điều nhánh này vẫn giữ nguyên từ Phase 12:
    ///
    ///   1. NÓI RÕ khi dữ liệu chưa kiểm chứng — trong câu trả lời, không phải
    ///      chỉ trong metadata. Nay có thêm điều kiện: chỉ cảnh báo khi KHÔNG
    ///      có nguồn đã kiểm nào đỡ lưng.
    ///   2. KHÔNG ghi gì vào memory hay knowledge.
    ///   3. Giữ nguyên dấu vết để truy nguồn — nay là cả EvidenceSet.
    /// </summary>
    private async Task<AgentResponse> HandleInternetAsync(
        string input, IntentPrediction prediction, List<DecisionStep> decisions, CancellationToken cancellationToken)
    {
        // --- Không có kênh nào ra ngoài được thì nói thẳng ---
        if (_internet is null && _evidence is null)
        {
            decisions.Add(new DecisionStep("Có công cụ Internet?", "KHÔNG — chưa được cấu hình"));

            return new AgentResponse(
                "Tôi chưa được cấp quyền truy cập Internet, nên không tra được thông tin mới. " +
                "Tôi chỉ trả lời được từ kho tài liệu nội bộ và những gì bạn đã bảo tôi ghi nhớ.",
                prediction.Intent, prediction.Confidence, null, decisions);
        }

        // --- Đường của PHASE 13: thu thập căn cứ từ mọi kênh ---
        if (_evidence is not null)
        {
            decisions.Add(new DecisionStep("Thu thập căn cứ từ đâu?", "mọi kênh: knowledge, memory, context, internet"));

            var set = await _evidence.CollectAsync(input, cancellationToken).ConfigureAwait(false);

            if (!set.HasEvidence)
            {
                string why = set.Warnings.Count > 0 ? string.Join("; ", set.Warnings) : "không có kênh nào trả về gì";

                decisions.Add(new DecisionStep("Có căn cứ nào không?", $"KHÔNG — {why}"));

                return new AgentResponse(
                    $"Tôi không tìm được căn cứ nào để trả lời câu này ({why}).",
                    prediction.Intent, prediction.Confidence, "evidence", decisions)
                {
                    Evidence = set,
                    InternetReport = set.InternetReport,
                };
            }

            decisions.Add(new DecisionStep("Có căn cứ nào không?",
                $"CÓ — {set.Items.Count} mẩu từ {set.DistinctChannelCount} kênh" +
                (set.DuplicatesRemoved > 0 ? $", đã gộp {set.DuplicatesRemoved} mẩu trùng" : "")));

            decisions.Add(new DecisionStep("Có nguồn đã kiểm chứng?",
                set.HasVerifiedEvidence
                    ? "CÓ — kho nội bộ xác nhận, không cần cảnh báo"
                    : "KHÔNG — chỉ có nguồn ngoài, phải cảnh báo rõ"));

            return new AgentResponse(ComposeFromEvidence(set), prediction.Intent, prediction.Confidence,
                "evidence", decisions, ExplainEvidence(set))
            {
                UsedUnverifiedInternetData = set.RequiresDisclosure,
                Evidence = set,
                InternetReport = set.InternetReport,
            };
        }

        // --- Đường của Phase 12: chỉ có công cụ Internet, không có evidence ---
        decisions.Add(new DecisionStep("Có công cụ Internet?", $"CÓ — dùng '{_internet!.Name}'"));

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

    /// <summary>
    /// Ghép các mẩu căn cứ thành câu trả lời — KHÔNG kết luận, KHÔNG diễn giải.
    ///
    /// Mỗi mẩu đi kèm trích dẫn nguồn của chính nó. Bỏ trích dẫn đi là bỏ luôn
    /// khả năng truy nguồn, và một khẳng định không truy được nguồn thì không
    /// khác gì bịa — dù nó tình cờ đúng.
    ///
    /// Việc rút ra một câu trả lời duy nhất từ tập căn cứ này, và kiểm xem các
    /// mẩu có mâu thuẫn nhau không, là của Phase 14.
    /// </summary>
    private static string ComposeFromEvidence(EvidenceSet set)
    {
        var builder = new System.Text.StringBuilder();

        builder.Append("Dựa trên ").Append(set.Items.Count).Append(" căn cứ từ ")
            .Append(set.DistinctChannelCount).Append(" kênh");

        builder.AppendLine(set.RequiresDisclosure
            ? ", TẤT CẢ đều từ nguồn ngoài CHƯA KIỂM CHỨNG:"
            : " (có nguồn đã kiểm chứng):");

        foreach (var item in set.Items.Take(3))
        {
            builder.AppendLine()
                .Append("  • ").Append(item.Preview(240)).AppendLine()
                .Append("    ").Append(item.Source.Cite())
                .Append("  [tin cậy ").Append(item.Reliability.ToString("F2")).Append(']');

            if (item.Agreement > 1)
            {
                builder.Append(", ").Append(item.Agreement).Append(" kênh cùng xác nhận");
            }

            builder.AppendLine();
        }

        if (set.Warnings.Count > 0)
        {
            builder.AppendLine().Append("  Lưu ý: ").Append(string.Join("; ", set.Warnings));
        }

        return builder.ToString().TrimEnd();
    }

    /// <summary>Dấu vết chấm điểm căn cứ — để giải trình được vì sao xếp thứ tự như vậy.</summary>
    private static string ExplainEvidence(EvidenceSet set)
    {
        var builder = new System.Text.StringBuilder();

        if (set.ExpandedQuestion is not null)
        {
            builder.Append("câu hỏi được ngữ cảnh bổ sung thành: \"").Append(set.ExpandedQuestion).AppendLine("\"");
        }

        foreach (var item in set.Items)
        {
            builder.Append("  ").Append(item.Score.ToString("F3")).Append("  ").AppendLine(item.Explanation);
        }

        return builder.ToString().TrimEnd();
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
