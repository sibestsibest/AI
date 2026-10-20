using Ai.Phase16.Agent;
using Ai.Phase16.Feedback;
using Ai.Phase16.Internet;
using Ai.Phase16.Learning;
using Ai.Phase16.Memory;
using Ai.Phase16.Retraining;
using Ai.Phase16.Tools;
using Ai.Phase16.Verification;
using Ai.Providers;
using Ai.Providers.Configuration;
using Ai.Providers.Context;
using Ai.Providers.Integration;
using Ai.Providers.OpenAi;
using Ai.Providers.Orchestration;
using Ai.Providers.Routing;
using Ai.Providers.Tools;

namespace Ai.Chat;

/// <summary>Một lượt hỏi–đáp, kèm id để phản hồi gắn vào được.</summary>
public sealed record ChatTurn(string Question, AgentResponse Response, AnswerRecord Record)
{
    /// <summary>
    /// Lượt này đã phải đi ĐƯỜNG LUI: bộ phân loại ý định không nhận ra câu
    /// hỏi, nên câu hỏi được đưa thẳng vào đường ống kiểm chứng của Phase 14.
    /// </summary>
    public bool UsedFallback { get; init; }

    /// <summary>Ý định mà bộ phân loại đã đoán trước khi đi đường lui, kèm độ tự tin.</summary>
    public IntentPrediction? RoutedAs { get; init; }
}

/// <summary>Vài con số về trạng thái hiện tại, để lệnh /trangthai in ra.</summary>
public sealed record SessionStatus(
    int ParameterCount,
    string ModelLabel,
    int DatasetVersion,
    string DatasetFingerprint,
    int TurnCount,
    int MemoryCount,
    int KnowledgeCount,
    int FeedbackCount,
    int ErrorCaseCount);

/// <summary>
/// PHIÊN CHAT — giữ toàn bộ trạng thái, và KHÔNG biết gì về màn hình.
///
/// Tách khỏi phần in ấn (<see cref="ChatConsole"/>) vì hai lý do thực dụng:
/// phần này test được mà không cần giả lập Console, và nếu sau này có ai muốn
/// bọc một giao diện khác (web, TUI) thì chỉ phải viết lại phần hiển thị.
///
/// Phiên chat nối đủ bốn thứ mà Phase 11–16 đã dựng:
///
///   agent      trả lời (ý định -> công cụ / căn cứ -> kiểm chứng)
///   journal    cấp id cho câu trả lời, để chấm được
///   learning   phản hồi -> ca lỗi -> ứng viên -> phiên bản dữ liệu mới
///   retraining dữ liệu -> model mới -> cửa đánh giá -> triển khai / từ chối
///
/// Nhờ vậy màn hình này không chỉ để hỏi đáp: nó chạy được TRỌN VẸN vòng
/// người dùng sửa lỗi cho agent, ngay trong một phiên.
/// </summary>
public sealed class ChatSession
{
    private readonly IClock _clock;
    private readonly InternetSearchTool _internet;
    private readonly LearningStack _learning;
    private readonly RetrainingPipeline _retraining;
    private readonly List<ChatTurn> _turns = [];

    private IntentClassifier _classifier;
    private AiController _agent;
    private LlmAnswerGenerator? _llmGenerator;

    private ChatSession(IntentClassifier classifier, IClock clock)
    {
        _clock = clock;
        _classifier = classifier;
        _internet = InternetStackFactory.BuildOffline(clock);
        _agent = AgentFactory.Build(classifier, clock, _internet);

        _learning = LearningStackFactory.Build(clock);
        Registry = new ModelRegistry(clock);
        _retraining = new RetrainingPipeline(Registry, clock: clock);

        // Lớp provider được DỰNG nhưng CHƯA DÙNG: agent ở trên vẫn là agent
        // dùng bộ sinh theo khuôn của Phase 14. Muốn LLM tham gia thì phải gọi
        // EnableLlmAsync — mặc định tắt, nên hành vi mặc định của màn hình y
        // như trước khi có lớp này.
        //
        // OpenAiProviders.Create là chỗ DUY NHẤT màn hình biết OpenAI tồn tại.
        // Không truyền nó thì loại "OpenAiResponses" trong cấu hình báo chưa
        // hỗ trợ, và mọi thứ còn lại chạy như cũ.
        Ai = AiProviderFactory.Build(clock: clock, extraCreator: OpenAiProviders.Create);
    }

    /// <summary>
    /// Dựng phiên chat. Bước tốn thời gian duy nhất là huấn luyện bộ phân loại
    /// ý định (3000 epoch, vài giây) — nên nó nhận một callback để màn hình báo
    /// tiến độ thay vì đứng im.
    /// </summary>
    public static ChatSession Start(Action<string>? onProgress = null)
    {
        onProgress?.Invoke("đang huấn luyện bộ phân loại ý định (3000 epoch)…");

        // Dùng CẢ 70 ví dụ, giống hệt agent trong các demo. Model của Phase 16
        // train trên 53 ví dụ (20% để dành đánh giá), nên nó yếu hơn ở một số
        // câu — muốn dùng nó thì gõ /trainlai, và màn hình sẽ nói rõ là đã đổi.
        var classifier = AgentFactory.TrainClassifier();

        onProgress?.Invoke($"xong — {classifier.ParameterCount} tham số, từ điển {classifier.Vocabulary.Count} từ");

        return new ChatSession(classifier, new SystemClock());
    }

    public ModelRegistry Registry { get; }

    /// <summary>Lớp provider (LLM ngoài). Luôn được dựng, nhưng chỉ được dùng khi <see cref="LlmEnabled"/>.</summary>
    public AiProviderStack Ai { get; }

    /// <summary>LLM ngoài có đang tham gia khâu diễn đạt câu trả lời hay không.</summary>
    public bool LlmEnabled => _llmGenerator is not null;

    /// <summary>Chuyện gì đã xảy ra ở lần diễn đạt gần nhất — null nếu chưa bật LLM.</summary>
    public LlmComposeDiagnostics? LastCompose => _llmGenerator?.Last;

    /// <summary>Số lần văn bản LLM được dùng / bị loại vì không qua kiểm chứng.</summary>
    public (int Used, int Rejected) LlmComposeCounts =>
        _llmGenerator is null ? (0, 0) : (_llmGenerator.UsedCount, _llmGenerator.RejectedCount);

    public IntentClassifier Classifier => _classifier;

    public AiController Agent => _agent;

    public IReadOnlyList<ChatTurn> Turns => _turns;

    public ChatTurn? LastTurn => _turns.Count > 0 ? _turns[^1] : null;

    /// <summary>Lượt cuối đã được chấm chưa — để không hỏi lại người dùng một câu đã chấm.</summary>
    public bool LastTurnRated =>
        LastTurn is not null && _learning.Feedback.All.Any(f => f.AnswerId == LastTurn.Record.Id);

    public async Task<ChatTurn> AskAsync(string question, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(question);

        // PHÂN LOẠI TRƯỚC, RỒI MỚI CHỌN ĐƯỜNG — và thứ tự này không đảo được.
        //
        // Bản đầu của màn hình này gọi HandleAsync trước rồi mới đi đường lui
        // nếu độ tự tin thấp. Hậu quả là một lỗi thật, nhìn thấy ngay trên màn
        // hình: HandleAsync ghi lượt hội thoại vào ConversationContext TRƯỚC,
        // nên khi đường lui gọi đường ống kiểm chứng, chính câu hỏi vừa gõ đã
        // nằm trong ngữ cảnh — và agent dẫn nó ra làm CĂN CỨ:
        //
        //     [SỰ THẬT] SSRF là gì
        //       ↳ bạn vừa nói ở lượt #1  [tin cậy 0.75]
        //
        // Đó đúng là kiểu lập luận vòng tròn mà Phase 14 đã chặn bằng cách ghi
        // lượt SAU khi trả lời. Gọi hai đường liên tiếp làm sống lại nó.
        var routed = _classifier.Classify(question);

        AgentResponse response;
        bool usedFallback = false;

        if (routed.IsConfident)
        {
            response = await _agent.HandleAsync(question, cancellationToken).ConfigureAwait(false);
        }
        else
        {
            response = await AnswerWithoutRoutingAsync(question, routed, cancellationToken).ConfigureAwait(false);
            usedFallback = true;
        }

        // Ghi vào sổ NGAY, trước khi in ra: phản hồi của người dùng đến sau đó
        // chỉ mang theo một cái id, nên id phải tồn tại từ trước.
        var record = _learning.Journal.Record(question, response);
        var turn = new ChatTurn(question, response, record) { UsedFallback = usedFallback, RoutedAs = routed };

        _turns.Add(turn);
        return turn;
    }

    /// <summary>
    /// ĐƯỜNG LUI khi bộ phân loại ý định không nhận ra câu hỏi.
    ///
    /// VÌ SAO CẦN, VÀ VÌ SAO NÓ KHÔNG PHẢI LÀ NỚI LUẬT
    ///
    /// Bộ phân loại là một bộ ĐỊNH TUYẾN học từ 70 ví dụ. Với câu "SSRF là gì"
    /// thì sau khi bỏ từ dừng chỉ còn một từ khoá — "ssrf" — và từ đó không có
    /// trong từ điển, nên vector bag-of-words rỗng và độ tự tin bằng 0. Với
    /// câu "2 + 3 * 4" thì mọi token đều dài một ký tự nên bị bỏ hết: cũng
    /// không còn từ nào. Agent trả lời "tôi không chắc bạn muốn gì" — đúng
    /// theo luật của Phase 11, và đúng là nó không định tuyến được.
    ///
    /// Nhưng hệ thống BIẾT trả lời cả hai câu đó: kho kiến thức có tài liệu
    /// SSRF, và engine ký hiệu của Phase 9 tính được biểu thức. Chỉ có bước
    /// định tuyến là mù. Khi định tuyến mù, đi thẳng vào đường ống kiểm chứng
    /// (thu thập căn cứ 5 kênh -> suy luận -> kiểm chứng) là lựa chọn đúng.
    ///
    /// Đây KHÔNG phải nới luật, vì đường ống Phase 14 tự có chốt riêng: căn cứ
    /// không đủ thì nó trả về <see cref="AnswerVerdict.Insufficient"/> và nói
    /// thẳng là không biết. Nên đường lui không thể bịa ra câu trả lời — nó chỉ
    /// bỏ qua một bước định tuyến đang vô dụng. Câu "cách nấu phở bò Hà Nội"
    /// vẫn bị trả về "không đủ căn cứ", y như trước.
    ///
    /// Và câu hỏi chỉ đi qua ĐÚNG MỘT đường, không phải hai đường nối tiếp —
    /// nếu không thì lượt hội thoại bị ghi hai lần vào ngữ cảnh.
    ///
    /// Khi đường ống cũng không đủ căn cứ thì giữ nguyên câu từ chối CỦA NÓ
    /// ("không đủ căn cứ để trả lời"), chứ không quay lại lấy câu "tôi không
    /// chắc bạn muốn gì" của Phase 11. Câu của đường ống nói đúng hơn về việc
    /// vừa xảy ra: agent đã đi tìm và không thấy, chứ không phải không hiểu.
    /// </summary>
    private async Task<AgentResponse> AnswerWithoutRoutingAsync(string question,
        IntentPrediction routed, CancellationToken cancellationToken)
    {
        var grounded = await _agent.AnswerAsync(question, cancellationToken).ConfigureAwait(false);

        var decisions = new List<DecisionStep>
        {
            new("Người dùng muốn gì?",
                $"{routed.Intent} (độ tự tin {routed.Confidence:P0}, nhận ra " +
                $"{_classifier.KnownWordCount(question)} từ quen) — KHÔNG đủ để định tuyến"),
            new("Định tuyến không được thì làm gì?",
                "đi thẳng vào đường ống kiểm chứng của Phase 14 (5 kênh căn cứ -> suy luận -> kiểm chứng)"),
            new("Kiểm chứng ra kết luận gì?",
                $"{grounded.Verdict} (tự tin {grounded.Confidence:P0}, dải {grounded.Band})"),
        };

        return new AgentResponse(grounded.Text, routed.Intent, routed.Confidence,
            "verification (đường lui)", decisions, string.Join("\n  ", grounded.Summary.Decisions))
        {
            UsedUnverifiedInternetData = grounded.RequiresDisclosure,
            Grounded = grounded,
        };
    }

    /// <summary>Chấm lượt cuối. Trả về lý do nếu bị từ chối (chưa hỏi gì, hoặc đã chấm rồi).</summary>
    public FeedbackResult Rate(FeedbackRating rating, Intent? expectedIntent = null, string? correction = null)
    {
        if (LastTurn is null)
        {
            return new FeedbackResult(false, "chưa có câu trả lời nào để chấm");
        }

        return _learning.Feedback.Submit(LastTurn.Record.Id, rating, correction, expectedIntent);
    }

    /// <summary>
    /// Chạy vòng học của Phase 15: phản hồi -> ca lỗi -> ứng viên -> kiểm -> tập dữ liệu mới.
    ///
    /// KHÔNG chạm tới trọng số. Sau lệnh này agent vẫn trả lời y như trước —
    /// muốn đổi hành vi thì phải <see cref="RetrainAsync"/>.
    /// </summary>
    public LearningReport Learn() =>
        _learning.Pipeline.Run(_classifier.Vocabulary, "học từ phiên chat");

    /// <summary>
    /// Chạy vòng huấn luyện lại của Phase 16 trên tập dữ liệu hiện tại.
    ///
    /// Chỉ khi cửa đánh giá cho qua thì model mới được áp dụng vào phiên chat.
    /// Bị từ chối thì agent giữ nguyên model cũ — đúng luật của Phase 16, và
    /// người dùng nhìn thấy điều đó xảy ra chứ không phải chỉ đọc mô tả.
    /// </summary>
    public Task<RetrainingReport> RetrainAsync(TrainingConfig? config = null) =>
        Task.Run(() =>
        {
            var report = _retraining.Run(_learning.Datasets.Current, config ?? new TrainingConfig(),
                "huấn luyện lại từ màn hình chat");

            if (report.Deployed)
            {
                // Dựng lại agent quanh model vừa được duyệt. Bộ nhớ hội thoại
                // bắt đầu lại — màn hình nói rõ chuyện đó, vì im lặng ở đây sẽ
                // làm người dùng tưởng agent vừa quên mất những gì họ kể.
                _classifier = report.Candidate.Restore();
                _agent = AgentFactory.Build(_classifier, _clock, _internet, answerGenerator: _llmGenerator);
            }

            return report;
        });

    public SessionStatus Status() => new(
        _classifier.ParameterCount,
        Registry.Active?.Label ?? "chưa qua huấn luyện lại (model gốc của Phase 11)",
        _learning.Datasets.Current.Version,
        _learning.Datasets.Current.Fingerprint,
        _turns.Count,
        _agent.Memory.LongTerm.Count,
        _agent.Knowledge?.Count ?? 0,
        _learning.Feedback.Count,
        _learning.Pipeline.ErrorCases.Count);

    /// <summary>Báo cáo hiệu chuẩn: độ tự tin agent tự nhận có khớp với thực tế không.</summary>
    public CalibrationReport Calibration() => _learning.Feedback.Calibrate();

    public IReadOnlyList<ErrorCase> TopErrorCases(int take = 5) => _learning.Pipeline.TopErrorCases(take);

    /// <summary>Bắt đầu lại hội thoại: quên bộ nhớ và ngữ cảnh, GIỮ model và tập dữ liệu.</summary>
    public void ResetConversation()
    {
        _agent = AgentFactory.Build(_classifier, _clock, _internet, answerGenerator: _llmGenerator);
        _turns.Clear();
    }

    /// <summary>
    /// BẬT LLM ngoài cho khâu diễn đạt.
    ///
    /// Kiểm tra provider TRƯỚC khi bật, và không bật nếu không có provider nào
    /// sẵn sàng. Bật rồi mới phát hiện không có gì phía sau sẽ khiến mọi câu
    /// trả lời phải chờ hết hạn thời gian rồi mới quay về văn bản khuôn mẫu —
    /// hệ thống vẫn đúng, nhưng chậm đi mà không ai hiểu vì sao.
    ///
    /// Trả về lý do nếu không bật được, null nếu bật thành công.
    /// </summary>
    public async Task<string?> EnableLlmAsync(CancellationToken cancellationToken = default)
    {
        if (Ai.Providers.Count == 0)
        {
            return $"không có provider nào được cấu hình (nguồn cấu hình: {Ai.ConfigSource})";
        }

        if (!await Ai.Router.HasAvailableProviderAsync(cancellationToken).ConfigureAwait(false))
        {
            return "không provider nào sẵn sàng — với Ollama hãy kiểm tra máy chủ đã chạy và đã có model";
        }

        _llmGenerator = Ai.CreateAnswerGenerator();
        _agent = AgentFactory.Build(_classifier, _clock, _internet, answerGenerator: _llmGenerator);

        return null;
    }

    /// <summary>Tắt LLM: dựng lại agent với bộ sinh theo khuôn của Phase 14.</summary>
    public void DisableLlm()
    {
        _llmGenerator = null;
        _agent = AgentFactory.Build(_classifier, _clock, _internet);
    }

    /// <summary>Id hội thoại cho lớp điều phối — giữ nguyên suốt phiên để nối lượt được.</summary>
    public string ConversationId { get; } = Guid.NewGuid().ToString("N")[..12];

    /// <summary>Chế độ trả lời đang dùng cho <see cref="AskModelAsync"/>.</summary>
    public AiMode Mode { get; set; } = AiMode.Auto;

    /// <summary>
    /// HỎI THẲNG MODEL, qua đường điều phối có công cụ.
    ///
    /// Đây là đường ĐỘC LẬP với <see cref="AskAsync"/>, và hai đường trả lời
    /// hai câu hỏi khác nhau:
    ///
    ///   AskAsync       "điều này có đúng không" — phân loại ý định, thu thập
    ///                  căn cứ 5 kênh, kiểm chứng, truy nguồn từng câu. Đường
    ///                  MẶC ĐỊNH của màn hình, và không đổi.
    ///
    ///   AskModelAsync  "làm hộ tôi việc này" — dựng ngữ cảnh, để model gọi
    ///                  công cụ của MiniAI, rồi kiểm câu trả lời.
    ///
    /// Sổ đăng ký công cụ và bộ dựng ngữ cảnh được dựng LẠI mỗi lượt từ agent
    /// hiện tại, vì agent bị thay mới sau mỗi lần huấn luyện lại hoặc reset.
    /// Giữ một bản dựng sẵn sẽ khiến model gọi công cụ của agent đã bị vứt.
    /// </summary>
    public async Task<ChatResult> AskModelAsync(string message, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(message);

        var registry = PhaseTools.ForAgent(_agent.Tools, _agent.AsyncTools);

        var context = new ContextBuilder(_agent.Memory, _agent.Context);

        var models = Ai.Options.EnabledProviders
            .Select(p => p.TaskModels)
            .FirstOrDefault(m => !m.IsEmpty) ?? new AiModelMap();

        var orchestrator = new ChatOrchestrator(Ai.Router, registry, context, models: models);

        var result = await orchestrator
            .AskAsync(new ChatRequest
            {
                Message = message,
                ConversationId = ConversationId,
                Mode = Mode,
            }, cancellationToken)
            .ConfigureAwait(false);

        // Ghi lượt vào ngữ cảnh SAU khi đã trả lời — cùng luật với
        // AiController.HandleAsync. Ghi trước thì chính câu hỏi đang xử lý sẽ
        // nằm trong ngữ cảnh của lượt sau và thành căn cứ cho chính nó.
        if (result.Success)
        {
            _agent.Context?.Record(message, result.Message);
        }

        return result;
    }
}
