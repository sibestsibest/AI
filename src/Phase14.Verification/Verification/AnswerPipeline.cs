using Ai.Phase14.Grounding;
using Ai.Phase14.Memory;
using Ai.Phase14.Reasoning;

namespace Ai.Phase14.Verification;

public interface IAnswerPipeline
{
    Task<GroundedAnswer> AnswerAsync(string question, CancellationToken cancellationToken = default);
}

public sealed record AnswerPipelineOptions
{
    /// <summary>Số mẩu căn cứ được dẫn trong câu trả lời.</summary>
    public int MaxCitedEvidence { get; init; } = 3;

    /// <summary>
    /// Có thử bước suy luận ký hiệu (Phase 9) hay không.
    ///
    /// Bước này chỉ có tác dụng với câu hỏi số học; câu hỏi khái niệm thì engine
    /// sẽ không phân tích được và bước này lặng lẽ bỏ qua.
    /// </summary>
    public bool UseReasoning { get; init; } = true;

    /// <summary>
    /// Câu trả lời không qua được kiểm tra cuối thì HẠ cấp kết luận thay vì
    /// vẫn phát ra.
    ///
    /// Mặc định BẬT. Tắt nó đi chỉ để test quan sát được câu trả lời xấu.
    /// </summary>
    public bool EnforceValidation { get; init; } = true;
}

/// <summary>
/// ĐƯỜNG ỐNG TRẢ LỜI — dây chuyền đầy đủ của đề bài Phase 14.
///
///     Câu hỏi
///        ↓
///     Retrieve + Collect Evidence     EvidenceCollector (Phase 13, 5 kênh)
///        ↓
///     Reason                          ReasoningEngine (Phase 9) -> evidence Derived
///        ↓
///     Verify                          ContradictionDetector + ConfidenceEstimator
///        ↓
///     Generate Answer                 AnswerGenerator (theo khuôn, từ căn cứ)
///        ↓
///     Validate                        AnswerValidator (chốt chặn cuối)
///        ↓
///     GroundedAnswer
///
/// THỨ TỰ BA BƯỚC CUỐI KHÔNG ĐẢO ĐƯỢC, và lý do đáng nói:
///
///   • Verify TRƯỚC Generate — vì kết luận (trả lời / không chắc / mâu thuẫn /
///     không đủ) quyết định VIẾT GÌ. Sinh câu trả lời trước rồi mới kiểm thì
///     đã có một câu trả lời dứt khoát tồn tại, và sức ép sẽ là sửa nó cho đỡ
///     dứt khoát — thay vì ngay từ đầu không viết nó ra.
///
///   • Validate SAU Generate — vì nó kiểm CHÍNH VĂN BẢN sắp phát ra, chứ
///     không kiểm ý định của bộ sinh.
/// </summary>
public sealed class AnswerPipeline : IAnswerPipeline
{
    private readonly IEvidenceCollector _collector;
    private readonly ReasoningEngine? _reasoning;
    private readonly IContradictionDetector _contradictions;
    private readonly ConfidenceEstimator _confidence;
    private readonly IAnswerGenerator _generator;
    private readonly IAnswerValidator _validator;
    private readonly IClock _clock;
    private readonly AnswerPipelineOptions _options;

    public AnswerPipeline(
        IEvidenceCollector collector,
        ReasoningEngine? reasoning = null,
        IContradictionDetector? contradictions = null,
        ConfidenceEstimator? confidence = null,
        IAnswerGenerator? generator = null,
        IAnswerValidator? validator = null,
        IClock? clock = null,
        AnswerPipelineOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(collector);

        _collector = collector;
        _reasoning = reasoning;
        _contradictions = contradictions ?? new ContradictionDetector();
        _confidence = confidence ?? new ConfidenceEstimator();
        _generator = generator ?? new AnswerGenerator();
        _validator = validator ?? new AnswerValidator();
        _clock = clock ?? new SystemClock();
        _options = options ?? new AnswerPipelineOptions();
    }

    public async Task<GroundedAnswer> AnswerAsync(string question, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(question);

        var startedAt = _clock.Now;
        var decisions = new List<string>();

        // ---------- 1 & 2. Retrieve + Collect Evidence ----------
        var evidence = await _collector.CollectAsync(question, cancellationToken).ConfigureAwait(false);

        decisions.Add($"gom được {evidence.Items.Count} căn cứ từ {evidence.DistinctChannelCount} kênh");

        // ---------- 3. Reason ----------
        var items = evidence.Items.ToList();

        if (_options.UseReasoning && _reasoning is not null)
        {
            var derived = TryReason(question);

            if (derived is not null)
            {
                items.Insert(0, derived);
                decisions.Add("suy luận ký hiệu cho ra một kết quả -> thêm làm căn cứ Derived");
            }
        }

        if (items.Count == 0)
        {
            decisions.Add("không có căn cứ -> KHÔNG trả lời");

            return BuildEmpty(question, evidence, startedAt, decisions);
        }

        // ---------- 4. Verify ----------
        var contradictions = _contradictions.Detect(items);

        if (contradictions.Count > 0)
        {
            decisions.Add($"phát hiện {contradictions.Count} mâu thuẫn, nặng nhất {contradictions[0].Severity:F2}");
        }

        var cited = items.Take(_options.MaxCitedEvidence).ToList();
        var claims = _generator.ExtractClaims(evidence with { Items = items }, cited);

        var assessment = _confidence.Estimate(evidence with { Items = items }, claims, contradictions);
        var verdict = _confidence.Decide(assessment.Confidence, contradictions, evidence.HasEvidence || items.Count > 0);

        decisions.Add($"độ tự tin {assessment.Confidence:F2} ({assessment.Explanation})");
        decisions.Add($"kết luận: {verdict}");

        bool requiresDisclosure = items.Any(e => e.Source.RequiresDisclosure)
            && !items.Any(e => e.Trust == TrustLevel.Verified);

        // ---------- 5. Generate Answer ----------
        string text = _generator.Compose(question, verdict, assessment.Confidence,
            claims, cited, contradictions, requiresDisclosure);

        // ---------- 6. Validate ----------
        //
        // Kiểm trên tập căn cứ đã dẫn CỘNG các mẩu tham gia mâu thuẫn: câu
        // trả lời có nêu phía nói ngược lại, nên nội dung của phía đó cũng là
        // nội dung có căn cứ — dù nó không phải mẩu được chọn để trả lời.
        var validationScope = cited
            .Concat(contradictions.SelectMany(c => new[] { c.Left, c.Right }))
            .DistinctBy(e => e.Id)
            .ToList();

        var validation = _validator.Validate(text, claims, validationScope);

        var warnings = new List<string>(evidence.Warnings);

        if (!validation.IsValid)
        {
            decisions.Add($"kiểm tra cuối KHÔNG đạt: {string.Join("; ", validation.Problems)}");
            warnings.AddRange(validation.Problems);

            if (_options.EnforceValidation && verdict != AnswerVerdict.Conflicted)
            {
                // Không qua được kiểm tra -> HẠ xuống "không đủ căn cứ", và
                // viết lại câu trả lời. Không phát ra một văn bản mà chính hệ
                // thống vừa kết luận là không truy được nguồn.
                decisions.Add("hạ kết luận xuống Insufficient và viết lại câu trả lời");

                verdict = AnswerVerdict.Insufficient;
                text = _generator.Compose(question, verdict, assessment.Confidence,
                    claims, cited, contradictions, requiresDisclosure);
            }
        }

        return new GroundedAnswer
        {
            Question = question,
            Text = text,
            Verdict = verdict,
            Confidence = assessment.Confidence,
            Claims = claims,
            Contradictions = contradictions,
            CitedEvidence = cited,
            Warnings = warnings,
            RequiresDisclosure = requiresDisclosure,
            AnsweredAt = startedAt,
            Validation = validation,
            Summary = new ReasoningSummary
            {
                Decisions = decisions,
                EvidenceIds = [.. cited.Select(e => e.Id)],
                ChannelCount = items.Select(e => e.Kind).Distinct().Count(),
                ContradictionCount = contradictions.Count,
                FactCount = claims.Count(c => c.Kind == ClaimKind.Fact),
                InferenceCount = claims.Count(c => c.Kind == ClaimKind.Inference),
            },
        };
    }

    /// <summary>
    /// Thử bước suy luận ký hiệu của Phase 9.
    ///
    /// Kết quả thành evidence kênh <see cref="SourceKind.Reasoning"/> — tức là
    /// <see cref="TrustLevel.Derived"/>, nên nó KHÔNG được thành kiến thức lâu
    /// dài (luật Phase 13), và mọi khẳng định dựng từ nó bị gắn nhãn
    /// <see cref="ClaimKind.Inference"/> chứ không phải Fact.
    ///
    /// Câu hỏi khái niệm sẽ làm engine ném ngoại lệ — đó là chuyện bình thường,
    /// không phải lỗi, nên ở đây bắt và bỏ qua.
    /// </summary>
    private Evidence? TryReason(string question)
    {
        try
        {
            var answer = _reasoning!.Ask(question);

            if (string.IsNullOrWhiteSpace(answer.Text)) return null;

            // Ghi lại câu trả lời kèm câu hỏi để nó có đủ từ khoá mà đỡ được
            // chính nó khi validator kiểm.
            return Evidence.FromReasoning(question.Trim(), $"{question.Trim()} = {answer.Text}") with
            {
                Relevance = 1.0,
                Reliability = 0.9,
            };
        }
        catch (Exception ex) when (ex is ReasoningException or ParseException)
        {
            return null;
        }
    }

    private GroundedAnswer BuildEmpty(string question, EvidenceSet evidence,
        DateTimeOffset at, List<string> decisions) =>
        new()
        {
            Question = question,
            Text = _generator.Compose(question, AnswerVerdict.Insufficient, 0, [], [], [], false),
            Verdict = AnswerVerdict.Insufficient,
            Confidence = 0,
            Warnings = evidence.Warnings,
            AnsweredAt = at,
            Summary = new ReasoningSummary { Decisions = decisions },
        };
}
