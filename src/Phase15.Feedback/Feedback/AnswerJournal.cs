using Ai.Phase15.Agent;
using Ai.Phase15.Memory;
using Ai.Phase15.Verification;

namespace Ai.Phase15.Feedback;

/// <summary>
/// SỔ CÂU TRẢ LỜI — nơi duy nhất cấp id cho một câu trả lời đã phát ra.
///
/// VÌ SAO CẦN MỘT SỔ RIÊNG, KHI <see cref="GroundedAnswer"/> ĐÃ CÓ ĐỦ MỌI THỨ?
///
/// Vì <see cref="GroundedAnswer"/> là một giá trị — nó được sinh ra rồi trả về
/// cho người gọi, và Phase 14 không giữ lại bản nào. Phản hồi thì đến SAU, có
/// thể là vài phút sau, và nó chỉ mang theo một cái id. Không có sổ thì id đó
/// không tra được về đâu, và mọi phân tích lỗi phải tin vào những gì người dùng
/// kể lại — tức là tin vào chính chỗ đang cần kiểm chứng.
///
/// SỔ CÓ DUNG LƯỢNG CỨNG, và đó là lựa chọn có chủ ý (giống
/// <see cref="ShortTermMemory"/> của Phase 10): phản hồi hầu như luôn đến về
/// những câu trả lời gần nhất, nên giữ vô hạn chỉ đổi lấy rò rỉ bộ nhớ. Câu
/// trả lời quá cũ bị đẩy ra, và phản hồi về nó bị từ chối THẲNG kèm lý do —
/// chứ không âm thầm được nhận rồi phân tích trên dữ liệu rỗng.
/// </summary>
public sealed class AnswerJournal
{
    private readonly List<AnswerRecord> _records = [];
    private readonly IClock _clock;
    private int _nextId = 1;

    public AnswerJournal(IClock? clock = null, int capacity = 200)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity), "Dung lượng phải dương.");

        _clock = clock ?? new SystemClock();
        Capacity = capacity;
    }

    public int Capacity { get; }

    public int Count => _records.Count;

    /// <summary>Số câu trả lời đã bị đẩy ra khỏi sổ vì quá cũ.</summary>
    public int EvictedCount { get; private set; }

    public IReadOnlyList<AnswerRecord> Records => _records;

    /// <summary>Ghi một câu trả lời đã qua đường ống kiểm chứng của Phase 14.</summary>
    public AnswerRecord Record(GroundedAnswer answer, Intent? predictedIntent = null, double intentConfidence = 0)
    {
        ArgumentNullException.ThrowIfNull(answer);

        return Add(answer.Question, answer.Text, answer, predictedIntent, intentConfidence);
    }

    /// <summary>
    /// Ghi một lượt trả lời của agent (<see cref="AiController.HandleAsync"/>).
    ///
    /// Đây là điểm nối Phase 15 với agent: phản hồi phải gắn được vào MỌI lượt
    /// trả lời, kể cả lượt đi đường công cụ — vì người dùng chấm cả những lượt
    /// đó, và ý định bị hiểu sai hay xảy ra nhất chính ở đó.
    /// </summary>
    public AnswerRecord Record(string question, AgentResponse response)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(response);

        return Add(question, response.Answer, response.Grounded, response.Intent, response.Confidence);
    }

    private AnswerRecord Add(string question, string answerText, GroundedAnswer? grounded,
        Intent? predictedIntent, double intentConfidence)
    {
        var record = new AnswerRecord
        {
            Id = $"a{_nextId++}",
            Question = question,
            AnswerText = answerText,
            Grounded = grounded,
            PredictedIntent = predictedIntent,
            IntentConfidence = intentConfidence,
            RecordedAt = _clock.Now,
        };

        _records.Add(record);

        while (_records.Count > Capacity)
        {
            _records.RemoveAt(0);
            EvictedCount++;
        }

        return record;
    }

    public AnswerRecord? Find(string answerId)
    {
        ArgumentNullException.ThrowIfNull(answerId);

        return _records.FirstOrDefault(r => r.Id == answerId);
    }

    public bool Contains(string answerId) => Find(answerId) is not null;

    public void Clear()
    {
        _records.Clear();
        EvictedCount = 0;
    }
}
