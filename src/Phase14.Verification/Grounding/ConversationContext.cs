using System.Text;
using Ai.Phase14.Agent;
using Ai.Phase14.Memory;

namespace Ai.Phase14.Grounding;

/// <summary>Một lượt hội thoại: người dùng nói gì, agent trả lời gì.</summary>
public sealed record ConversationTurn
{
    public required int Index { get; init; }

    public required string UserInput { get; init; }

    public string? AgentAnswer { get; init; }

    public Intent? Intent { get; init; }

    public required DateTimeOffset At { get; init; }

    /// <summary>Lượt này có dùng dữ liệu Internet chưa kiểm chứng hay không.</summary>
    public bool UsedUnverifiedData { get; init; }

    public override string ToString() => $"#{Index} \"{UserInput}\"";
}

/// <summary>
/// NGỮ CẢNH HỘI THOẠI — phần "Context" của Phase 13.
///
/// Đây là mảnh thứ ba, và nó khác hẳn hai mảnh kia:
///
/// | | Ghi vào bằng | Tồn tại | Phai đi |
/// |---|---|---|---|
/// | Knowledge | người biên tập | đến khi bị sửa | không |
/// | Memory    | `Store()`      | đến khi bị quên | có, theo giá trị |
/// | Context   | mỗi lượt nói   | **hết phiên là mất** | không, bị cắt cứng |
///
/// Context bị CẮT CỨNG theo số lượt (cửa sổ trượt), không phai dần theo giá
/// trị như memory. Đó chính là hành vi của context window trong một LLM thật,
/// và cũng chính là giới hạn của nó: lượt thứ 21 đẩy lượt thứ 1 ra ngoài, bất
/// kể lượt thứ 1 có quan trọng đến đâu. Muốn giữ thì phải chuyển sang memory —
/// đó là lý do memory tồn tại.
///
/// Trong Phase 13, context có hai việc:
///
///   1. Làm nguồn evidence (người dùng vừa nói gì)
///   2. Mở rộng truy vấn — "còn cái kia thì sao?" chỉ hiểu được nếu biết lượt
///      trước đang nói về cái gì
/// </summary>
public sealed class ConversationContext
{
    private readonly List<ConversationTurn> _turns = [];
    private readonly IClock _clock;
    private int _nextIndex = 1;

    public ConversationContext(IClock? clock = null, int maxTurns = 20)
    {
        if (maxTurns <= 0) throw new ArgumentOutOfRangeException(nameof(maxTurns));

        _clock = clock ?? new SystemClock();
        MaxTurns = maxTurns;
    }

    /// <summary>Cửa sổ trượt: giữ tối đa bao nhiêu lượt gần nhất.</summary>
    public int MaxTurns { get; }

    public IReadOnlyList<ConversationTurn> Turns => _turns;

    public int Count => _turns.Count;

    /// <summary>Tổng số lượt đã đi qua, kể cả những lượt đã bị đẩy ra khỏi cửa sổ.</summary>
    public int TotalTurns => _nextIndex - 1;

    public ConversationTurn? Last => _turns.Count > 0 ? _turns[^1] : null;

    public ConversationTurn Record(string userInput, string? answer = null, Intent? intent = null,
        bool usedUnverifiedData = false)
    {
        ArgumentNullException.ThrowIfNull(userInput);

        var turn = new ConversationTurn
        {
            Index = _nextIndex++,
            UserInput = userInput.Trim(),
            AgentAnswer = answer,
            Intent = intent,
            At = _clock.Now,
            UsedUnverifiedData = usedUnverifiedData,
        };

        _turns.Add(turn);

        // Cắt cứng, không thương lượng — đúng như context window thật.
        while (_turns.Count > MaxTurns) _turns.RemoveAt(0);

        return turn;
    }

    /// <summary>Các lượt gần nhất, mới nhất trước.</summary>
    public IReadOnlyList<ConversationTurn> Recent(int take = 3) =>
        [.. _turns.AsEnumerable().Reverse().Take(take)];

    /// <summary>
    /// Từ khoá của mấy lượt gần nhất — để mở rộng một câu hỏi thiếu chủ ngữ.
    ///
    /// Chỉ lấy từ NGƯỜI DÙNG nói, không lấy từ câu agent trả lời. Nếu lấy cả
    /// câu trả lời thì ngữ cảnh sẽ bị chính agent làm loãng: nó trả lời dài,
    /// nên từ khoá của nó áp đảo từ khoá thật của người dùng, và các lượt sau
    /// bị kéo lệch theo những gì agent đã nói.
    /// </summary>
    public HashSet<string> RecentKeywords(int take = 3)
    {
        var words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var turn in Recent(take))
        {
            words.UnionWith(Keywords.Extract(turn.UserInput));
        }

        return words;
    }

    /// <summary>
    /// Từ CHỈ TRỎ LẠI (anaphora) — chúng thay cho một chủ đề đã nói ở lượt
    /// trước, chứ tự chúng không mang chủ đề nào.
    ///
    /// Đây là danh sách quyết định khi nào cần bổ sung ngữ cảnh.
    /// </summary>
    private static readonly HashSet<string> Anaphora = new(StringComparer.OrdinalIgnoreCase)
    {
        // tiếng Việt
        "còn", "nó", "đó", "kia", "ấy", "này", "thế", "sao", "nữa", "thêm", "luôn",
        // tiếng Anh
        "it", "that", "this", "those", "these", "them", "more", "else", "again", "too",
    };

    /// <summary>
    /// Bổ sung ngữ cảnh vào một câu hỏi không tự đứng được.
    ///
    /// ĐIỀU KIỆN CAN THIỆP — và chỗ này rất dễ làm sai. Cách làm hiển nhiên là
    /// đếm số từ khoá: ít từ thì bổ sung. Nhưng đếm từ là thước đo SAI:
    ///
    ///     "còn nó"            2 từ khoá, mà KHÔNG có chủ đề nào  -> cần bổ sung
    ///     "overfitting là gì" 1 từ khoá, chủ đề rất rõ            -> để nguyên
    ///
    /// Đếm từ cho ra kết luận ngược hẳn ở cả hai câu. Thước đo đúng không phải
    /// SỐ LƯỢNG từ mà là có từ nào MANG CHỦ ĐỀ hay không: câu chỉ gồm từ trỏ
    /// lại ("còn nó", "thế còn cái đó") thì mới phải đi mượn chủ đề của lượt
    /// trước.
    ///
    /// Câu đã có chủ đề riêng thì để NGUYÊN. Nhồi ngữ cảnh vào một câu đã rõ
    /// ràng không chỉ gây nhiễu — nó lôi chủ đề cũ vào câu hỏi mới, khiến agent
    /// trả lời chuyện người dùng đã hỏi xong từ ba lượt trước.
    /// </summary>
    public string Expand(string question, int take = 2)
    {
        ArgumentNullException.ThrowIfNull(question);

        if (_turns.Count == 0) return question;

        var own = Keywords.Extract(question);

        // Có ít nhất một từ mang chủ đề -> câu tự đứng được.
        if (own.Any(w => !Anaphora.Contains(w))) return question;

        var added = RecentKeywords(take).Where(w => !own.Contains(w)).ToList();
        if (added.Count == 0) return question;

        return $"{question.Trim()} {string.Join(' ', added)}";
    }

    /// <summary>Bản ghi hội thoại để in ra.</summary>
    public string Transcript()
    {
        var builder = new StringBuilder();

        foreach (var turn in _turns)
        {
            builder.Append('#').Append(turn.Index).Append(" NGƯỜI DÙNG: ").AppendLine(turn.UserInput);

            if (turn.AgentAnswer is not null)
            {
                builder.Append("   AGENT     : ")
                    .AppendLine(turn.AgentAnswer.Length > 100
                        ? turn.AgentAnswer[..100] + "…"
                        : turn.AgentAnswer);
            }
        }

        return builder.ToString().TrimEnd();
    }

    public void Clear()
    {
        _turns.Clear();
        _nextIndex = 1;
    }
}
