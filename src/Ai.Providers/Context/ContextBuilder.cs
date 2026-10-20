using System.Text;
using Ai.Phase16.Grounding;
using Ai.Phase16.Memory;
using Ai.Providers.Contracts;

namespace Ai.Providers.Context;

/// <summary>
/// NGỮ CẢNH đã chọn lọc cho MỘT lượt hỏi — và thứ đã bị BỎ RA.
///
/// <see cref="Omitted"/> không phải để trang trí. Khi model trả lời thiếu, câu
/// hỏi đầu tiên luôn là "nó có được đọc thứ đó không" — và nếu không ghi lại
/// những gì đã bị cắt thì không ai trả lời được câu đó. Nó cũng là cách duy
/// nhất để thấy hạn mức đang cắt nhầm.
/// </summary>
public sealed record BuiltContext
{
    public required string SystemPrompt { get; init; }

    public IReadOnlyList<ChatMessage> History { get; init; } = [];

    public required string UserPrompt { get; init; }

    /// <summary>Mẩu nhớ đã được chọn, kèm điểm — để giải trình được vì sao chọn nó.</summary>
    public IReadOnlyList<string> MemoryUsed { get; init; } = [];

    public IReadOnlyList<string> Omitted { get; init; } = [];

    public int CharacterCount =>
        SystemPrompt.Length + UserPrompt.Length + History.Sum(m => m.Content.Length);

    public override string ToString() =>
        $"{CharacterCount} ký tự, {History.Count} lượt cũ, {MemoryUsed.Count} mẩu nhớ" +
        (Omitted.Count > 0 ? $", bỏ {Omitted.Count} thứ" : "");
}

public sealed record ContextBuilderOptions
{
    /// <summary>Số lượt hội thoại gần nhất được đưa vào. Mặc định nhỏ, có chủ ý.</summary>
    public int MaxHistoryTurns { get; init; } = 6;

    /// <summary>Số mẩu nhớ dài hạn được đưa vào.</summary>
    public int MaxMemoryFacts { get; init; } = 5;

    /// <summary>Mẩu nhớ dưới điểm này coi như không liên quan.</summary>
    public double MinimumMemoryScore { get; init; } = 0.15;

    /// <summary>
    /// Hạn ký tự của TOÀN BỘ ngữ cảnh, kiểm TRƯỚC khi gọi mạng.
    ///
    /// Nhỏ hơn <c>AiProvidersOptions.MaxPromptCharacters</c> một khoảng, vì bộ
    /// định tuyến còn kiểm lần nữa ở ngoài — và một yêu cầu bị chặn ở đó thì đã
    /// đi qua cả đường ống rồi mới bị bỏ.
    /// </summary>
    public int MaxCharacters { get; init; } = 10_000;

    /// <summary>Kết quả công cụ dài hơn mức này bị cắt trước khi vào ngữ cảnh.</summary>
    public int MaxToolResultCharacters { get; init; } = 2_000;
}

public interface IContextBuilder
{
    BuiltContext Build(string question, string systemInstructions,
        IReadOnlyList<AiToolResult>? toolResults = null);
}

/// <summary>
/// BỘ DỰNG NGỮ CẢNH — chọn cái LIÊN QUAN, không gửi cả kho.
///
/// Đây là lớp quyết định điều mà đề bài nói ngắn gọn là "đừng gửi cả cơ sở dữ
/// liệu cho OpenAI". Nó ghép bốn nguồn, và mỗi nguồn có một luật riêng:
///
///   CHỈ THỊ HỆ THỐNG   đưa vào nguyên văn. Đây là thứ DUY NHẤT trong ngữ cảnh
///                      do ứng dụng viết, nên cũng là thứ duy nhất được tin.
///
///   NHỚ DÀI HẠN        tra theo CÂU HỎI, lấy vài mẩu điểm cao nhất. Không
///                      phải "mọi thứ người dùng từng nói" — kho nhớ lớn dần
///                      mãi, còn hạn mức thì không.
///
///   LỊCH SỬ HỘI THOẠI  vài lượt gần nhất. Lượt cũ hơn đã nằm trong nhớ dài
///                      hạn nếu nó đáng nhớ; nếu không đáng thì gửi đi cũng
///                      chỉ làm loãng.
///
///   KẾT QUẢ CÔNG CỤ    chỉ của lượt ĐANG chạy, và bị cắt độ dài. Một lượt tra
///                      web trả về hàng chục nghìn ký tự là chuyện bình thường.
///
/// NỘI DUNG KHÔNG TIN CẬY KHÔNG ĐƯỢC VÀO CHỈ THỊ HỆ THỐNG. Nhớ, lịch sử và kết
/// quả công cụ đều là thứ người dùng hoặc Internet viết ra, nên chúng đi vào
/// phần NGƯỜI DÙNG của ngữ cảnh, có nhãn rõ ràng. Trộn chúng vào chỉ thị hệ
/// thống là mở đúng cánh cửa mà <c>UntrustedText</c> của Phase 12 được viết ra
/// để đóng lại.
/// </summary>
public sealed class ContextBuilder : IContextBuilder
{
    private readonly MemoryStore? _memory;
    private readonly ConversationContext? _conversation;
    private readonly ContextBuilderOptions _options;

    public ContextBuilder(
        MemoryStore? memory = null,
        ConversationContext? conversation = null,
        ContextBuilderOptions? options = null)
    {
        _memory = memory;
        _conversation = conversation;
        _options = options ?? new ContextBuilderOptions();
    }

    public BuiltContext Build(string question, string systemInstructions,
        IReadOnlyList<AiToolResult>? toolResults = null)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(systemInstructions);

        var omitted = new List<string>();
        var memoryUsed = new List<string>();

        // --- Nhớ dài hạn: tra theo câu hỏi ---
        var facts = new StringBuilder();

        if (_memory is not null)
        {
            var hits = _memory.Search(question, _options.MaxMemoryFacts, _options.MinimumMemoryScore);

            foreach (var hit in hits)
            {
                facts.Append("  • ").AppendLine(hit.Fact.Content);
                memoryUsed.Add($"{hit.Fact.Content} (điểm {hit.Score:F3})");
            }

            int total = _memory.LongTerm.Count;

            if (total > hits.Count)
            {
                omitted.Add($"{total - hits.Count}/{total} mẩu nhớ không liên quan đến câu hỏi này");
            }
        }

        // --- Lịch sử hội thoại: vài lượt gần nhất ---
        var history = new List<ChatMessage>();

        if (_conversation is not null)
        {
            var recent = _conversation.Recent(_options.MaxHistoryTurns);

            foreach (var turn in recent)
            {
                history.Add(new ChatMessage(ChatRole.User, turn.UserInput));

                if (!string.IsNullOrWhiteSpace(turn.AgentAnswer))
                {
                    history.Add(new ChatMessage(ChatRole.Assistant, turn.AgentAnswer));
                }
            }

            if (_conversation.Count > recent.Count)
            {
                omitted.Add($"{_conversation.Count - recent.Count} lượt hội thoại cũ hơn");
            }
        }

        // --- Câu hỏi + những gì đã biết ---
        var prompt = new StringBuilder();

        if (facts.Length > 0)
        {
            prompt.AppendLine("Những điều đã biết về người dùng (do chính họ nói, chưa kiểm chứng độc lập):")
                .Append(facts)
                .AppendLine();
        }

        if (toolResults is { Count: > 0 })
        {
            prompt.AppendLine("Kết quả công cụ cho lượt này:");

            foreach (var result in toolResults)
            {
                string output = result.Output.Length > _options.MaxToolResultCharacters
                    ? result.Output[.._options.MaxToolResultCharacters] + "… (đã cắt)"
                    : result.Output;

                if (result.Output.Length > _options.MaxToolResultCharacters)
                {
                    omitted.Add($"kết quả công cụ '{result.Name}' bị cắt còn " +
                                $"{_options.MaxToolResultCharacters}/{result.Output.Length} ký tự");
                }

                prompt.Append("  [").Append(result.Name).Append("] ").AppendLine(output);
            }

            prompt.AppendLine();
        }

        prompt.Append("Câu hỏi: ").Append(question.Trim());

        var context = new BuiltContext
        {
            SystemPrompt = systemInstructions,
            History = history,
            UserPrompt = prompt.ToString(),
            MemoryUsed = memoryUsed,
            Omitted = omitted,
        };

        return Trim(context, omitted);
    }

    /// <summary>
    /// Quá hạn thì BỎ LỊCH SỬ TRƯỚC, không bỏ câu hỏi và không bỏ nhớ.
    ///
    /// Thứ tự hy sinh có lý do: câu hỏi là thứ đang được hỏi, nhớ là thứ được
    /// chọn VÌ liên quan tới nó, còn lịch sử được đưa vào chỉ vì nó gần đây.
    /// Cắt từ cuối danh sách lên nên lượt gần nhất sống lâu nhất.
    /// </summary>
    private BuiltContext Trim(BuiltContext context, List<string> omitted)
    {
        if (context.CharacterCount <= _options.MaxCharacters) return context;

        var history = context.History.ToList();

        while (history.Count > 0 && Size(context, history) > _options.MaxCharacters)
        {
            history.RemoveAt(0);
        }

        int dropped = context.History.Count - history.Count;

        if (dropped > 0)
        {
            omitted.Add($"bỏ thêm {dropped} lượt hội thoại để vừa hạn {_options.MaxCharacters} ký tự");
        }

        return context with { History = history, Omitted = omitted };
    }

    private static int Size(BuiltContext context, List<ChatMessage> history) =>
        context.SystemPrompt.Length + context.UserPrompt.Length + history.Sum(m => m.Content.Length);
}
