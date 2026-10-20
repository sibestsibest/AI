using Ai.Providers.Contracts;

namespace Ai.Providers.Orchestration;

/// <summary>Kết quả kiểm một câu trả lời trước khi phát ra.</summary>
public sealed record ValidationOutcome(bool IsValid, IReadOnlyList<string> Problems)
{
    public static ValidationOutcome Pass { get; } = new(true, []);

    public static ValidationOutcome Fail(params string[] problems) => new(false, problems);

    public override string ToString() =>
        IsValid ? "ĐẠT" : $"KHÔNG ĐẠT — {string.Join("; ", Problems)}";
}

/// <summary>Một luật kiểm. Trả về null nếu đạt, hoặc lý do không đạt.</summary>
public delegate string? ResponseCheck(AiResponse response, IReadOnlyList<AiToolExchange> tools);

/// <summary>
/// KIỂM CÂU TRẢ LỜI TRƯỚC KHI PHÁT — bước "Result Verification" của đường ống.
///
/// Lớp này KHÔNG thay <c>AnswerValidator</c> của Phase 14. Hai lớp kiểm hai thứ
/// khác nhau, và gộp lại thì mất cả hai:
///
///   AnswerValidator (P14)   mọi câu có truy được về CĂN CỨ không
///   ResponseValidator (đây) câu trả lời có DÙNG ĐƯỢC không: rỗng, bị cắt giữa
///                           chừng, bỏ qua kết quả công cụ, thiếu nguồn khi
///                           đáng lẽ phải có
///
/// Đường đi khi không đạt là SỬA RỒI THỬ LẠI, tối đa 2 lần (đúng hạn mức đề
/// bài), rồi DỪNG. Không có nhánh nào lặp vô hạn — và quan trọng hơn: hết lượt
/// thử thì trả về câu trả lời TỐT NHẤT đã có kèm cảnh báo, chứ không trả về
/// rỗng. Người dùng nhận một câu trả lời chưa hoàn hảo vẫn hơn nhận một ô trống.
/// </summary>
public sealed class ResponseValidator
{
    private readonly List<ResponseCheck> _checks = [];

    /// <summary>Bộ luật mặc định — những thứ hỏng thấy được mà không cần biết gì về nghiệp vụ.</summary>
    public static ResponseValidator Default() => new ResponseValidator()
        .Add(static (response, _) => string.IsNullOrWhiteSpace(response.Content)
            ? "câu trả lời rỗng"
            : null)

        // Bị cắt vì hết token là lỗi HAY BỊ BỎ QUA NHẤT: câu trả lời trông bình
        // thường, chỉ cụt ở cuối, và không có gì báo. Người đọc tưởng model
        // nói xong rồi.
        .Add(static (response, _) => response.FinishReason == AiFinishReason.Length
            ? "câu trả lời bị cắt vì chạm trần token — tăng MaxTokens hoặc hỏi ngắn lại"
            : null)

        .Add(static (response, _) => response.FinishReason == AiFinishReason.ContentFilter
            ? "nhà cung cấp đã chặn nội dung này"
            : null)

        // Công cụ chạy xong mà câu trả lời không nhắc gì tới kết quả là dấu
        // hiệu model đã bỏ qua nó và trả lời bằng trí nhớ của chính nó — đúng
        // thứ mà cả kiến trúc công cụ được dựng ra để tránh.
        .Add(static (response, tools) =>
        {
            var used = tools.Where(t => t.Result.Succeeded).ToList();

            if (used.Count == 0) return null;

            bool mentionsAny = used.Any(t => Mentions(response.Content, t.Result.Output));

            return mentionsAny
                ? null
                : $"đã chạy {used.Count} công cụ nhưng câu trả lời không dùng kết quả nào";
        });

    public ResponseValidator Add(ResponseCheck check)
    {
        ArgumentNullException.ThrowIfNull(check);
        _checks.Add(check);

        return this;
    }

    public ValidationOutcome Validate(AiResponse response, IReadOnlyList<AiToolExchange>? tools = null)
    {
        ArgumentNullException.ThrowIfNull(response);

        var problems = _checks
            .Select(check => check(response, tools ?? []))
            .Where(problem => problem is not null)
            .Select(problem => problem!)
            .ToList();

        return problems.Count == 0 ? ValidationOutcome.Pass : new ValidationOutcome(false, problems);
    }

    /// <summary>
    /// Câu trả lời có nhắc tới kết quả công cụ không.
    ///
    /// Cố ý THÔ: lấy vài mẩu dài của kết quả và xem chúng có xuất hiện không.
    /// So khớp tinh vi hơn (ngữ nghĩa, nhúng vector) sẽ đắt hơn và vẫn sai;
    /// phép kiểm này chỉ cần bắt được trường hợp model LỜ HẲN kết quả, và với
    /// việc đó thì so chuỗi là đủ.
    /// </summary>
    private static bool Mentions(string answer, string toolOutput)
    {
        var tokens = toolOutput
            .Split([' ', '\n', '\r', '\t', ',', ';', '(', ')', '"'], StringSplitOptions.RemoveEmptyEntries)
            .Where(t => t.Length >= 4)
            .Take(40)
            .ToList();

        if (tokens.Count == 0) return true;

        int hits = tokens.Count(t => answer.Contains(t, StringComparison.OrdinalIgnoreCase));

        return hits > 0;
    }
}
