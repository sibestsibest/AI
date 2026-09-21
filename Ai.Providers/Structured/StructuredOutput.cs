using System.Text.Json;
using Ai.Providers.Contracts;
using Ai.Providers.Routing;

namespace Ai.Providers.Structured;

/// <summary>Kết quả đọc một câu trả lời có cấu trúc: hoặc ra dữ liệu, hoặc ra lý do không ra.</summary>
public sealed record StructuredResult<T>(bool Success, T? Value, string? Problem)
{
    public static StructuredResult<T> Ok(T value) => new(true, value, null);

    public static StructuredResult<T> Fail(string problem) => new(false, default, problem);

    public override string ToString() => Success ? $"OK: {Value}" : $"HỎNG: {Problem}";
}

/// <summary>
/// ĐỌC CÂU TRẢ LỜI CÓ CẤU TRÚC — và đây là chốt chặn thật, không phải lược đồ.
///
/// ============================================================
/// LUẬT: KẾT QUẢ TỰ DO CỦA MODEL KHÔNG ĐƯỢC ĐIỀU KHIỂN NGHIỆP VỤ
/// ============================================================
///
/// Mọi thứ đi qua đây đều được coi là KHÔNG TIN CẬY, kể cả khi provider nói nó
/// đã bảo đảm lược đồ. Lý do rất cụ thể: "bảo đảm lược đồ" chỉ bảo đảm HÌNH
/// DẠNG, không bảo đảm Ý NGHĨA. Model vẫn trả về được
/// <c>{"quantity": -5, "unit": "kg"}</c> — khớp lược đồ hoàn hảo, và vô nghĩa.
///
/// Nên chỗ này làm đúng hai việc, theo thứ tự:
///
///   1. ĐỌC   — JSON hỏng thì dừng ở đây, không đi tiếp.
///   2. KIỂM  — người gọi đưa vào một hàm kiểm ý nghĩa. Không qua thì coi như
///              model không trả lời được, và bước sau KHÔNG chạy.
///
/// Việc TÍNH TOÁN sau đó là của dịch vụ tất định (máy tính, bộ tính dinh dưỡng,
/// truy vấn dữ liệu), không phải của model. Model chỉ RÚT ra cấu trúc từ câu
/// chữ — đó là việc nó giỏi và việc mà một bộ phân tích viết tay làm rất tệ.
/// </summary>
public static class StructuredOutput
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true,
        AllowTrailingCommas = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
    };

    /// <summary>
    /// Đọc nội dung câu trả lời thành <typeparamref name="T"/>.
    ///
    /// <paramref name="validate"/> trả về null nếu hợp lệ, hoặc lý do từ chối.
    /// Không truyền thì chỉ kiểm được hình dạng — dùng cho dữ liệu mà mọi giá
    /// trị đọc được đều chấp nhận được, và những trường hợp đó hiếm hơn ta tưởng.
    /// </summary>
    public static StructuredResult<T> Read<T>(string? content, Func<T, string?>? validate = null)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return StructuredResult<T>.Fail("câu trả lời rỗng");
        }

        string json = Unwrap(content);

        T? value;

        try
        {
            value = JsonSerializer.Deserialize<T>(json, Options);
        }
        catch (JsonException ex)
        {
            return StructuredResult<T>.Fail($"không đọc được JSON: {ex.Message}");
        }

        if (value is null)
        {
            return StructuredResult<T>.Fail("JSON đọc được nhưng ra null");
        }

        if (validate?.Invoke(value) is { } problem)
        {
            return StructuredResult<T>.Fail($"dữ liệu không hợp lệ: {problem}");
        }

        return StructuredResult<T>.Ok(value);
    }

    /// <summary>
    /// Hỏi model một lượt và đọc kết quả có cấu trúc.
    ///
    /// Thử tối đa <paramref name="maxAttempts"/> lần, và lần sau NÓI RÕ lần
    /// trước sai ở đâu. Đây là khác biệt giữa "thử lại" có ích và thử lại vô
    /// ích: gửi y nguyên yêu cầu cũ thì phần lớn thời gian nhận lại y nguyên
    /// câu trả lời cũ. Mặc định 2 lần, đúng hạn mức của đề bài.
    /// </summary>
    public static async Task<StructuredResult<T>> ExtractAsync<T>(
        IAiRouter router,
        AiRequest request,
        StructuredFormat format,
        Func<T, string?>? validate = null,
        int maxAttempts = 2,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(router);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(format);

        string? lastProblem = null;

        for (int attempt = 1; attempt <= Math.Max(1, maxAttempts); attempt++)
        {
            var attemptRequest = request with
            {
                ResponseFormat = format,
                TaskType = AiTaskType.ToolCalling,
                UserPrompt = lastProblem is null
                    ? request.UserPrompt
                    : $"{request.UserPrompt}\n\nLần trước bạn trả về không dùng được ({lastProblem}). " +
                      "Trả về ĐÚNG một object JSON khớp lược đồ, không kèm giải thích.",
            };

            var routed = await router.GenerateAsync(attemptRequest, cancellationToken).ConfigureAwait(false);

            if (!routed.Response.Success)
            {
                // Provider hỏng là chuyện khác hẳn model trả lời sai khuôn:
                // thử lại ở đây không sửa được gì, vì bộ định tuyến đã thử mọi
                // provider rồi mới trả về thất bại.
                return StructuredResult<T>.Fail($"không gọi được model: {routed.Response.Error}");
            }

            var result = Read(routed.Response.Content, validate);

            if (result.Success) return result;

            lastProblem = result.Problem;
        }

        return StructuredResult<T>.Fail($"thử {maxAttempts} lần vẫn không ra dữ liệu hợp lệ — {lastProblem}");
    }

    /// <summary>
    /// Gỡ khối mã bao quanh JSON.
    ///
    /// Model hay bọc JSON trong ```json … ``` dù prompt đã cấm. Dọn ở đây thay
    /// vì hy vọng prompt đủ nghiêm — cùng lý lẽ với <c>LlmAnswerGenerator</c>:
    /// prompt là lời đề nghị, code mới là luật.
    /// </summary>
    private static string Unwrap(string content)
    {
        string text = content.Trim();

        if (text.StartsWith("```", StringComparison.Ordinal))
        {
            int firstBreak = text.IndexOf('\n');
            if (firstBreak > 0) text = text[(firstBreak + 1)..];

            text = text.TrimEnd('`', '\n', '\r', ' ');
        }

        // Model đôi khi thêm một câu dẫn trước JSON. Lấy từ dấu mở đầu tiên tới
        // dấu đóng cuối cùng là cách vớt lại được phần lớn những lần như vậy.
        int start = text.IndexOfAny(['{', '[']);
        int end = text.LastIndexOfAny(['}', ']']);

        return start >= 0 && end > start ? text[start..(end + 1)] : text.Trim();
    }
}
