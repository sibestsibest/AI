using System.Text.RegularExpressions;

namespace Ai.Providers.Security;

/// <summary>
/// LỌC DỮ LIỆU BÍ MẬT — chạy ở HAI chiều, và cả hai đều cần.
///
///   ra ngoài : trước khi prompt được gửi cho một provider bên thứ ba
///   vào log  : trước khi bất kỳ thông điệp lỗi nào được ghi lại
///
/// CHIỀU RA NGOÀI LÀ CHIỀU DỄ BỊ BỎ QUÊN, và cũng là chiều tệ hơn khi sai.
/// Prompt của hệ thống này được dựng từ căn cứ — mà căn cứ gồm cả ký ức người
/// dùng (Phase 10) và nội dung web (Phase 12). Nếu người dùng từng bảo agent
/// "hãy nhớ token của tôi là …", thì token đó là một mẩu ký ức hợp lệ, và nó sẽ
/// lặng lẽ đi vào prompt rồi ra khỏi máy. Không có cách nào lấy lại.
///
/// Bộ lọc này KHÔNG phải chốt an toàn tuyệt đối — lọc theo mẫu không bao giờ
/// bắt hết được. Nó là rào giảm tốc cho những dạng bí mật có hình dạng rõ ràng.
/// Chốt thật là nguyên tắc: đừng đưa dữ liệu không cần thiết vào prompt. Xem
/// <c>EvidencePrompt</c> — nó chỉ gửi những mẩu căn cứ ĐÃ ĐƯỢC DẪN trong câu
/// trả lời, không gửi cả kho.
/// </summary>
public static class OutboundRedactor
{
    private const string Mask = "[ĐÃ-LỌC]";

    /// <summary>
    /// Các dạng bí mật có hình dạng nhận ra được.
    ///
    /// Thứ tự có ý nghĩa: mẫu cụ thể (tiền tố khoá của từng nhà cung cấp) chạy
    /// trước mẫu chung (chuỗi dài ngẫu nhiên), để thông điệp cắt bỏ đúng phần
    /// cần cắt thay vì cắt cả câu.
    /// </summary>
    private static readonly Regex[] Patterns =
    [
        // khoá API theo tiền tố quen gặp
        new(@"\bsk-[A-Za-z0-9_\-]{16,}", RegexOptions.CultureInvariant),
        new(@"\bgsk_[A-Za-z0-9_\-]{16,}", RegexOptions.CultureInvariant),
        new(@"\bhf_[A-Za-z0-9]{16,}", RegexOptions.CultureInvariant),
        new(@"\bAIza[0-9A-Za-z_\-]{20,}", RegexOptions.CultureInvariant),
        new(@"\bghp_[A-Za-z0-9]{20,}", RegexOptions.CultureInvariant),
        new(@"\bxox[baprs]-[A-Za-z0-9\-]{10,}", RegexOptions.CultureInvariant),

        // header uỷ quyền
        new(@"(?i)\bauthorization\s*[:=]\s*\S+", RegexOptions.CultureInvariant),
        new(@"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}", RegexOptions.CultureInvariant),

        // JWT
        new(@"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}", RegexOptions.CultureInvariant),

        // khoá riêng dạng PEM
        new(@"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            RegexOptions.CultureInvariant),

        // chuỗi kết nối
        new(@"(?i)\b(password|pwd|pass)\s*[:=]\s*[^\s;,""']+", RegexOptions.CultureInvariant),
        new(@"(?i)\b(api[_\-]?key|apikey|secret|token)\s*[:=]\s*[^\s;,""']{8,}", RegexOptions.CultureInvariant),
        new(@"(?i)\b(Server|Data Source|Initial Catalog|User ID)\s*=\s*[^;]+;", RegexOptions.CultureInvariant),

        // khoá trong query string — dạng hay làm lộ khoá qua log và qua URL
        new(@"(?i)([?&](key|api_key|apikey|access_token|token)=)[^&\s]+", RegexOptions.CultureInvariant),
    ];

    /// <summary>Thay mọi đoạn trông như bí mật bằng một mốc nhìn thấy được.</summary>
    public static string Redact(string? text)
    {
        if (string.IsNullOrEmpty(text)) return text ?? "";

        string result = text;

        foreach (var pattern in Patterns)
        {
            result = pattern.Replace(result, Mask);
        }

        return result;
    }

    /// <summary>Có phát hiện dạng bí mật nào trong văn bản hay không.</summary>
    public static bool ContainsSecret(string? text) =>
        !string.IsNullOrEmpty(text) && Patterns.Any(p => p.IsMatch(text));

    /// <summary>
    /// Đếm số đoạn bị lọc — để ghi log rằng "đã lọc 2 chỗ" mà không ghi chúng
    /// là gì. Con số đó đủ để biết prompt đang chứa thứ không nên chứa.
    /// </summary>
    public static int CountSecrets(string? text) =>
        string.IsNullOrEmpty(text) ? 0 : Patterns.Sum(p => p.Matches(text).Count);
}
