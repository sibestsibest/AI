using System.Net;
using System.Text;

namespace Ai.Phase12.Internet;

/// <summary>Bóc văn bản đọc được ra khỏi một trang đã tải về.</summary>
public interface IContentExtractor
{
    /// <summary>Kiểu nội dung này có bóc được không.</summary>
    bool CanExtract(string? contentType);

    /// <summary>Bóc thành <see cref="WebDocument"/>. Trả về null nếu không còn gì đáng giữ.</summary>
    WebDocument? Extract(FetchResult fetched, DateTimeOffset retrievedAt);
}

public sealed record ContentExtractorOptions
{
    /// <summary>Cắt văn bản ở đây. Trang tin tức dài 200 KB chữ, agent chỉ cần phần đầu.</summary>
    public int MaxTextLength { get; init; } = 8_000;

    /// <summary>Ngắn hơn ngưỡng này thì coi như trang rỗng (menu, trang lỗi, trang chặn bot).</summary>
    public int MinTextLength { get; init; } = 40;
}

/// <summary>
/// Bóc nội dung HTML bằng máy trạng thái tự viết — không dùng thư viện ngoài.
///
/// VÌ SAO KHÔNG DÙNG REGEX ĐỂ BỎ THẺ?
///
/// Vì `<[^>]*>` sai ở đúng những chỗ quan trọng nhất:
///
///     <script>if (a < b) { x = "</div>" }</script>
///
/// Regex sẽ cắt ở dấu `<` trong `a < b`, rồi để lại nguyên mã JavaScript
/// trong phần "văn bản" — và mã đó chính là thứ hay chứa văn bản do kẻ tấn
/// công đặt vào. Nên phải đi qua chuỗi một lượt bằng máy trạng thái, và xử
/// lý riêng ba vùng thô: script, style, comment.
///
/// THỨ TỰ CÁC BƯỚC (thứ tự này là bản chất, không phải tuỳ ý):
///
///   1. Lấy &lt;title&gt;                  — trước khi xoá thẻ
///   2. Bỏ hẳn nội dung script/style/comment/noscript/svg
///   3. Thẻ khối -> ký tự xuống dòng      — giữ lại ranh giới câu
///   4. Bỏ các thẻ còn lại
///   5. Giải mã thực thể HTML             — SAU khi bỏ thẻ, không được trước.
///                                           Giải mã trước thì "&lt;script&gt;"
///                                           biến thành thẻ script thật.
///   6. Gom khoảng trắng, cắt độ dài
/// </summary>
public sealed class HtmlContentExtractor(ContentExtractorOptions? options = null) : IContentExtractor
{
    private static readonly HashSet<string> DroppedElements = new(StringComparer.OrdinalIgnoreCase)
    {
        "script", "style", "noscript", "svg", "canvas", "template", "iframe", "head",
    };

    private static readonly HashSet<string> BlockElements = new(StringComparer.OrdinalIgnoreCase)
    {
        "p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "header", "footer", "nav", "aside", "blockquote", "pre",
        "table", "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "main", "hr",
    };

    private readonly ContentExtractorOptions _options = options ?? new ContentExtractorOptions();

    public ContentExtractorOptions Options => _options;

    public bool CanExtract(string? contentType) => contentType is null || contentType switch
    {
        "text/html" or "application/xhtml+xml" or "text/plain" or "text/markdown" or "application/json" => true,
        _ => false,
    };

    public WebDocument? Extract(FetchResult fetched, DateTimeOffset retrievedAt)
    {
        ArgumentNullException.ThrowIfNull(fetched);

        if (!fetched.Success || fetched.Body.IsEmpty) return null;

        // Dùng bản THÔ để bóc: làm cùn trước khi phân tích HTML sẽ phá vỡ cấu
        // trúc thẻ. Kết quả bóc xong mới bọc lại thành UntrustedText.
        string html = fetched.Body.Raw;
        bool isMarkup = fetched.ContentType is null
            or "text/html" or "application/xhtml+xml";

        string title = isMarkup ? ExtractTitle(html) : string.Empty;
        string text = isMarkup ? StripMarkup(html) : Collapse(html);

        if (text.Length > _options.MaxTextLength)
        {
            text = string.Concat(text.AsSpan(0, _options.MaxTextLength).TrimEnd(), "…");
        }

        if (text.Length < _options.MinTextLength) return null;

        return new WebDocument
        {
            Url = fetched.Url,
            Title = UntrustedText.FromWeb(title.Length > 0 ? title : fetched.Url.Host),
            Text = UntrustedText.FromWeb(text),
            RetrievedAt = retrievedAt,
            ContentType = fetched.ContentType,
            ByteCount = fetched.ByteCount,
            Truncated = fetched.Truncated,
        };
    }

    /// <summary>Lấy nội dung &lt;title&gt; — làm TRƯỚC khi xoá thẻ, vì title nằm trong head mà head thì bị bỏ.</summary>
    public static string ExtractTitle(string html)
    {
        ArgumentNullException.ThrowIfNull(html);

        int open = html.IndexOf("<title", StringComparison.OrdinalIgnoreCase);
        if (open < 0) return string.Empty;

        int textStart = html.IndexOf('>', open);
        if (textStart < 0) return string.Empty;

        int close = html.IndexOf("</title", textStart, StringComparison.OrdinalIgnoreCase);
        if (close < 0) return string.Empty;

        return Collapse(WebUtility.HtmlDecode(html[(textStart + 1)..close]));
    }

    /// <summary>
    /// Máy trạng thái đi qua HTML một lượt: giữ text, bỏ thẻ, bỏ hẳn vùng thô.
    /// </summary>
    public static string StripMarkup(string html)
    {
        ArgumentNullException.ThrowIfNull(html);

        var output = new StringBuilder(html.Length);
        int i = 0;

        while (i < html.Length)
        {
            char c = html[i];

            if (c != '<')
            {
                output.Append(c);
                i++;
                continue;
            }

            // --- Comment: <!-- ... --> ---
            if (html.AsSpan(i).StartsWith("<!--", StringComparison.Ordinal))
            {
                int end = html.IndexOf("-->", i + 4, StringComparison.Ordinal);
                i = end < 0 ? html.Length : end + 3;
                output.Append(' ');
                continue;
            }

            // --- Doctype / CDATA: <! ... > ---
            if (i + 1 < html.Length && html[i + 1] == '!')
            {
                int end = html.IndexOf('>', i);
                i = end < 0 ? html.Length : end + 1;
                continue;
            }

            var (name, isClosing, tagEnd) = ReadTagName(html, i);

            // Dấu '<' đứng một mình ("a < b") — là ký tự văn bản, không phải thẻ.
            if (name.Length == 0)
            {
                output.Append(c);
                i++;
                continue;
            }

            if (!isClosing && DroppedElements.Contains(name))
            {
                // Bỏ TỪ thẻ mở ĐẾN thẻ đóng cùng tên — kể cả nội dung bên trong.
                i = SkipElement(html, tagEnd, name);
                output.Append(' ');
                continue;
            }

            if (BlockElements.Contains(name))
            {
                output.Append('\n');
            }
            else
            {
                output.Append(' ');   // thẻ inline vẫn là ranh giới từ: "a<b>b</b>" != "ab"
            }

            i = tagEnd;
        }

        // Giải mã thực thể SAU khi thẻ đã biến mất — không bao giờ làm trước.
        return Collapse(WebUtility.HtmlDecode(output.ToString()));
    }

    /// <summary>Đọc tên thẻ tại vị trí '<'. Trả về tên rỗng nếu đây không phải thẻ.</summary>
    private static (string Name, bool IsClosing, int TagEnd) ReadTagName(string html, int start)
    {
        int i = start + 1;
        bool closing = false;

        if (i < html.Length && html[i] == '/')
        {
            closing = true;
            i++;
        }

        int nameStart = i;
        while (i < html.Length && (char.IsLetterOrDigit(html[i]) || html[i] == '-')) i++;

        if (i == nameStart) return (string.Empty, false, start + 1);

        string name = html[nameStart..i];

        // Đi tới '>' đóng thẻ, nhảy qua các thuộc tính (kể cả khi giá trị
        // thuộc tính có chứa '>' trong dấu ngoặc kép).
        while (i < html.Length && html[i] != '>')
        {
            if (html[i] is '"' or '\'')
            {
                char quote = html[i++];
                while (i < html.Length && html[i] != quote) i++;
            }

            i++;
        }

        return (name, closing, Math.Min(i + 1, html.Length));
    }

    /// <summary>Nhảy qua toàn bộ nội dung của một thẻ thô tới thẻ đóng của nó.</summary>
    private static int SkipElement(string html, int contentStart, string name)
    {
        int i = contentStart;

        while (i < html.Length)
        {
            int next = html.IndexOf('<', i);
            if (next < 0) return html.Length;

            var (tagName, isClosing, tagEnd) = ReadTagName(html, next);

            if (isClosing && tagName.Equals(name, StringComparison.OrdinalIgnoreCase))
            {
                return tagEnd;
            }

            i = tagName.Length == 0 ? next + 1 : tagEnd;
        }

        // Thẻ đóng thiếu (HTML thật hay bị) -> bỏ tới hết chuỗi cho an toàn.
        return html.Length;
    }

    /// <summary>Gom khoảng trắng: nhiều dòng trống thành một, nhiều dấu cách thành một.</summary>
    public static string Collapse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var output = new StringBuilder(text.Length);
        bool pendingSpace = false;
        bool pendingNewline = false;

        foreach (char c in text)
        {
            if (c is '\n' or '\r')
            {
                if (output.Length > 0) pendingNewline = true;
                continue;
            }

            if (char.IsWhiteSpace(c) || char.IsControl(c))
            {
                if (output.Length > 0) pendingSpace = true;
                continue;
            }

            if (pendingNewline) output.Append('\n');
            else if (pendingSpace) output.Append(' ');

            pendingSpace = false;
            pendingNewline = false;
            output.Append(c);
        }

        return output.ToString();
    }
}
