using Ai.Phase14.Internet;

namespace Ai.Phase14.Tests;

public class ContentExtractorTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 9, 0, 0, TimeSpan.Zero);

    private static FetchResult Html(string html, string? contentType = "text/html") => new()
    {
        Success = true,
        Url = new Uri("https://example.com/page"),
        Body = UntrustedText.FromWeb(html),
        StatusCode = 200,
        ContentType = contentType,
        ByteCount = html.Length,
    };

    [Fact]
    public void LayDuocTieuDe()
    {
        Assert.Equal("Trang thử", HtmlContentExtractor.ExtractTitle("<html><head><title>Trang thử</title></head></html>"));
        Assert.Equal("Có thuộc tính", HtmlContentExtractor.ExtractTitle("<title lang=\"vi\">Có thuộc tính</title>"));
        Assert.Equal(string.Empty, HtmlContentExtractor.ExtractTitle("<html><body>không có title</body></html>"));
    }

    [Fact]
    public void GiaiMaThucTheTrongTieuDe()
    {
        Assert.Equal("A & B < C", HtmlContentExtractor.ExtractTitle("<title>A &amp; B &lt; C</title>"));
    }

    /// <summary>
    /// Đây là lý do không dùng regex `<[^>]*>`: dấu '<' trong JavaScript sẽ
    /// làm regex cắt sai chỗ và để lại nguyên mã script trong phần "văn bản".
    /// </summary>
    [Fact]
    public void BoHanNoiDungScript_KeCaKhiScriptCoDauBeHon()
    {
        string html = """
            <html><body>
            <p>Nội dung thật</p>
            <script>if (a < b) { document.write("</div> xin chào"); }</script>
            <p>Nội dung thật thứ hai</p>
            </body></html>
            """;

        string text = HtmlContentExtractor.StripMarkup(html);

        Assert.Contains("Nội dung thật", text);
        Assert.Contains("Nội dung thật thứ hai", text);
        Assert.DoesNotContain("document.write", text);
        Assert.DoesNotContain("a < b", text);
    }

    [Fact]
    public void BoStyleNoscriptVaComment()
    {
        string html = """
            <style>body { color: red; }</style>
            <!-- ghi chú bí mật -->
            <noscript>bật javascript đi</noscript>
            <p>Phần cần giữ</p>
            """;

        string text = HtmlContentExtractor.StripMarkup(html);

        Assert.Contains("Phần cần giữ", text);
        Assert.DoesNotContain("color: red", text);
        Assert.DoesNotContain("ghi chú bí mật", text);
        Assert.DoesNotContain("bật javascript", text);
    }

    /// <summary>
    /// Giải mã thực thể phải làm SAU khi bỏ thẻ. Làm trước thì
    /// "&lt;script&gt;alert(1)&lt;/script&gt;" biến thành thẻ script thật và
    /// nội dung của nó lọt vào văn bản.
    /// </summary>
    [Fact]
    public void GiaiMaThucTheSauKhiBoThe_KhongTuTaoRaTheMoi()
    {
        string html = "<p>Ví dụ mã: &lt;script&gt;alert(1)&lt;/script&gt; hết ví dụ</p>";

        string text = HtmlContentExtractor.StripMarkup(html);

        // Chuỗi được hiện ra dưới dạng VĂN BẢN, và alert(1) vẫn còn nguyên vì
        // nó chưa từng là script thật — đúng như trang gốc muốn diễn đạt.
        Assert.Contains("<script>alert(1)</script>", text);
        Assert.Contains("hết ví dụ", text);
    }

    [Fact]
    public void TheKhoiTaoRanhGioiDongVaTheInlineTaoRanhGioiTu()
    {
        Assert.Contains("\n", HtmlContentExtractor.StripMarkup("<p>một</p><p>hai</p>"));

        // "a<b>b</b>" không được dính thành "ab"
        string inline = HtmlContentExtractor.StripMarkup("<span>một</span><span>hai</span>");
        Assert.Equal("một hai", inline);
    }

    [Fact]
    public void ThuocTinhChuaDauLonHon_KhongLamHongViecDocThe()
    {
        string html = """<a title="1 > 0" href="/x">liên kết</a> xong""";

        string text = HtmlContentExtractor.StripMarkup(html);

        Assert.Contains("liên kết", text);
        Assert.Contains("xong", text);
        Assert.DoesNotContain("href", text);
    }

    [Fact]
    public void TheDongThieu_ThiBoTiHetChuKhongVangRaScript()
    {
        // <script> không đóng: phải bỏ tới hết chuỗi, không được để lọt.
        string text = HtmlContentExtractor.StripMarkup("<p>giữ</p><script>evil();");

        Assert.Contains("giữ", text);
        Assert.DoesNotContain("evil", text);
    }

    [Fact]
    public void GomKhoangTrang()
    {
        Assert.Equal("a b", HtmlContentExtractor.Collapse("a     b"));
        Assert.Equal("a\nb", HtmlContentExtractor.Collapse("a\n\n\n\nb"));
        Assert.Equal("a b", HtmlContentExtractor.Collapse("   a \t b   "));
    }

    [Fact]
    public void BocThanhWebDocument()
    {
        string html = """
            <html><head><title>SSRF là gì</title></head>
            <body><p>SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ của chính nó.</p></body></html>
            """;

        var document = new HtmlContentExtractor().Extract(Html(html), Now);

        Assert.NotNull(document);
        Assert.Equal("SSRF là gì", document.Title.Raw);
        Assert.Contains("lỗ hổng", document.Text.Raw);
        Assert.Equal(Now, document.RetrievedAt);
        Assert.Equal("example.com", document.Host);
        Assert.True(document.HasText);
    }

    /// <summary>Trang quá ít chữ (trang chặn bot, trang lỗi) -> coi như không có nội dung.</summary>
    [Fact]
    public void TrangQuaNgan_TraVeNull()
    {
        Assert.Null(new HtmlContentExtractor().Extract(Html("<html><body><p>ngắn</p></body></html>"), Now));
    }

    [Fact]
    public void CatVanBanQuaDai()
    {
        string html = "<p>" + new string('x', 5_000) + "</p>";

        var document = new HtmlContentExtractor(new ContentExtractorOptions { MaxTextLength = 100 })
            .Extract(Html(html), Now);

        Assert.NotNull(document);
        Assert.True(document.Text.Length <= 101, $"dài {document.Text.Length}");
        Assert.EndsWith("…", document.Text.Raw);
    }

    [Fact]
    public void KhongBocKhiTaiThatBai()
    {
        var failed = FetchResult.Fail(new Uri("https://example.com/x"), "HTTP 404");

        Assert.Null(new HtmlContentExtractor().Extract(failed, Now));
    }

    [Fact]
    public void VanBanThuan_KhongBocTheGiCa()
    {
        var plain = Html("Đây là văn bản thuần, dài hơn bốn mươi ký tự để không bị loại.", "text/plain");

        var document = new HtmlContentExtractor().Extract(plain, Now);

        Assert.NotNull(document);
        Assert.Contains("văn bản thuần", document.Text.Raw);
        Assert.Equal("example.com", document.Title.Raw);   // không có <title> -> lấy host
    }

    [Theory]
    [InlineData("text/html", true)]
    [InlineData("text/plain", true)]
    [InlineData("application/json", true)]
    [InlineData(null, true)]
    [InlineData("image/png", false)]
    [InlineData("application/pdf", false)]
    [InlineData("application/octet-stream", false)]
    public void ChiBocKieuNoiDungHieuDuoc(string? contentType, bool expected)
    {
        Assert.Equal(expected, new HtmlContentExtractor().CanExtract(contentType));
    }

    [Fact]
    public void GiuNhanDaBiCat()
    {
        var truncated = Html("<p>" + new string('y', 200) + "</p>") with { Truncated = true, ByteCount = 2048 };

        var document = new HtmlContentExtractor().Extract(truncated, Now);

        Assert.NotNull(document);
        Assert.True(document.Truncated);
        Assert.Equal(2048, document.ByteCount);
    }
}
