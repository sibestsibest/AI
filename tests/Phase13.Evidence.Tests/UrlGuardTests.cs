using System.Net;
using Ai.Phase13.Internet;

namespace Ai.Phase13.Tests;

/// <summary>
/// Bộ test quan trọng nhất của Phase 12.
///
/// Mọi test khác kiểm tra tính năng; những test này kiểm tra AN TOÀN. Một
/// tính năng sai thì agent trả lời sai; một lỗ SSRF thì kẻ tấn công lấy được
/// khoá truy cập của cả hệ thống.
/// </summary>
public class UrlGuardTests
{
    private static readonly UrlGuard Guard = new(new UrlGuardOptions { SkipDnsResolution = true });

    // ------------------------------------------------------------------
    // Lớp 1: scheme
    // ------------------------------------------------------------------

    [Theory]
    [InlineData("file:///C:/Windows/win.ini")]
    [InlineData("file:///etc/passwd")]
    [InlineData("ftp://example.com/secret.txt")]
    [InlineData("gopher://example.com:70/")]
    public void ChanSchemeKhongPhaiHttp(string url)
    {
        var verdict = Guard.Check(new Uri(url));

        Assert.False(verdict.Allowed);
        Assert.Contains("scheme", verdict.Reason);
    }

    [Theory]
    [InlineData("http://example.com/")]
    [InlineData("https://example.com/a/b?c=d")]
    public void ChoPhepHttpVaHttps(string url)
    {
        Assert.True(Guard.Check(new Uri(url)).Allowed);
    }

    // ------------------------------------------------------------------
    // Lớp 2: cú pháp
    // ------------------------------------------------------------------

    /// <summary>
    /// "http://google.com@169.254.169.254/" TRỎ TỚI 169.254.169.254, không phải
    /// Google — phần trước '@' chỉ là userinfo. Đây là mẹo che tên miền kinh
    /// điển, và nó lừa được cả người xem log.
    /// </summary>
    [Fact]
    public void ChanUserInfo_MeoCheTenMien()
    {
        var url = new Uri("http://google.com@169.254.169.254/latest/meta-data/");

        Assert.Equal("169.254.169.254", url.Host);   // xác nhận host thật là gì

        var verdict = Guard.Check(url);

        Assert.False(verdict.Allowed);
        Assert.Contains("userinfo", verdict.Reason);
    }

    [Theory]
    [InlineData("http://example.com:22/")]      // SSH
    [InlineData("http://example.com:3306/")]    // MySQL
    [InlineData("http://example.com:6379/")]    // Redis
    [InlineData("http://example.com:11211/")]   // Memcached
    public void ChanCongNgoaiDanhSach(string url)
    {
        var verdict = Guard.Check(new Uri(url));

        Assert.False(verdict.Allowed);
        Assert.Contains("cổng", verdict.Reason);
    }

    // ------------------------------------------------------------------
    // Lớp 3: tên máy
    // ------------------------------------------------------------------

    [Theory]
    [InlineData("http://localhost/")]
    [InlineData("http://redis.local/")]
    [InlineData("http://vault.internal/")]
    [InlineData("http://metadata.google.internal/computeMetadata/v1/")]
    public void ChanTenMayNoiBo(string url)
    {
        Assert.False(Guard.Check(new Uri(url)).Allowed);
    }

    // ------------------------------------------------------------------
    // Lớp 4: địa chỉ IP
    // ------------------------------------------------------------------

    /// <summary>
    /// Endpoint metadata của cloud. Đây là mục tiêu số một của SSRF: nó trả về
    /// thông tin đăng nhập tạm của chính máy chủ, không cần xác thực gì cả.
    /// </summary>
    [Theory]
    [InlineData("http://169.254.169.254/latest/meta-data/iam/security-credentials/")]
    [InlineData("http://169.254.170.2/v2/credentials/")]
    public void ChanEndpointMetadataCuaCloud(string url)
    {
        var verdict = Guard.Check(new Uri(url));

        Assert.False(verdict.Allowed);
        Assert.Contains("169.254", verdict.Reason);
    }

    [Theory]
    [InlineData("http://127.0.0.1:8080/")]
    [InlineData("http://127.1.2.3/")]
    [InlineData("http://10.0.0.5:8080/admin")]
    [InlineData("http://172.16.0.1/")]
    [InlineData("http://172.31.255.254/")]
    [InlineData("http://192.168.1.1/")]
    [InlineData("http://100.64.0.1/")]
    [InlineData("http://0.0.0.0/")]
    [InlineData("http://198.18.0.1/")]
    [InlineData("http://224.0.0.1/")]
    [InlineData("http://240.0.0.1/")]
    public void ChanDaiIpNoiBoVaDuTru(string url)
    {
        Assert.False(Guard.Check(new Uri(url)).Allowed);
    }

    [Theory]
    [InlineData("http://[::1]/")]
    [InlineData("http://[fe80::1]/")]
    [InlineData("http://[fc00::1]/")]
    [InlineData("http://[fd12:3456::1]/")]
    public void ChanIpv6NoiBo(string url)
    {
        Assert.False(Guard.Check(new Uri(url)).Allowed);
    }

    /// <summary>
    /// ::ffff:127.0.0.1 là 127.0.0.1 viết dưới dạng IPv6. Bỏ sót dạng này là
    /// bỏ sót cả lớp phòng vệ loopback.
    /// </summary>
    [Fact]
    public void ChanIpv4BocTrongIpv6()
    {
        Assert.True(UrlGuard.IsUnsafe(IPAddress.Parse("::ffff:127.0.0.1"), out string reason));
        Assert.Contains("loopback", reason);

        Assert.True(UrlGuard.IsUnsafe(IPAddress.Parse("::ffff:169.254.169.254"), out string metadata));
        Assert.Contains("169.254", metadata);
    }

    /// <summary>
    /// 2130706433 là 127.0.0.1 viết dạng số nguyên. .NET phân giải được dạng
    /// này, nên guard cũng phải chặn được.
    /// </summary>
    [Fact]
    public void ChanLoopbackVietDangSoNguyen()
    {
        Assert.True(IPAddress.TryParse("2130706433", out var parsed));
        Assert.True(UrlGuard.IsUnsafe(parsed!, out _));
    }

    [Theory]
    [InlineData("http://8.8.8.8/")]
    [InlineData("https://1.1.1.1/")]
    [InlineData("https://93.184.216.34/")]
    public void ChoPhepIpCongCong(string url)
    {
        Assert.True(Guard.Check(new Uri(url)).Allowed);
    }

    // ------------------------------------------------------------------
    // Danh sách trắng
    // ------------------------------------------------------------------

    [Fact]
    public void DanhSachTrang_ChiChoPhepTenMienDaKhaiBao()
    {
        var guard = new UrlGuard(new UrlGuardOptions
        {
            SkipDnsResolution = true,
            AllowedHostSuffixes = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "wikipedia.org" },
        });

        Assert.True(guard.Check(new Uri("https://en.wikipedia.org/wiki/SSRF")).Allowed);
        Assert.True(guard.Check(new Uri("https://wikipedia.org/")).Allowed);
        Assert.False(guard.Check(new Uri("https://example.com/")).Allowed);
    }

    /// <summary>
    /// "wikipedia.org.attacker.com" KHÔNG được coi là wikipedia.org. Khớp đuôi
    /// phải tính theo ranh giới dấu chấm, không phải EndsWith thô.
    /// </summary>
    [Fact]
    public void DanhSachTrang_KhongBiLuaBoiTenMienGiaMao()
    {
        var guard = new UrlGuard(new UrlGuardOptions
        {
            SkipDnsResolution = true,
            AllowedHostSuffixes = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "wikipedia.org" },
        });

        Assert.False(guard.Check(new Uri("https://wikipedia.org.attacker.com/")).Allowed);
        Assert.False(guard.Check(new Uri("https://notwikipedia.org/")).Allowed);
    }

    // ------------------------------------------------------------------
    // DNS
    // ------------------------------------------------------------------

    /// <summary>Tên máy không tồn tại thì bị chặn, và chặn có lý do đọc được.</summary>
    [Fact]
    public async Task PhanGiaiDnsThatBai_ThiChan()
    {
        var guard = new UrlGuard();

        var verdict = await guard.CheckAsync(
            new Uri("http://khong-ton-tai-that-su-12345.invalid/"), CancellationToken.None);

        Assert.False(verdict.Allowed);
        Assert.Contains("phân giải", verdict.Reason);
    }

    /// <summary>
    /// DNS rebinding: một tên miền công cộng trỏ về 127.0.0.1. Chặn theo tên
    /// không bắt được, chỉ soi IP sau phân giải mới bắt được.
    /// </summary>
    [Fact]
    public async Task ChanTenMienPhanGiaiVeLoopback()
    {
        var guard = new UrlGuard();

        // localtest.me là tên miền công cộng thật, luôn trỏ về 127.0.0.1.
        var verdict = await guard.CheckAsync(
            new Uri("http://localtest.me/"), CancellationToken.None);

        // Không có mạng thì thành lỗi phân giải; có mạng thì thành chặn vì IP.
        // Cả hai đều là KHÔNG CHO PHÉP — đó là điều test này khẳng định.
        Assert.False(verdict.Allowed);
    }

    [Fact]
    public async Task CheckAsync_TonTrongHuy()
    {
        var guard = new UrlGuard();
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => guard.CheckAsync(new Uri("http://example.com/"), cts.Token));
    }

    [Fact]
    public void IpVietThangKhongCanPhanGiaiDns()
    {
        // SkipDnsResolution = false mà vẫn không chạm mạng, vì host là IP.
        var guard = new UrlGuard();

        Assert.False(guard.Check(new Uri("http://10.1.2.3/")).Allowed);
        Assert.True(guard.Check(new Uri("http://8.8.8.8/")).Allowed);
    }
}
