using System.Net;
using System.Net.Sockets;

namespace Ai.Phase12.Internet;

public sealed record UrlVerdict(bool Allowed, string Reason)
{
    public static UrlVerdict Ok(string reason = "hợp lệ") => new(true, reason);

    public static UrlVerdict Block(string reason) => new(false, reason);
}

/// <summary>Cửa kiểm soát: mọi URL phải đi qua đây trước khi có ai mở kết nối.</summary>
public interface IUrlGuard
{
    /// <summary>Kiểm tra nhanh, không cần mạng: scheme, cổng, tên miền, IP viết thẳng.</summary>
    UrlVerdict Check(Uri url);

    /// <summary>Kiểm tra đầy đủ: thêm bước phân giải DNS và soi mọi địa chỉ trả về.</summary>
    Task<UrlVerdict> CheckAsync(Uri url, CancellationToken cancellationToken = default);
}

public sealed record UrlGuardOptions
{
    /// <summary>Chỉ hai scheme này. Không có file://, ftp://, gopher://, data://.</summary>
    public IReadOnlySet<string> AllowedSchemes { get; init; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "http", "https" };

    /// <summary>Cổng cho phép. Rỗng = cho phép mọi cổng (không khuyến khích).</summary>
    public IReadOnlySet<int> AllowedPorts { get; init; } = new HashSet<int> { 80, 443, 8080, 8443 };

    /// <summary>Đuôi tên miền nội bộ — chặn theo tên, trước cả khi phân giải DNS.</summary>
    public IReadOnlySet<string> BlockedHostSuffixes { get; init; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".local", ".localhost", ".internal", ".intranet", ".corp", ".home.arpa",
        };

    /// <summary>Tên máy chặn thẳng.</summary>
    public IReadOnlySet<string> BlockedHosts { get; init; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "localhost", "metadata.google.internal", "metadata",
        };

    /// <summary>
    /// Chỉ cho phép các tên miền này (danh sách trắng). Rỗng = không giới hạn.
    /// Môi trường thật nên bật cái này — nó là biện pháp mạnh nhất trong cả file.
    /// </summary>
    public IReadOnlySet<string> AllowedHostSuffixes { get; init; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    /// <summary>Bỏ bước phân giải DNS — chỉ dùng trong test chạy offline.</summary>
    public bool SkipDnsResolution { get; init; }
}

/// <summary>
/// CHỐNG SSRF (Server-Side Request Forgery).
///
/// Đây là lỗ hổng nguy hiểm nhất của toàn bộ Phase 12, và nó rất dễ mắc.
///
/// Kịch bản: agent nhận một URL — từ người dùng, từ kết quả tìm kiếm, hoặc từ
/// một liên kết trong trang web nó vừa đọc — rồi ngoan ngoãn gọi HTTP GET.
/// Nhưng agent chạy TRONG mạng nội bộ. Nên những URL này đều có thật:
///
///     http://169.254.169.254/latest/meta-data/iam/security-credentials/
///         -> khoá IAM của cả máy chủ, trên AWS
///     http://localhost:6379/
///         -> Redis không mật khẩu
///     http://10.0.0.5:8080/admin/
///         -> trang quản trị nội bộ
///     file:///etc/passwd  |  file:///C:/Windows/win.ini
///         -> đọc file ngay trên máy
///
/// Kẻ tấn công không cần vào được mạng nội bộ. Nó chỉ cần nhờ agent vào giúp.
///
/// BỐN LỚP KIỂM TRA:
///
///   1. Scheme     — chỉ http/https. Chặn file, ftp, gopher, data, blob.
///   2. Cú pháp    — không userinfo (http://user:pass@host là mẹo che tên miền
///                   thật: "http://google.com@169.254.169.254" trỏ tới METADATA,
///                   không phải Google), cổng phải nằm trong danh sách.
///   3. Tên máy    — chặn localhost, .local, .internal, tên metadata của cloud.
///   4. Địa chỉ IP — phân giải DNS rồi soi TỪNG địa chỉ trả về. Đây là lớp
///                   không thể thiếu, vì một tên miền công cộng hoàn toàn có
///                   thể trỏ về 127.0.0.1 — kỹ thuật "DNS rebinding". Tên miền
///                   trông vô hại, IP thì không.
///
/// GIỚI HẠN CÒN LẠI (phải nói thẳng): giữa lúc guard phân giải DNS và lúc
/// HttpClient mở kết nối, DNS có thể đổi — cửa sổ TOCTOU. Bịt hẳn thì phải
/// tự nối socket tới đúng IP đã kiểm tra rồi đặt lại header Host. Ở quy mô
/// project học thuật này, guard + danh sách trắng tên miền là mức đánh đổi
/// hợp lý; mã đã tách thành interface nên thay được mà không sửa chỗ khác.
/// </summary>
public sealed class UrlGuard(UrlGuardOptions? options = null) : IUrlGuard
{
    private readonly UrlGuardOptions _options = options ?? new UrlGuardOptions();

    public UrlGuardOptions Options => _options;

    public UrlVerdict Check(Uri url)
    {
        ArgumentNullException.ThrowIfNull(url);

        if (!url.IsAbsoluteUri)
        {
            return UrlVerdict.Block("URL không tuyệt đối");
        }

        // --- Lớp 1: scheme ---
        if (!_options.AllowedSchemes.Contains(url.Scheme))
        {
            return UrlVerdict.Block($"scheme '{url.Scheme}' không được phép (chỉ http/https)");
        }

        // --- Lớp 2: cú pháp ---
        if (!string.IsNullOrEmpty(url.UserInfo))
        {
            return UrlVerdict.Block("URL chứa userinfo — mẹo che tên miền thật");
        }

        if (_options.AllowedPorts.Count > 0 && !_options.AllowedPorts.Contains(url.Port))
        {
            return UrlVerdict.Block($"cổng {url.Port} không nằm trong danh sách cho phép");
        }

        // --- Lớp 3: tên máy ---
        string host = url.Host;

        if (string.IsNullOrWhiteSpace(host))
        {
            return UrlVerdict.Block("URL không có tên máy");
        }

        if (_options.BlockedHosts.Contains(host))
        {
            return UrlVerdict.Block($"tên máy '{host}' bị chặn");
        }

        foreach (string suffix in _options.BlockedHostSuffixes)
        {
            if (host.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            {
                return UrlVerdict.Block($"tên máy '{host}' thuộc vùng nội bộ '{suffix}'");
            }
        }

        if (_options.AllowedHostSuffixes.Count > 0)
        {
            bool allowed = _options.AllowedHostSuffixes.Any(s =>
                host.Equals(s, StringComparison.OrdinalIgnoreCase) ||
                host.EndsWith("." + s.TrimStart('.'), StringComparison.OrdinalIgnoreCase));

            if (!allowed)
            {
                return UrlVerdict.Block($"tên máy '{host}' không nằm trong danh sách trắng");
            }
        }

        // --- Lớp 4a: IP viết thẳng trong URL, không cần DNS ---
        if (IPAddress.TryParse(host.Trim('[', ']'), out var literal))
        {
            return IsUnsafe(literal, out string why)
                ? UrlVerdict.Block($"IP {literal} {why}")
                : UrlVerdict.Ok($"IP công cộng {literal}");
        }

        return UrlVerdict.Ok($"tên máy '{host}' qua được kiểm tra cú pháp");
    }

    public async Task<UrlVerdict> CheckAsync(Uri url, CancellationToken cancellationToken = default)
    {
        var syntax = Check(url);
        if (!syntax.Allowed) return syntax;

        // IP viết thẳng đã được soi ở Check() — không có gì để phân giải.
        if (IPAddress.TryParse(url.Host.Trim('[', ']'), out _)) return syntax;

        if (_options.SkipDnsResolution) return syntax;

        // --- Lớp 4b: phân giải DNS và soi MỌI địa chỉ ---
        IPAddress[] addresses;
        try
        {
            addresses = await Dns.GetHostAddressesAsync(url.Host, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (SocketException ex)
        {
            return UrlVerdict.Block($"không phân giải được tên máy '{url.Host}': {ex.SocketErrorCode}");
        }
        catch (ArgumentException ex)
        {
            return UrlVerdict.Block($"tên máy '{url.Host}' không hợp lệ: {ex.Message}");
        }

        if (addresses.Length == 0)
        {
            return UrlVerdict.Block($"tên máy '{url.Host}' không có địa chỉ nào");
        }

        // Chặn nếu BẤT KỲ địa chỉ nào không an toàn — không phải "nếu tất cả".
        // Một tên miền trỏ tới cả IP công cộng lẫn 127.0.0.1 thì vẫn là lối
        // vào mạng nội bộ, vì ta không chọn được HttpClient sẽ dùng địa chỉ nào.
        foreach (var address in addresses)
        {
            if (IsUnsafe(address, out string why))
            {
                return UrlVerdict.Block($"tên máy '{url.Host}' phân giải ra {address} — {why}");
            }
        }

        return UrlVerdict.Ok($"tên máy '{url.Host}' phân giải ra {addresses.Length} địa chỉ công cộng");
    }

    /// <summary>
    /// Một địa chỉ IP có thuộc vùng KHÔNG được với tới hay không.
    ///
    /// Danh sách này phải đầy đủ. Bỏ sót một dải là bỏ ngỏ một lối vào —
    /// và dải hay bị bỏ sót nhất chính là 169.254.0.0/16 (link-local), nơi
    /// mọi nhà cung cấp cloud đặt endpoint metadata.
    /// </summary>
    public static bool IsUnsafe(IPAddress address, out string reason)
    {
        ArgumentNullException.ThrowIfNull(address);

        if (IPAddress.IsLoopback(address))
        {
            reason = "là địa chỉ loopback";
            return true;
        }

        if (address.Equals(IPAddress.Any) || address.Equals(IPAddress.IPv6Any))
        {
            reason = "là địa chỉ 'any'";
            return true;
        }

        if (address.AddressFamily == AddressFamily.InterNetworkV6)
        {
            // IPv6 bọc IPv4 (::ffff:127.0.0.1) phải được soi theo IPv4 thật.
            if (address.IsIPv4MappedToIPv6)
            {
                return IsUnsafe(address.MapToIPv4(), out reason);
            }

            if (address.IsIPv6LinkLocal)
            {
                reason = "là IPv6 link-local (fe80::/10)";
                return true;
            }

            if (address.IsIPv6SiteLocal)
            {
                reason = "là IPv6 site-local (fec0::/10)";
                return true;
            }

            if (address.IsIPv6UniqueLocal)
            {
                reason = "là IPv6 unique-local (fc00::/7)";
                return true;
            }

            if (address.IsIPv6Multicast)
            {
                reason = "là IPv6 multicast";
                return true;
            }

            reason = string.Empty;
            return false;
        }

        if (address.AddressFamily != AddressFamily.InterNetwork)
        {
            reason = $"thuộc họ địa chỉ không hỗ trợ ({address.AddressFamily})";
            return true;
        }

        byte[] b = address.GetAddressBytes();

        // 0.0.0.0/8 — "this network"
        if (b[0] == 0)
        {
            reason = "thuộc 0.0.0.0/8";
            return true;
        }

        // 10.0.0.0/8 — riêng tư (RFC 1918)
        if (b[0] == 10)
        {
            reason = "thuộc mạng riêng 10.0.0.0/8";
            return true;
        }

        // 100.64.0.0/10 — CGNAT (RFC 6598)
        if (b[0] == 100 && b[1] >= 64 && b[1] <= 127)
        {
            reason = "thuộc CGNAT 100.64.0.0/10";
            return true;
        }

        // 127.0.0.0/8 — loopback (đã bắt ở trên, giữ lại cho chắc)
        if (b[0] == 127)
        {
            reason = "thuộc loopback 127.0.0.0/8";
            return true;
        }

        // 169.254.0.0/16 — link-local. ĐÂY là endpoint metadata của cloud.
        if (b[0] == 169 && b[1] == 254)
        {
            reason = "thuộc link-local 169.254.0.0/16 (endpoint metadata của cloud)";
            return true;
        }

        // 172.16.0.0/12 — riêng tư (RFC 1918)
        if (b[0] == 172 && b[1] >= 16 && b[1] <= 31)
        {
            reason = "thuộc mạng riêng 172.16.0.0/12";
            return true;
        }

        // 192.0.0.0/24 — IETF protocol assignments
        if (b[0] == 192 && b[1] == 0 && b[2] == 0)
        {
            reason = "thuộc 192.0.0.0/24";
            return true;
        }

        // 192.168.0.0/16 — riêng tư (RFC 1918)
        if (b[0] == 192 && b[1] == 168)
        {
            reason = "thuộc mạng riêng 192.168.0.0/16";
            return true;
        }

        // 198.18.0.0/15 — benchmark (RFC 2544)
        if (b[0] == 198 && (b[1] == 18 || b[1] == 19))
        {
            reason = "thuộc dải benchmark 198.18.0.0/15";
            return true;
        }

        // 224.0.0.0/4 — multicast, và 240.0.0.0/4 — dự trữ
        if (b[0] >= 224)
        {
            reason = b[0] >= 240 ? "thuộc dải dự trữ 240.0.0.0/4" : "thuộc dải multicast 224.0.0.0/4";
            return true;
        }

        reason = string.Empty;
        return false;
    }
}
