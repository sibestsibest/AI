using Ai.Phase16.Memory;
using Ai.Providers;
using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.Providers;
using Ai.Providers.Resilience;
using Ai.Providers.Routing;
using Ai.Providers.Security;
using Ai.Providers.Usage;

namespace Ai.Providers.Tests;

internal static class Cfg
{
    public static readonly DateTimeOffset Now = new(2026, 1, 1, 9, 0, 0, TimeSpan.Zero);

    public static ManualClock Clock() => new(Now);

    public static ProviderOptions Local(string name = "local", params AiTaskType[] preferredFor) => new()
    {
        Name = name,
        Type = ProviderKind.Scripted,
        Endpoint = "",
        DefaultModel = "scripted-model",
        MaxRetries = 0,
        PreferredFor = preferredFor,
    };

    public static ProviderOptions Remote(string name = "remote", params AiTaskType[] preferredFor) => new()
    {
        Name = name,
        Type = ProviderKind.OpenAiCompatible,
        Endpoint = "https://provider.invalid/v1",
        ApiKeyEnvironmentVariable = "TEST_KEY",
        DefaultModel = "scripted-model",
        MaxRetries = 0,
        PreferredFor = preferredFor,
    };

    public static AiProvidersOptions Options(ProviderStrategy strategy, params ProviderOptions[] providers) => new()
    {
        Strategy = strategy,
        Providers = providers,
    };

    public static AiRequest Ask(string prompt = "câu hỏi thử", AiTaskType task = AiTaskType.GeneralChat) =>
        new() { UserPrompt = prompt, TaskType = task };
}

public class SecretResolverTests
{
    [Fact]
    public void DocDuocBiMatTuBienMoiTruong()
    {
        var resolver = new SecretResolver(environment: name => name == "MY_KEY" ? "giá-trị-bí-mật" : null);

        Assert.Equal("giá-trị-bí-mật", resolver.Resolve("MY_KEY"));
        Assert.True(resolver.Has("MY_KEY"));
    }

    [Fact]
    public void GiaTriTiemVaoThangHonBienMoiTruong()
    {
        var resolver = new SecretResolver(
            injected: new Dictionary<string, string> { ["MY_KEY"] = "từ-tiêm" },
            environment: _ => "từ-môi-trường");

        Assert.Equal("từ-tiêm", resolver.Resolve("MY_KEY"));
    }

    [Fact]
    public void KhongCoThiTraVeNullChuKhongNem()
    {
        var resolver = new SecretResolver(environment: _ => null);

        Assert.Null(resolver.Resolve("KHONG_CO"));
        Assert.Null(resolver.Resolve(null));
        Assert.False(resolver.Has("KHONG_CO"));
    }

    [Fact]
    public void BienRongDuocCoiNhuKhongDat()
    {
        var resolver = new SecretResolver(environment: _ => "   ");

        Assert.Null(resolver.Resolve("MY_KEY"));
    }

    /// <summary>
    /// <see cref="SecretResolver.Describe"/> đi vào log và ra màn hình, nên nó
    /// phải nói được "đã đặt hay chưa" mà KHÔNG nói giá trị.
    /// </summary>
    [Fact]
    public void MoTaKhongBaoGioChuaGiaTriBiMat()
    {
        const string secret = "sk-abcdefghijklmnopqrstuvwxyz";
        var resolver = new SecretResolver(injected: new Dictionary<string, string> { ["MY_KEY"] = secret });

        string description = resolver.Describe("MY_KEY");

        Assert.DoesNotContain(secret, description, StringComparison.Ordinal);
        Assert.Contains("đã đặt", description, StringComparison.Ordinal);

        // Chỉ nói ĐỘ DÀI — đủ để phân biệt "chưa đặt" với "đặt nhưng rỗng",
        // không đủ để lộ gì.
        Assert.Contains(secret.Length.ToString(), description, StringComparison.Ordinal);
    }
}

public class OutboundRedactorTests
{
    [Theory]
    [InlineData("khoá của tôi là sk-abcdefghijklmnopqrstuv1234")]
    [InlineData("Authorization: Bearer abcdefghijklmnopqrstuvwxyz123")]
    [InlineData("password=SieuMatKhau123")]
    [InlineData("api_key: abcdefghijklmnop")]
    [InlineData("token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefg")]
    [InlineData("Server=db1;Initial Catalog=prod;User ID=sa;")]
    [InlineData("https://api.example.com/v1?api_key=abcdef123456")]
    public void LocMoiDangBiMatCoHinhDangRoRang(string text)
    {
        string redacted = OutboundRedactor.Redact(text);

        Assert.Contains("[ĐÃ-LỌC]", redacted, StringComparison.Ordinal);
        Assert.True(OutboundRedactor.ContainsSecret(text));
    }

    [Fact]
    public void VanBanBinhThuongKhongBiCatXen()
    {
        const string text = "SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ.";

        Assert.Equal(text, OutboundRedactor.Redact(text));
        Assert.False(OutboundRedactor.ContainsSecret(text));
    }

    [Fact]
    public void KhoaRiengPemBiLocCaKhoi()
    {
        string pem = "-----BEGIN RSA PRIVATE KEY-----\nAAAABBBBCCCC\n-----END RSA PRIVATE KEY-----";

        string redacted = OutboundRedactor.Redact(pem);

        Assert.DoesNotContain("AAAABBBBCCCC", redacted, StringComparison.Ordinal);
    }
}

public class ConfigurationTests
{
    /// <summary>
    /// LUẬT AN TOÀN QUAN TRỌNG NHẤT của khâu cấu hình: file có khoá thật thì bị
    /// TỪ CHỐI toàn bộ, vì file cấu hình bị commit còn biến môi trường thì không.
    /// </summary>
    [Fact]
    public void FileChuaKhoaThatThiBiTuChoiToanBo()
    {
        string json = """
        {
          "Providers": [
            { "Name": "x", "Type": "OpenAiCompatible", "Endpoint": "https://a.invalid/v1",
              "ApiKey": "sk-abcdefghijklmnopqrstuvwxyz0123" }
          ]
        }
        """;

        var result = AiProvidersFile.Parse(json);

        Assert.True(result.HasProblems);
        Assert.Contains(result.Problems, p => p.Contains("khoá API thật", StringComparison.Ordinal));
        Assert.Same(AiProvidersOptions.LocalOnlyDefault, result.Options);
    }

    /// <summary>Thông điệp từ chối cũng KHÔNG được chứa chính cái khoá — nó sẽ vào log.</summary>
    [Fact]
    public void ThongDiepTuChoiKhongChuaKhoa()
    {
        const string key = "sk-abcdefghijklmnopqrstuvwxyz0123";
        var result = AiProvidersFile.Parse($$"""{"Providers":[{"Name":"x","ApiKey":"{{key}}"}]}""");

        Assert.All(result.Problems, p => Assert.DoesNotContain(key, p, StringComparison.Ordinal));
    }

    [Fact]
    public void KhongCoFileThiDungMacDinhChiLocal()
    {
        var result = AiProvidersFile.Load("khong-ton-tai.json", _ => null);

        Assert.Single(result.Options.Providers);
        Assert.Equal(ProviderKind.Ollama, result.Options.Providers[0].Type);
        Assert.False(result.HasProblems);
    }

    [Fact]
    public void JsonSaiCuPhapThiKhongLamSapMaQuayVeMacDinh()
    {
        var result = AiProvidersFile.Parse("{ đây không phải json }");

        Assert.True(result.HasProblems);
        Assert.Same(AiProvidersOptions.LocalOnlyDefault, result.Options);
    }

    [Fact]
    public void DocDuocCauHinhHopLe()
    {
        string json = """
        {
          "Strategy": "ConfiguredOrder",
          "DefaultProvider": "ollama",
          "Providers": [
            { "Name": "ollama", "Type": "Ollama", "Endpoint": "http://localhost:11434", "DefaultModel": "m" }
          ]
        }
        """;

        var result = AiProvidersFile.Parse(json);

        Assert.False(result.HasProblems, string.Join("; ", result.Problems));
        Assert.Equal(ProviderStrategy.ConfiguredOrder, result.Options.Strategy);
        Assert.Equal("ollama", result.Options.DefaultProvider);
    }

    [Fact]
    public void ThieuEndpointThiBaoLoi()
    {
        var problems = AiProvidersFile.Validate(Cfg.Options(ProviderStrategy.LocalFirst,
            new ProviderOptions { Name = "x", Type = ProviderKind.Ollama, Endpoint = "" }));

        Assert.Contains(problems, p => p.Contains("thiếu Endpoint", StringComparison.Ordinal));
    }

    /// <summary>HTTP trần ra ngoài máy nghĩa là khoá đi không mã hoá — phải báo.</summary>
    [Fact]
    public void HttpRaNgoaiMayThiBaoLoiNhungLocalhostThiKhong()
    {
        var remote = AiProvidersFile.Validate(Cfg.Options(ProviderStrategy.LocalFirst,
            Cfg.Remote() with { Endpoint = "http://provider.invalid/v1" }));

        var local = AiProvidersFile.Validate(Cfg.Options(ProviderStrategy.LocalFirst,
            new ProviderOptions { Name = "ollama", Type = ProviderKind.Ollama, Endpoint = "http://localhost:11434" }));

        Assert.Contains(remote, p => p.Contains("không mã hoá", StringComparison.Ordinal));
        Assert.DoesNotContain(local, p => p.Contains("không mã hoá", StringComparison.Ordinal));
    }

    [Fact]
    public void TenProviderTrungThiBaoLoi()
    {
        var problems = AiProvidersFile.Validate(Cfg.Options(ProviderStrategy.LocalFirst,
            Cfg.Local("same"), Cfg.Local("same")));

        Assert.Contains(problems, p => p.Contains("hai lần", StringComparison.Ordinal));
    }

    [Fact]
    public void ThuLaiQuaNhieuLanThiBaoLoi()
    {
        var problems = AiProvidersFile.Validate(Cfg.Options(ProviderStrategy.LocalFirst,
            Cfg.Local() with { MaxRetries = 99 }));

        Assert.Contains(problems, p => p.Contains("MaxRetries", StringComparison.Ordinal));
    }

    [Fact]
    public void ProviderKieuOpenAiMaThieuTenBienKhoaThiBaoLoi()
    {
        var problems = AiProvidersFile.Validate(Cfg.Options(ProviderStrategy.LocalFirst,
            Cfg.Remote() with { ApiKeyEnvironmentVariable = null }));

        Assert.Contains(problems, p => p.Contains("ApiKeyEnvironmentVariable", StringComparison.Ordinal));
    }

    [Fact]
    public void DefaultProviderTroToiTenKhongTonTaiThiBaoLoi()
    {
        var problems = AiProvidersFile.Validate(Cfg.Options(ProviderStrategy.LocalFirst, Cfg.Local())
            with { DefaultProvider = "khong-co" });

        Assert.Contains(problems, p => p.Contains("DefaultProvider", StringComparison.Ordinal));
    }
}

public class RetryPolicyTests
{
    [Fact]
    public async Task LoiTamThoiThiThuLaiVaCoTheThanhCong()
    {
        var provider = ScriptedAiProvider.FailsThenSucceeds("p", failures: 2, reply: "xong");
        var policy = new RetryPolicy(maxAttempts: 3, TimeSpan.FromMilliseconds(1), new Random(1));

        var response = await policy.ExecuteAsync(ct => provider.GenerateAsync(Cfg.Ask(), ct), CancellationToken.None);

        Assert.True(response.Success);
        Assert.Equal(3, provider.Calls);
    }

    /// <summary>Khoá sai thì thử lại bao nhiêu lần cũng vẫn sai — phải dừng ngay.</summary>
    [Fact]
    public async Task LoiKhongDangThuLaiThiKhongThuLai()
    {
        var provider = ScriptedAiProvider.AlwaysFails("p", AiErrorKind.Unauthorized);
        var policy = new RetryPolicy(maxAttempts: 4, TimeSpan.FromMilliseconds(1), new Random(1));

        var response = await policy.ExecuteAsync(ct => provider.GenerateAsync(Cfg.Ask(), ct), CancellationToken.None);

        Assert.False(response.Success);
        Assert.Equal(1, provider.Calls);
    }

    [Fact]
    public async Task CoTranSoLanThu()
    {
        var provider = ScriptedAiProvider.AlwaysFails("p", AiErrorKind.Timeout);
        var policy = new RetryPolicy(maxAttempts: 3, TimeSpan.FromMilliseconds(1), new Random(1));

        await policy.ExecuteAsync(ct => provider.GenerateAsync(Cfg.Ask(), ct), CancellationToken.None);

        Assert.Equal(3, provider.Calls);
    }

    [Fact]
    public void KhoangChoTangTheoCapSoNhan()
    {
        var policy = new RetryPolicy(4, TimeSpan.FromMilliseconds(100), new Random(7));
        var error = new AiError(AiErrorKind.Timeout, "thử");

        var first = policy.DelayFor(1, error);
        var third = policy.DelayFor(3, error);

        Assert.True(third > first, $"{third} phải lâu hơn {first}");
        Assert.True(third <= policy.MaxDelay);
    }

    /// <summary>Provider nói chờ bao lâu thì chờ đúng thế — họ biết hạn mức của họ, ta thì đoán.</summary>
    [Fact]
    public void TonTrongRetryAfterCuaProvider()
    {
        var policy = new RetryPolicy(3, TimeSpan.FromMilliseconds(10), new Random(1));
        var error = new AiError(AiErrorKind.RateLimited, "429") { RetryAfter = TimeSpan.FromSeconds(2) };

        Assert.Equal(TimeSpan.FromSeconds(2), policy.DelayFor(1, error));
    }

    [Fact]
    public void KhoangChoCoTran()
    {
        var policy = new RetryPolicy(6, TimeSpan.FromSeconds(10), new Random(1)) { MaxDelay = TimeSpan.FromSeconds(3) };
        var error = new AiError(AiErrorKind.RateLimited, "429") { RetryAfter = TimeSpan.FromHours(1) };

        Assert.Equal(TimeSpan.FromSeconds(3), policy.DelayFor(1, error));
    }
}

public class ProviderHealthTrackerTests
{
    [Fact]
    public void ProviderMoiThiLuonDangThu()
    {
        var tracker = new ProviderHealthTracker(Cfg.Clock());

        Assert.True(tracker.ShouldTry("chua-goi-lan-nao"));
    }

    /// <summary>
    /// Hỏng lẻ một hai lần là nhiễu mạng, không phải provider chết — chưa loại.
    /// Đủ ngưỡng mới loại, vì loại quá sớm sẽ bỏ qua một provider vẫn tốt.
    /// </summary>
    [Fact]
    public void HongDuNguongLienTiepThiTamLoai()
    {
        var clock = Cfg.Clock();
        var tracker = new ProviderHealthTracker(clock) { FailureThreshold = 3 };
        var error = new AiError(AiErrorKind.ProviderError, "500");

        tracker.RecordFailure("p", error);
        tracker.RecordFailure("p", error);
        Assert.True(tracker.ShouldTry("p"));

        tracker.RecordFailure("p", error);
        Assert.False(tracker.ShouldTry("p"));
    }

    [Fact]
    public void HetKhoangNghiThiDuocThuLai()
    {
        var clock = Cfg.Clock();
        var tracker = new ProviderHealthTracker(clock)
        {
            FailureThreshold = 1,
            BaseCooldown = TimeSpan.FromSeconds(30),
        };

        tracker.RecordFailure("p", new AiError(AiErrorKind.ProviderError, "500"));
        Assert.False(tracker.ShouldTry("p"));

        clock.Advance(TimeSpan.FromSeconds(31));
        Assert.True(tracker.ShouldTry("p"));
    }

    /// <summary>Khoá sai thì thử lại vô nghĩa — nghỉ dài nhất ngay từ lần đầu.</summary>
    [Fact]
    public void KhoaSaiThiBiLoaiNgayVoiKhoangNghiDaiNhat()
    {
        var clock = Cfg.Clock();
        var tracker = new ProviderHealthTracker(clock) { MaxCooldown = TimeSpan.FromMinutes(10) };

        tracker.RecordFailure("p", new AiError(AiErrorKind.Unauthorized, "401"));

        Assert.False(tracker.ShouldTry("p"));

        clock.Advance(TimeSpan.FromMinutes(9));
        Assert.False(tracker.ShouldTry("p"));

        clock.Advance(TimeSpan.FromMinutes(2));
        Assert.True(tracker.ShouldTry("p"));
    }

    [Fact]
    public void VuotHanMucThiNghiDungThoiGianProviderYeuCau()
    {
        var clock = Cfg.Clock();
        var tracker = new ProviderHealthTracker(clock);

        tracker.RecordFailure("p", new AiError(AiErrorKind.RateLimited, "429")
        {
            RetryAfter = TimeSpan.FromSeconds(5),
        });

        Assert.False(tracker.ShouldTry("p"));

        clock.Advance(TimeSpan.FromSeconds(6));
        Assert.True(tracker.ShouldTry("p"));
    }

    [Fact]
    public void ThanhCongThiXoaHanKhoangNghiVaChuoiHong()
    {
        var clock = Cfg.Clock();
        var tracker = new ProviderHealthTracker(clock) { FailureThreshold = 1 };

        tracker.RecordFailure("p", new AiError(AiErrorKind.ProviderError, "500"));
        clock.Advance(TimeSpan.FromMinutes(1));

        tracker.RecordSuccess("p", TimeSpan.FromMilliseconds(100));

        var health = tracker.For("p");

        Assert.Null(health.UnavailableUntil);
        Assert.Equal(0, health.ConsecutiveFailures);
        Assert.True(tracker.ShouldTry("p"));
    }

    [Fact]
    public void GhiLaiSoLieuDocDuoc()
    {
        var tracker = new ProviderHealthTracker(Cfg.Clock());

        tracker.RecordSuccess("p", TimeSpan.FromMilliseconds(100));
        tracker.RecordSuccess("p", TimeSpan.FromMilliseconds(300));
        tracker.RecordFailure("p", new AiError(AiErrorKind.Timeout, "hết hạn"));

        var health = tracker.For("p");

        Assert.Equal(2, health.Successes);
        Assert.Equal(1, health.Failures);
        Assert.Equal(1, health.Timeouts);
        Assert.Equal(200, health.AverageLatency.TotalMilliseconds, 1);
        Assert.Equal(2.0 / 3, health.SuccessRate, 3);
    }
}
