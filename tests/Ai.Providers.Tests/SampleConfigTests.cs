using Ai.Providers.Configuration;
using Ai.Providers.Contracts;

namespace Ai.Providers.Tests;

/// <summary>
/// KIỂM CHÍNH FILE MẪU — vì file mẫu là thứ người ta sao chép.
///
/// Một file mẫu sai cú pháp, hoặc gõ sai tên trường, sẽ không làm hỏng build
/// và không làm đỏ test nào — nó chỉ làm người đầu tiên làm theo nó mất một
/// buổi chiều. Những test này biến nó thành một lỗi build.
/// </summary>
public class SampleConfigTests
{
    private static string SamplePath =>
        Path.Combine(AppContext.BaseDirectory, "aiproviders.sample.json");

    private static ConfigLoadResult Load() =>
        AiProvidersFile.Parse(File.ReadAllText(SamplePath), "file mẫu");

    [Fact]
    public void FileMauCoTonTaiCanhFileChay() => Assert.True(File.Exists(SamplePath));

    /// <summary>File mẫu phải đọc được và KHÔNG có vấn đề nào.</summary>
    [Fact]
    public void FileMauDocDuocVaKhongCoVanDe()
    {
        var loaded = Load();

        Assert.False(loaded.HasProblems,
            "file mẫu có vấn đề: " + string.Join("; ", loaded.Problems));
    }

    /// <summary>
    /// File mẫu KHÔNG được chứa thứ trông như khoá thật — kể cả khoá giả đặt
    /// làm ví dụ, vì bộ nạp sẽ từ chối cả file và người dùng sẽ không hiểu vì sao.
    /// </summary>
    [Fact]
    public void FileMauKhongChuaGiTrongNhuKhoaThat() =>
        Assert.Empty(AiProvidersFile.RejectLiteralSecrets(File.ReadAllText(SamplePath)));

    /// <summary>Mẫu phải khai đủ ba loại provider để người đọc thấy được lựa chọn.</summary>
    [Fact]
    public void FileMauKhaiDuBaLoaiProvider()
    {
        var kinds = Load().Options.Providers.Select(p => p.Type).ToList();

        Assert.Contains(ProviderKind.Ollama, kinds);
        Assert.Contains(ProviderKind.OpenAiCompatible, kinds);
        Assert.Contains(ProviderKind.OpenAiResponses, kinds);
    }

    /// <summary>
    /// Mọi provider gọi RA NGOÀI phải mặc định TẮT trong file mẫu.
    ///
    /// Sao file mẫu vào rồi chạy KHÔNG được phép gửi một byte nào ra Internet
    /// cho tới khi có người bật tường minh.
    /// </summary>
    [Fact]
    public void ProviderGoiRaNgoaiMacDinhTatTrongFileMau()
    {
        foreach (var provider in Load().Options.Providers.Where(p => !p.IsLocal))
        {
            Assert.False(provider.Enabled, $"{provider.Name} đáng lẽ phải tắt trong file mẫu");
        }
    }

    /// <summary>Trường mới của loại Responses phải ĐỌC RA ĐÚNG, không lặng lẽ thành mặc định.</summary>
    [Fact]
    public void DocDungTruongMoiCuaLoaiResponses()
    {
        var openAi = Load().Options.Find("openai");

        Assert.NotNull(openAi);
        Assert.Equal("OPENAI_API_KEY", openAi.ApiKeyEnvironmentVariable);
        Assert.Equal(1500, openAi.MaxOutputTokens);
        Assert.Equal(4, openAi.MaxToolCalls);
        Assert.False(openAi.EnableWebSearch);
        Assert.False(openAi.AllowProviderSideStorage);
        Assert.NotNull(openAi.TaskModels);
    }

    /// <summary>Trần token ra phải CẮT giá trị chỗ gọi xin, không phải chỉ nằm đó.</summary>
    [Fact]
    public void TranTokenRaCatGiaTriChoGoiXin()
    {
        var openAi = Load().Options.Find("openai")!;

        Assert.Equal(1500, openAi.CapOutputTokens(4_000));
        Assert.Equal(500, openAi.CapOutputTokens(500));
    }
}
