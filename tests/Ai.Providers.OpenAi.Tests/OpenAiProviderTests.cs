using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.OpenAi;

namespace Ai.Providers.OpenAi.Tests;

/// <summary>
/// KIỂM PROVIDER OPENAI MÀ KHÔNG GỌI MẠNG.
///
/// KHÔNG có test nào ở đây chạm tới api.openai.com. Mọi thứ kiểm được ở mức
/// này — cấu hình thiếu, khoá thiếu, model không được phép, dịch lỗi — đều xảy
/// ra TRƯỚC lượt gọi HTTP đầu tiên, nên kiểm chúng không cần khoá và không tốn
/// tiền. Test cần khoá thật nằm riêng (xem OpenAiLiveTests) và mặc định bị bỏ
/// qua.
/// </summary>
public class OpenAiProviderTests
{
    private static ProviderOptions Options(
        string? defaultModel = "model-cau-hinh",
        string? keyVariable = "TEST_OPENAI_KEY",
        AiModelMap? taskModels = null) =>
        new()
        {
            Name = "openai",
            Type = ProviderKind.OpenAiResponses,
            Endpoint = "https://api.openai.com/v1",
            ApiKeyEnvironmentVariable = keyVariable,
            DefaultModel = defaultModel,
            TaskModels = taskModels ?? new AiModelMap(),
        };

    private static SecretResolver WithKey(string? value) =>
        new(injected: value is null
            ? new Dictionary<string, string>()
            : new Dictionary<string, string> { ["TEST_OPENAI_KEY"] = value });

    /// <summary>Không có khoá thì KHÔNG sẵn sàng — và không gọi mạng để biết điều đó.</summary>
    [Fact]
    public async Task ChuaDatKhoaThiKhongSanSang()
    {
        var provider = new OpenAiResponsesProvider(Options(), WithKey(null));

        Assert.False(await provider.IsAvailableAsync());
    }

    [Fact]
    public async Task CoKhoaThiSanSang()
    {
        var provider = new OpenAiResponsesProvider(Options(), WithKey("sk-test-khong-that"));

        Assert.True(await provider.IsAvailableAsync());
    }

    /// <summary>
    /// Thiếu khoá -> lỗi Unauthorized, và Unauthorized KHÔNG được thử lại.
    /// Thử lại một lỗi sai khoá là thử lại một thứ không thể khác đi.
    /// </summary>
    [Fact]
    public async Task ThieuKhoaTraVeUnauthorizedVaKhongThuLai()
    {
        var provider = new OpenAiResponsesProvider(Options(), WithKey(null));

        var response = await provider.GenerateAsync(new AiRequest { UserPrompt = "xin chào" });

        Assert.False(response.Success);
        Assert.Equal(AiErrorKind.Unauthorized, response.Error!.Kind);
        Assert.False(response.Error.IsRetryable);
    }

    /// <summary>Chưa khai model nào thì nói rõ là thiếu cấu hình, không đoán một tên model.</summary>
    [Fact]
    public async Task ChuaKhaiModelThiBaoThieuCauHinh()
    {
        var provider = new OpenAiResponsesProvider(Options(defaultModel: null), WithKey("sk-test"));

        var response = await provider.GenerateAsync(new AiRequest { UserPrompt = "xin chào" });

        Assert.False(response.Success);
        Assert.Equal(AiErrorKind.Rejected, response.Error!.Kind);
        Assert.Contains("chưa cấu hình model", response.Error.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// Danh sách cho phép chặn model lạ TRƯỚC khi gọi mạng.
    ///
    /// Đây là chốt chi phí: một chỗ gọi lỡ yêu cầu model đắt hơn thì bị chặn
    /// tại chỗ, chứ không phải sau khi hoá đơn đã phát sinh.
    /// </summary>
    [Fact]
    public async Task ModelNgoaiDanhSachChoPhepBiChanTruocKhiGoiMang()
    {
        var options = Options() with { Models = ["model-re"] };
        var provider = new OpenAiResponsesProvider(options, WithKey("sk-test"));

        var response = await provider.GenerateAsync(new AiRequest
        {
            UserPrompt = "xin chào",
            Model = "model-rat-dat",
        });

        Assert.False(response.Success);
        Assert.Equal(AiErrorKind.Rejected, response.Error!.Kind);
    }

    /// <summary>Model theo loại việc được dùng khi yêu cầu không nói model nào.</summary>
    [Fact]
    public async Task ModelTheoLoaiViecDuocDungKhiKhongChiDinh()
    {
        var options = Options(defaultModel: null, taskModels: new AiModelMap
        {
            Chat = "model-chat",
            Coding = "model-code",
        });

        var provider = new OpenAiResponsesProvider(options, WithKey(null));

        // Khoá thiếu nên nó dừng ở bước khoá — nhưng bước CHỌN MODEL đã chạy
        // xong trước đó, và model được chọn hiện trong phản hồi.
        var response = await provider.GenerateAsync(new AiRequest
        {
            UserPrompt = "viết hàm C#",
            TaskType = AiTaskType.Coding,
        });

        Assert.Equal("model-code", response.Model);
    }

    /// <summary>SupportedModels gom được cả danh sách cho phép lẫn model theo loại việc.</summary>
    [Fact]
    public void SupportedModelsLayTuTaskModelsKhiKhongCoDanhSachChoPhep()
    {
        var options = Options(defaultModel: null, taskModels: new AiModelMap
        {
            Chat = "a",
            Reasoning = "b",
            Embedding = "c",
        });

        var provider = new OpenAiResponsesProvider(options, WithKey("sk-test"));

        Assert.Equal(["a", "b", "c"], provider.SupportedModels);
    }

    /// <summary>Provider dùng đúng tên trong cấu hình, vì đó là khoá tra cứu của bộ định tuyến.</summary>
    [Fact]
    public void TenProviderLayTuCauHinh()
    {
        var provider = new OpenAiResponsesProvider(Options() with { Name = "openai-chinh" }, WithKey("sk-test"));

        Assert.Equal("openai-chinh", provider.ProviderName);
    }
}

/// <summary>
/// KIỂM PHẦN NHÚNG VECTOR — cũng không gọi mạng.
/// </summary>
public class OpenAiEmbeddingTests
{
    private static ProviderOptions Options(string? embeddingModel) => new()
    {
        Name = "openai",
        Type = ProviderKind.OpenAiResponses,
        Endpoint = "https://api.openai.com/v1",
        ApiKeyEnvironmentVariable = "TEST_OPENAI_KEY",
        TaskModels = new AiModelMap { Embedding = embeddingModel },
    };

    private static SecretResolver Key(string? value) =>
        new(injected: value is null
            ? new Dictionary<string, string>()
            : new Dictionary<string, string> { ["TEST_OPENAI_KEY"] = value });

    [Fact]
    public async Task ChuaKhaiModelNhungThiBaoThieuCauHinh()
    {
        var provider = new OpenAiEmbeddingProvider(Options(null), Key("sk-test"));

        var result = await provider.EmbedAsync([new EmbeddingInput("d1", "c1", "nội dung")]);

        Assert.False(result.Success);
        Assert.Equal(AiErrorKind.Rejected, result.Error!.Kind);
    }

    /// <summary>Danh sách rỗng KHÔNG phải lỗi — và cũng không được gọi mạng.</summary>
    [Fact]
    public async Task DanhSachRongTraVeThanhCongVaKhongGoiMang()
    {
        var provider = new OpenAiEmbeddingProvider(Options("model-nhung"), Key(null));

        var result = await provider.EmbedAsync([]);

        Assert.True(result.Success);
        Assert.Empty(result.Vectors);
    }

    /// <summary>
    /// Mẩu rỗng bị loại TRƯỚC khi gọi. Một chuỗi trắng lọt vào sẽ làm nhà cung
    /// cấp từ chối cả lô, và lúc đó mọi mẩu hợp lệ cũng mất theo.
    /// </summary>
    [Fact]
    public async Task MoiMauDeuRongThiBiTuChoiTruocKhiGoiMang()
    {
        var provider = new OpenAiEmbeddingProvider(Options("model-nhung"), Key("sk-test"));

        var result = await provider.EmbedAsync(
        [
            new EmbeddingInput("d1", "c1", "   "),
            new EmbeddingInput("d1", "c2", ""),
        ]);

        Assert.False(result.Success);
        Assert.Equal(AiErrorKind.Rejected, result.Error!.Kind);
    }

    /// <summary>Chưa gọi lần nào thì số chiều là 0 — nói "chưa biết", không đoán 1536.</summary>
    [Fact]
    public void SoChieuChuaBietTruocLanGoiDauTien()
    {
        var provider = new OpenAiEmbeddingProvider(Options("model-nhung"), Key("sk-test"));

        Assert.Equal(0, provider.Dimensions);
    }
}

/// <summary>Điểm vào của project: chỉ nhận đúng loại của nó, trả null cho loại khác.</summary>
public class OpenAiProvidersFactoryTests
{
    [Fact]
    public void ChiDungProviderDungLoai()
    {
        var secrets = new SecretResolver();

        var openAi = new ProviderOptions
        {
            Name = "openai",
            Type = ProviderKind.OpenAiResponses,
            Endpoint = "https://api.openai.com/v1",
            ApiKeyEnvironmentVariable = "K",
        };

        var ollama = new ProviderOptions
        {
            Name = "ollama",
            Type = ProviderKind.Ollama,
            Endpoint = "http://localhost:11434",
        };

        Assert.NotNull(OpenAiProviders.Create(openAi, secrets, 8_000));
        Assert.Null(OpenAiProviders.Create(ollama, secrets, 8_000));
    }

    /// <summary>Không khai model nhúng thì KHÔNG dựng bộ nhúng — thà không có còn hơn có mà hỏng.</summary>
    [Fact]
    public void KhongKhaiModelNhungThiKhongDungBoNhung()
    {
        var secrets = new SecretResolver();

        var withoutEmbedding = new ProviderOptions
        {
            Name = "openai",
            Type = ProviderKind.OpenAiResponses,
            ApiKeyEnvironmentVariable = "K",
        };

        var withEmbedding = withoutEmbedding with
        {
            TaskModels = new AiModelMap { Embedding = "model-nhung" },
        };

        Assert.Null(OpenAiProviders.CreateEmbedding(withoutEmbedding, secrets));
        Assert.NotNull(OpenAiProviders.CreateEmbedding(withEmbedding, secrets));
    }

    /// <summary>
    /// Bộ dựng ngoài phải được nối vào nhà máy, nếu không loại OpenAiResponses
    /// sẽ lặng lẽ biến mất khỏi danh sách provider.
    /// </summary>
    [Fact]
    public void NhaMayDungDuocProviderOpenAiKhiDuocTruyenBoDungNgoai()
    {
        var options = new AiProvidersOptions
        {
            Providers =
            [
                new ProviderOptions
                {
                    Name = "openai",
                    Type = ProviderKind.OpenAiResponses,
                    Endpoint = "https://api.openai.com/v1",
                    ApiKeyEnvironmentVariable = "TEST_OPENAI_KEY",
                    DefaultModel = "model-cau-hinh",
                },
            ],
        };

        var without = AiProviderFactory.Build(options);
        var with = AiProviderFactory.Build(options, extraCreator: OpenAiProviders.Create);

        Assert.Empty(without.Providers);
        Assert.Contains(without.Problems, p => p.Contains("Ai.Providers.OpenAi", StringComparison.Ordinal));

        Assert.Single(with.Providers);
        Assert.Equal("openai", with.Providers[0].ProviderName);
    }
}
