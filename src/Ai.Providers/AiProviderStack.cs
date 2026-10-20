using Ai.Phase16.Memory;
using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.Integration;
using Ai.Providers.Providers;
using Ai.Providers.Resilience;
using Ai.Providers.Routing;
using Ai.Providers.Usage;

namespace Ai.Providers;

/// <summary>
/// Cả lớp provider, dựng sẵn và nối đúng — để chỗ dùng chỉ cần một dòng.
///
/// Giữ luôn <see cref="Problems"/> và <see cref="ConfigSource"/> vì câu hỏi đầu
/// tiên khi LLM không chạy bao giờ cũng là "nó đọc cấu hình ở đâu, và có báo
/// gì không". Không trả lời được câu đó thì mọi việc dò lỗi sau đó là mò mẫm.
/// </summary>
public sealed record AiProviderStack(
    AiProvidersOptions Options,
    IReadOnlyList<IAiProvider> Providers,
    AiProviderRouter Router,
    SecretResolver Secrets,
    string ConfigSource,
    IReadOnlyList<string> Problems)
{
    public ProviderHealthTracker Health => Router.Health;

    public AiUsageTracker Usage => Router.Usage;

    /// <summary>Bộ sinh câu trả lời dùng LLM, cắm được vào đường ống Phase 14.</summary>
    public LlmAnswerGenerator CreateAnswerGenerator(LlmComposeOptions? options = null) =>
        new(Router, options: options);

    /// <summary>Mô tả an toàn để in ra màn hình: có khoá hay chưa, KHÔNG có giá trị khoá.</summary>
    public IReadOnlyList<string> Describe() =>
        [.. Options.Providers.Select(p =>
            $"{p.Name,-14} {p.Type,-18} {(p.Enabled ? "bật" : "tắt")}  " +
            $"model {p.DefaultModel ?? "(tự chọn)"}  {Secrets.Describe(p.ApiKeyEnvironmentVariable)}")];
}

/// <summary>
/// Dựng một provider cho loại mà lớp này KHÔNG tự dựng được.
///
/// Trả về null nghĩa là "tôi không nhận loại này" — người gọi sẽ thử cách khác
/// rồi mới báo là không hỗ trợ.
/// </summary>
public delegate IAiProvider? ProviderCreator(ProviderOptions options, SecretResolver secrets, int maxResponseCharacters);

/// <summary>
/// DỰNG LỚP PROVIDER TỪ CẤU HÌNH.
///
/// Một <see cref="ProviderKind"/> ứng với một lớp, và đó là toàn bộ chỗ cần sửa
/// khi thêm nhà cung cấp mới có hợp đồng khác (ví dụ Gemini). Nhà cung cấp nào
/// dùng hợp đồng kiểu OpenAI thì KHÔNG cần sửa code gì cả — chỉ thêm một mục
/// vào file cấu hình.
///
/// NHÀ CUNG CẤP CẦN GÓI NGOÀI thì không dựng được ở đây, và đó là chủ ý: lớp
/// này nằm trong project KHÔNG có PackageReference nào. Những loại đó được
/// dựng qua <see cref="ProviderCreator"/> truyền vào — ví dụ
/// <c>ProviderKind.OpenAiResponses</c> do project <c>Ai.Providers.OpenAi</c>
/// dựng.
///
/// VÌ SAO TRUYỀN VÀO CHỨ KHÔNG ĐĂNG KÝ TĨNH? Vì một bảng tĩnh toàn cục sẽ khiến
/// thứ tự nạp assembly quyết định provider nào tồn tại — và trong test thì thứ
/// tự đó không đoán được. Truyền tường minh thì chỗ gọi nào dựng gì là đọc ra
/// được ngay tại chỗ gọi.
/// </summary>
public static class AiProviderFactory
{
    /// <summary>Dựng toàn bộ lớp provider: đọc cấu hình, dựng provider, nối router.</summary>
    public static AiProviderStack Build(
        string? configPath = null,
        SecretResolver? secrets = null,
        HttpClient? http = null,
        IClock? clock = null,
        ProviderCreator? extraCreator = null)
    {
        var loaded = AiProvidersFile.Load(configPath);

        return Build(loaded.Options, secrets, http, clock, loaded.Source, loaded.Problems, extraCreator);
    }

    /// <summary>Dựng từ một bộ cấu hình có sẵn — dùng cho test và cho nơi tự nạp cấu hình.</summary>
    public static AiProviderStack Build(
        AiProvidersOptions options,
        SecretResolver? secrets = null,
        HttpClient? http = null,
        IClock? clock = null,
        string configSource = "cấu hình truyền vào",
        IReadOnlyList<string>? problems = null,
        ProviderCreator? extraCreator = null)
    {
        ArgumentNullException.ThrowIfNull(options);

        secrets ??= new SecretResolver();

        var allProblems = new List<string>(problems ?? []);
        allProblems.AddRange(AiProvidersFile.Validate(options).Where(p => !allProblems.Contains(p)));

        var providers = new List<IAiProvider>();

        foreach (var provider in options.EnabledProviders)
        {
            // Bộ dựng ngoài được thử TRƯỚC, nên nó cũng thay thế được một loại
            // sẵn có — hữu ích khi muốn cắm bản giả vào trong test.
            var created = extraCreator?.Invoke(provider, secrets, options.MaxResponseCharacters)
                          ?? Create(provider, secrets, http, options.MaxResponseCharacters);

            if (created is null)
            {
                allProblems.Add(provider.Type == ProviderKind.OpenAiResponses
                    ? $"{provider.Name}: loại 'OpenAiResponses' cần project Ai.Providers.OpenAi — " +
                      "truyền OpenAiProviders.Create vào AiProviderFactory.Build"
                    : $"{provider.Name}: loại '{provider.Type}' chưa được hỗ trợ");

                continue;
            }

            providers.Add(created);
        }

        var health = new ProviderHealthTracker(clock);
        var usage = new AiUsageTracker(clock);
        var router = new AiProviderRouter(options, providers, health, usage);

        return new AiProviderStack(options, providers, router, secrets, configSource, allProblems);
    }

    /// <summary>Dựng một provider. Trả về null nếu loại đó chưa có lớp nào hiện thực.</summary>
    public static IAiProvider? Create(ProviderOptions options, SecretResolver secrets,
        HttpClient? http = null, int maxResponseCharacters = 8_000) => options.Type switch
    {
        ProviderKind.Ollama => new OllamaProvider(options, http, maxResponseCharacters),
        ProviderKind.OpenAiCompatible => new OpenAiCompatibleProvider(options, secrets, http, maxResponseCharacters),

        // Provider kịch bản không dựng từ cấu hình được: kịch bản là code, và
        // một file cấu hình có thể mô tả "trả lời gì" thì nó thành một cách
        // nhét câu trả lời giả vào hệ thống từ bên ngoài.
        ProviderKind.Scripted => null,

        // Cần SDK ngoài -> phải do ProviderCreator truyền vào dựng.
        ProviderKind.OpenAiResponses => null,

        _ => null,
    };
}
