using Ai.Providers.Configuration;
using Ai.Providers.Contracts;

namespace Ai.Providers.OpenAi;

/// <summary>
/// ĐIỂM VÀO của project này — một hàm, đúng khuôn <see cref="ProviderCreator"/>.
///
/// Cách dùng ở nơi dựng ứng dụng:
///
///     var stack = AiProviderFactory.Build(extraCreator: OpenAiProviders.Create);
///
/// Đó là TOÀN BỘ chỗ phần còn lại của MiniAI biết rằng OpenAI tồn tại. Không
/// có using nào của SDK ngoài thư mục này, và bỏ dòng trên đi thì hệ thống
/// quay về đúng hành vi trước khi có nó — Ollama tại máy và endpoint kiểu
/// OpenAI, như cũ.
/// </summary>
public static class OpenAiProviders
{
    /// <summary>Dựng provider OpenAI nếu cấu hình nói loại này; null cho mọi loại khác.</summary>
    public static IAiProvider? Create(ProviderOptions options, SecretResolver secrets, int maxResponseCharacters)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(secrets);

        return options.Type == ProviderKind.OpenAiResponses
            ? new OpenAiResponsesProvider(options, secrets, maxResponseCharacters)
            : null;
    }

    /// <summary>
    /// Bộ nhúng vector cho một provider đã cấu hình, hoặc null nếu nó không
    /// khai model nhúng.
    ///
    /// Tách khỏi <see cref="Create"/> vì nhúng là một NĂNG LỰC KHÁC: một cấu
    /// hình có thể chỉ dùng OpenAI để sinh chữ và dùng vector cục bộ của
    /// Phase 5, hoặc ngược lại. Ép cả hai vào một lượt dựng sẽ làm lựa chọn đó
    /// biến mất.
    /// </summary>
    public static IEmbeddingProvider? CreateEmbedding(ProviderOptions options, SecretResolver secrets)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(secrets);

        if (options.Type != ProviderKind.OpenAiResponses) return null;

        return string.IsNullOrWhiteSpace(options.TaskModels.Embedding)
            ? null
            : new OpenAiEmbeddingProvider(options, secrets);
    }

    /// <summary>Bộ nhúng đầu tiên dựng được từ cả cấu hình, hoặc null nếu không có.</summary>
    public static IEmbeddingProvider? FindEmbedding(AiProvidersOptions options, SecretResolver secrets)
    {
        ArgumentNullException.ThrowIfNull(options);

        return options.EnabledProviders
            .Select(p => CreateEmbedding(p, secrets))
            .FirstOrDefault(e => e is not null);
    }
}
