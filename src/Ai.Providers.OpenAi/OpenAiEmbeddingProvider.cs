using System.ClientModel;
using System.Diagnostics;
using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.Security;
using OpenAI;
using OpenAI.Embeddings;

namespace Ai.Providers.OpenAi;

/// <summary>
/// NHÚNG VECTOR QUA OPENAI — hiện thực <see cref="IEmbeddingProvider"/>.
///
/// Lớp riêng, KHÔNG gộp vào <see cref="OpenAiResponsesProvider"/>, và lý do
/// giống hệt lý do <see cref="IEmbeddingProvider"/> tách khỏi
/// <see cref="IAiProvider"/>: hai năng lực khác nhau, hai hợp đồng khác nhau,
/// hai model khác nhau, và phần lớn nhà cung cấp làm được cái này thì không làm
/// được cái kia. Gộp lại là tạo ra một lớp mà nửa số phương thức ném
/// <c>NotSupportedException</c>.
///
/// GHI NHỚ QUAN TRỌNG VỀ SỐ CHIỀU: <see cref="Dimensions"/> chỉ biết được SAU
/// lần gọi đầu, vì nó phụ thuộc model. Trả về 0 trước đó là nói thật; đoán một
/// con số quen thuộc (1536) sẽ đúng với model này và sai với model khác, mà
/// chỗ dùng thì không có cách nào biết nó đã bị đoán.
/// </summary>
public sealed class OpenAiEmbeddingProvider : IEmbeddingProvider
{
    private readonly ProviderOptions _options;
    private readonly SecretResolver _secrets;
    private readonly Func<string, EmbeddingClient>? _clientFactory;

    public OpenAiEmbeddingProvider(
        ProviderOptions options,
        SecretResolver secrets,
        Func<string, EmbeddingClient>? clientFactory = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(secrets);

        _options = options;
        _secrets = secrets;
        _clientFactory = clientFactory;
    }

    public string ProviderName => _options.Name;

    public int Dimensions { get; private set; }

    /// <summary>Model nhúng đã cấu hình. Null nghĩa là chưa khai — và lúc đó lớp này không chạy.</summary>
    public string? Model => _options.TaskModels.Embedding;

    public async Task<EmbeddingResult> EmbedAsync(IReadOnlyList<EmbeddingInput> inputs,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);

        if (inputs.Count == 0)
        {
            return EmbeddingResult.Ok(ProviderName, Model ?? "", [], TimeSpan.Zero);
        }

        if (string.IsNullOrWhiteSpace(Model))
        {
            return EmbeddingResult.Fail(ProviderName, null, new AiError(AiErrorKind.Rejected,
                $"provider '{ProviderName}' chưa khai TaskModels.Embedding"));
        }

        string? key = _secrets.Resolve(_options.ApiKeyEnvironmentVariable);

        if (string.IsNullOrWhiteSpace(key))
        {
            return EmbeddingResult.Fail(ProviderName, Model, new AiError(AiErrorKind.Unauthorized,
                $"chưa đặt biến môi trường '{_options.ApiKeyEnvironmentVariable}'"));
        }

        // Mẩu rỗng bị loại TRƯỚC khi gọi: nhà cung cấp từ chối cả lô vì một mẩu
        // rỗng, nên một chuỗi trắng lọt vào sẽ làm hỏng toàn bộ lượt nhúng.
        var usable = inputs.Where(i => !string.IsNullOrWhiteSpace(i.Text)).ToList();

        if (usable.Count == 0)
        {
            return EmbeddingResult.Fail(ProviderName, Model,
                new AiError(AiErrorKind.Rejected, "mọi mẩu văn bản đều rỗng"));
        }

        var stopwatch = Stopwatch.StartNew();

        try
        {
            var client = CreateClient(key, Model!);

            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(_options.Timeout);

            var result = await client
                .GenerateEmbeddingsAsync(usable.Select(i => i.Text).ToList(), cancellationToken: timeout.Token)
                .ConfigureAwait(false);

            stopwatch.Stop();

            var collection = result.Value;
            var vectors = new List<EmbeddingVector>(usable.Count);
            var now = DateTimeOffset.UtcNow;

            for (int i = 0; i < collection.Count && i < usable.Count; i++)
            {
                var values = collection[i].ToFloats().ToArray();
                var input = usable[i];

                vectors.Add(new EmbeddingVector
                {
                    DocumentId = input.DocumentId,
                    ChunkId = input.ChunkId,
                    Values = values,
                    Model = Model!,
                    Source = input.Source,
                    Metadata = input.Metadata,
                    CreatedAt = now,
                });
            }

            Dimensions = vectors.Count > 0 ? vectors[0].Dimensions : Dimensions;

            var usage = new AiUsage(collection.Usage?.InputTokenCount, 0);

            return EmbeddingResult.Ok(ProviderName, Model!, vectors, stopwatch.Elapsed, usage);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return EmbeddingResult.Fail(ProviderName, Model,
                new AiError(AiErrorKind.Timeout, $"quá {_options.Timeout.TotalSeconds:F0}s không có phản hồi"));
        }
        catch (ClientResultException ex)
        {
            return EmbeddingResult.Fail(ProviderName, Model, OpenAiErrors.FromStatus(ex));
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or TimeoutException)
        {
            return EmbeddingResult.Fail(ProviderName, Model,
                new AiError(AiErrorKind.Network, OutboundRedactor.Redact(ex.Message)));
        }
    }

    private EmbeddingClient CreateClient(string apiKey, string model)
    {
        if (_clientFactory is not null) return _clientFactory(model);

        var credential = new ApiKeyCredential(apiKey);

        if (string.IsNullOrWhiteSpace(_options.Endpoint))
        {
            return new EmbeddingClient(model, credential);
        }

        return new EmbeddingClient(model, credential, new OpenAIClientOptions
        {
            Endpoint = new Uri(_options.Endpoint),
        });
    }
}
