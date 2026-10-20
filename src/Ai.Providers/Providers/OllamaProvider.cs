using System.Diagnostics;
using System.Text.Json;
using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.Http;

namespace Ai.Providers.Providers;

/// <summary>
/// OLLAMA — model nguồn mở chạy ngay trên máy.
///
/// Đây là provider được ưu tiên trong chiến lược mặc định, và lý do không phải
/// chất lượng: nó không tốn tiền mỗi lượt gọi, không có hạn mức, và KHÔNG GỬI
/// dữ liệu người dùng ra khỏi máy. Với khâu mà lớp này đảm nhiệm — diễn đạt
/// lại những căn cứ đã có — ba điều đó quan trọng hơn việc model thông minh
/// đến đâu.
///
/// KHÔNG GHI CỨNG TÊN MODEL. <see cref="ProviderOptions.DefaultModel"/> để
/// trống thì provider lấy model đầu tiên máy đang có (qua <c>/api/tags</c>).
/// Ghi cứng một tên (kiểu "llama3") sẽ làm cấu hình sai trên mọi máy chưa tải
/// đúng model đó, và thông điệp lỗi thì lại nói về model — chứ không nói rằng
/// chính cấu hình mới là chỗ sai.
///
/// Endpoint mặc định là <c>http://localhost:11434</c>. HTTP trần ở đây là hợp
/// lệ vì gói tin không ra khỏi máy; <c>AiProvidersFile.Validate</c> chỉ báo lỗi
/// http:// khi endpoint trỏ ra ngoài.
/// </summary>
public sealed class OllamaProvider : IAiProvider
{
    private readonly ProviderOptions _options;
    private readonly HttpClient _http;
    private readonly int _maxResponseCharacters;
    private List<string>? _cachedModels;

    public OllamaProvider(ProviderOptions options, HttpClient? http = null, int maxResponseCharacters = 8_000)
    {
        ArgumentNullException.ThrowIfNull(options);

        _options = options;
        _http = http ?? ProviderHttp.Shared;
        _maxResponseCharacters = maxResponseCharacters;
    }

    public string ProviderName => _options.Name;

    /// <summary>
    /// Model máy đang có. Rỗng cho tới lần đầu gọi <see cref="IsAvailableAsync"/>
    /// — danh sách này do máy quyết định, không phải do cấu hình, nên không thể
    /// biết trước khi hỏi.
    /// </summary>
    public IReadOnlyList<string> SupportedModels =>
        _cachedModels ?? (_options.DefaultModel is null ? [] : [_options.DefaultModel]);

    public async Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        var models = await ListModelsAsync(cancellationToken).ConfigureAwait(false);

        if (models.Count == 0) return false;

        // Cấu hình chỉ định một model cụ thể thì model đó phải CÓ THẬT trên máy.
        // Không kiểm chỗ này thì lỗi chỉ lộ ra ở lần gọi thật, dưới dạng một mã
        // 404 trông như provider hỏng.
        return _options.DefaultModel is null
               || models.Any(m => m.Equals(_options.DefaultModel, StringComparison.OrdinalIgnoreCase)
                                  || m.StartsWith(_options.DefaultModel + ":", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>Danh sách model máy đang có. Lỗi mạng = danh sách rỗng, không ném.</summary>
    public async Task<IReadOnlyList<string>> ListModelsAsync(CancellationToken cancellationToken = default)
    {
        var outcome = await ProviderHttp
            .GetAsync(_http, Endpoint("api/tags"), null, TimeSpan.FromSeconds(3), _maxResponseCharacters,
                cancellationToken)
            .ConfigureAwait(false);

        if (!outcome.Ok) return [];

        try
        {
            using var document = JsonDocument.Parse(outcome.Body);

            var models = document.RootElement.TryGetProperty("models", out var array)
                         && array.ValueKind == JsonValueKind.Array
                ? array.EnumerateArray()
                    .Select(m => m.TryGetProperty("name", out var name) ? name.GetString() : null)
                    .Where(n => !string.IsNullOrWhiteSpace(n))
                    .Select(n => n!)
                    .ToList()
                : [];

            _cachedModels = models;
            return models;
        }
        catch (JsonException)
        {
            return [];
        }
    }

    public async Task<AiResponse> GenerateAsync(AiRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        string? model = await ResolveModelAsync(request.Model, cancellationToken).ConfigureAwait(false);

        if (model is null)
        {
            return AiResponse.Fail(ProviderName, null, new AiError(AiErrorKind.ModelUnavailable,
                "máy chưa có model nào — hãy chạy 'ollama pull <model>' rồi thử lại"));
        }

        if (!_options.AllowsModel(model))
        {
            return AiResponse.Fail(ProviderName, model, new AiError(AiErrorKind.Rejected,
                $"model '{model}' không nằm trong danh sách cho phép của provider '{ProviderName}'"));
        }

        string payload = JsonSerializer.Serialize(new
        {
            model,
            messages = request.BuildMessages().Select(m => new
            {
                role = m.Role switch
                {
                    ChatRole.System => "system",
                    ChatRole.Assistant => "assistant",
                    ChatRole.Tool => "tool",
                    _ => "user",
                },
                content = m.Content,
            }),
            stream = false,
            options = new
            {
                temperature = request.Temperature,
                num_predict = request.MaxTokens,
            },
        });

        var stopwatch = Stopwatch.StartNew();

        var outcome = await ProviderHttp
            .PostJsonAsync(_http, Endpoint("api/chat"), payload, null, _options.Timeout,
                _maxResponseCharacters, cancellationToken)
            .ConfigureAwait(false);

        stopwatch.Stop();

        if (!outcome.Ok)
        {
            return AiResponse.Fail(ProviderName, model, outcome.Error!, stopwatch.Elapsed);
        }

        return Parse(outcome.Body, model, stopwatch.Elapsed);
    }

    /// <summary>Chọn model: yêu cầu → cấu hình → model đầu tiên máy có.</summary>
    private async Task<string?> ResolveModelAsync(string? requested, CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(requested)) return requested;
        if (!string.IsNullOrWhiteSpace(_options.DefaultModel)) return _options.DefaultModel;

        var models = _cachedModels ?? await ListModelsAsync(cancellationToken).ConfigureAwait(false);
        return models.Count > 0 ? models[0] : null;
    }

    private AiResponse Parse(string body, string model, TimeSpan latency)
    {
        try
        {
            using var document = JsonDocument.Parse(body);
            var root = document.RootElement;

            string content = root.TryGetProperty("message", out var message)
                             && message.TryGetProperty("content", out var text)
                ? text.GetString() ?? ""
                : "";

            if (string.IsNullOrWhiteSpace(content))
            {
                return AiResponse.Fail(ProviderName, model,
                    new AiError(AiErrorKind.InvalidResponse, "Ollama trả về nội dung rỗng"), latency);
            }

            var usage = new AiUsage(
                root.TryGetProperty("prompt_eval_count", out var input) && input.TryGetInt32(out int i) ? i : null,
                root.TryGetProperty("eval_count", out var output) && output.TryGetInt32(out int o) ? o : null);

            var finish = root.TryGetProperty("done_reason", out var reason)
                ? reason.GetString() switch
                {
                    "stop" => AiFinishReason.Stop,
                    "length" => AiFinishReason.Length,
                    _ => AiFinishReason.Unknown,
                }
                : AiFinishReason.Stop;

            return AiResponse.Ok(ProviderName, model, content.Trim(), latency, usage, finish);
        }
        catch (JsonException ex)
        {
            return AiResponse.Fail(ProviderName, model,
                new AiError(AiErrorKind.InvalidResponse, $"không đọc được JSON: {ex.Message}"), latency);
        }
    }

    private Uri Endpoint(string path) =>
        new(new Uri(_options.Endpoint.TrimEnd('/') + "/"), path);
}
