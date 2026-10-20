using System.Diagnostics;
using System.Text.Json;
using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.Http;

namespace Ai.Providers.Providers;

/// <summary>
/// PROVIDER THEO HỢP ĐỒNG KIỂU OPENAI — một lớp cho nhiều nhà cung cấp.
///
/// VÌ SAO KHÔNG ĐẶT TÊN THEO MỘT NHÀ CUNG CẤP?
///
/// Vì hợp đồng <c>POST /chat/completions</c> đã thành khuôn dạng chung: rất
/// nhiều dịch vụ (kể cả một số máy chủ tự dựng) nhận đúng khuôn đó. Đặt tên
/// lớp theo một hãng sẽ dẫn tới việc sao lớp này ra thành <c>XProvider</c>,
/// <c>YProvider</c> giống nhau 95% — rồi sửa lỗi ở một bản, quên ba bản còn
/// lại. Ở đây khác biệt giữa các nhà cung cấp nằm trong CẤU HÌNH (endpoint,
/// tên biến môi trường chứa khoá, model), không nằm trong code.
///
/// KHÔNG KHẲNG ĐỊNH NHÀ CUNG CẤP NÀO MIỄN PHÍ. Giá và hạn mức đổi liên tục, và
/// lớp này không có cách nào biết. Nó chỉ biết provider nào chạy TẠI MÁY (không
/// tốn tiền mỗi request) và provider nào gọi ra ngoài — đó là phân biệt duy
/// nhất dùng được để định tuyến.
///
/// KHOÁ API: đọc qua <see cref="SecretResolver"/>, và chỉ đọc lúc gọi. Không
/// giữ trong trường, không đưa vào thông điệp lỗi, không ghi log.
/// </summary>
public sealed class OpenAiCompatibleProvider : IAiProvider
{
    private readonly ProviderOptions _options;
    private readonly SecretResolver _secrets;
    private readonly HttpClient _http;
    private readonly int _maxResponseCharacters;

    public OpenAiCompatibleProvider(ProviderOptions options, SecretResolver secrets,
        HttpClient? http = null, int maxResponseCharacters = 8_000)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(secrets);

        _options = options;
        _secrets = secrets;
        _http = http ?? ProviderHttp.Shared;
        _maxResponseCharacters = maxResponseCharacters;
    }

    public string ProviderName => _options.Name;

    public IReadOnlyList<string> SupportedModels => _options.Models.Count > 0
        ? _options.Models
        : _options.DefaultModel is null ? [] : [_options.DefaultModel];

    /// <summary>
    /// Sẵn sàng = CÓ khoá và endpoint hợp lệ.
    ///
    /// Cố ý KHÔNG gọi thử một lượt sinh chữ để kiểm tra: mỗi lần kiểm tra như
    /// vậy là một lượt bị tính tiền, và màn hình chat gọi kiểm tra mỗi khi mở.
    /// Provider hỏng sẽ lộ ra ở lần gọi thật, và <c>ProviderHealthTracker</c>
    /// tạm loại nó ra — rẻ hơn và đúng hơn.
    ///
    /// Không có khoá thì trả về false NGAY, không gọi mạng: gọi mà không có khoá
    /// chỉ đổi lấy một HTTP 401 và một dòng log vô ích.
    /// </summary>
    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        bool ready = _secrets.Has(_options.ApiKeyEnvironmentVariable)
                     && Uri.TryCreate(_options.Endpoint, UriKind.Absolute, out _);

        return Task.FromResult(ready);
    }

    public async Task<AiResponse> GenerateAsync(AiRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        string? model = request.Model ?? _options.DefaultModel;

        if (string.IsNullOrWhiteSpace(model))
        {
            return AiResponse.Fail(ProviderName, null, new AiError(AiErrorKind.Rejected,
                $"provider '{ProviderName}' chưa cấu hình DefaultModel và yêu cầu cũng không nói model nào"));
        }

        if (!_options.AllowsModel(model))
        {
            return AiResponse.Fail(ProviderName, model, new AiError(AiErrorKind.Rejected,
                $"model '{model}' không nằm trong danh sách cho phép của provider '{ProviderName}'"));
        }

        string? key = _secrets.Resolve(_options.ApiKeyEnvironmentVariable);

        if (string.IsNullOrWhiteSpace(key))
        {
            return AiResponse.Fail(ProviderName, model, new AiError(AiErrorKind.Unauthorized,
                $"chưa đặt biến môi trường '{_options.ApiKeyEnvironmentVariable}'"));
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
            temperature = request.Temperature,
            max_tokens = request.MaxTokens,
            stream = false,
        });

        var stopwatch = Stopwatch.StartNew();

        var outcome = await ProviderHttp
            .PostJsonAsync(_http, Endpoint("chat/completions"), payload, key, _options.Timeout,
                _maxResponseCharacters, cancellationToken)
            .ConfigureAwait(false);

        stopwatch.Stop();

        if (!outcome.Ok)
        {
            return AiResponse.Fail(ProviderName, model, outcome.Error!, stopwatch.Elapsed);
        }

        return Parse(outcome.Body, model, stopwatch.Elapsed);
    }

    private AiResponse Parse(string body, string model, TimeSpan latency)
    {
        try
        {
            using var document = JsonDocument.Parse(body);
            var root = document.RootElement;

            // Một số endpoint trả lỗi kèm HTTP 200 — khuôn dạng thì vẫn là JSON
            // có trường "error". Không xét trường này thì lỗi đó đi tiếp dưới
            // dạng "nội dung rỗng", và thông điệp mất hẳn nguyên nhân thật.
            if (root.TryGetProperty("error", out var error))
            {
                string message = error.TryGetProperty("message", out var text)
                    ? text.GetString() ?? error.ToString()
                    : error.ToString();

                return AiResponse.Fail(ProviderName, model,
                    new AiError(AiErrorKind.ProviderError, Security.OutboundRedactor.Redact(message)), latency);
            }

            if (!root.TryGetProperty("choices", out var choices)
                || choices.ValueKind != JsonValueKind.Array
                || choices.GetArrayLength() == 0)
            {
                return AiResponse.Fail(ProviderName, model,
                    new AiError(AiErrorKind.InvalidResponse, "phản hồi không có 'choices'"), latency);
            }

            var choice = choices[0];

            string content = choice.TryGetProperty("message", out var message2)
                             && message2.TryGetProperty("content", out var contentElement)
                ? contentElement.GetString() ?? ""
                : "";

            if (string.IsNullOrWhiteSpace(content))
            {
                return AiResponse.Fail(ProviderName, model,
                    new AiError(AiErrorKind.InvalidResponse, "nội dung trả về rỗng"), latency);
            }

            var finish = choice.TryGetProperty("finish_reason", out var reason)
                ? reason.GetString() switch
                {
                    "stop" => AiFinishReason.Stop,
                    "length" => AiFinishReason.Length,
                    "tool_calls" => AiFinishReason.ToolCall,
                    "content_filter" => AiFinishReason.ContentFilter,
                    _ => AiFinishReason.Unknown,
                }
                : AiFinishReason.Unknown;

            var usage = root.TryGetProperty("usage", out var usageElement)
                ? new AiUsage(
                    usageElement.TryGetProperty("prompt_tokens", out var input) && input.TryGetInt32(out int i) ? i : null,
                    usageElement.TryGetProperty("completion_tokens", out var output) && output.TryGetInt32(out int o) ? o : null)
                : AiUsage.Unknown;

            string? returnedModel = root.TryGetProperty("model", out var modelElement)
                ? modelElement.GetString()
                : null;

            return AiResponse.Ok(ProviderName, returnedModel ?? model, content.Trim(), latency, usage, finish);
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
