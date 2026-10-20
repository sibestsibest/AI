using System.ClientModel;
using System.Diagnostics;
using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.Security;
using OpenAI.Responses;

namespace Ai.Providers.OpenAi;

/// <summary>
/// PROVIDER OPENAI QUA HỢP ĐỒNG RESPONSES — dùng SDK chính thức.
///
/// ============================================================
/// TOÀN BỘ CHỖ HỆ THỐNG CHẠM VÀO KIỂU CỦA OPENAI LÀ Ở ĐÂY
/// ============================================================
///
/// Vào: <see cref="AiRequest"/>. Ra: <see cref="AiResponse"/>. Ở giữa là
/// <c>ResponsesClient</c>. Không có kiểu nào của SDK lọt ra ngoài chữ ký công
/// khai của lớp này — đó là điều kiện để đổi sang Azure OpenAI hay nhà cung
/// cấp khác mà không ai ngoài file này phải biết.
///
/// VÌ SAO RESPONSES CHỨ KHÔNG PHẢI CHAT COMPLETIONS? Vì bốn thứ mà hợp đồng
/// kia không có, và cả bốn đều đã được đề bài yêu cầu: gọi công cụ có mã lượt
/// gọi, công cụ tra web phía nhà cung cấp, kết quả ép theo lược đồ, và tham số
/// mức suy luận. Làm bằng tay từng cái trên /chat/completions là viết lại đúng
/// thứ SDK đã có.
///
/// LƯU Ý VỀ PHIÊN BẢN SDK: tên kiểu trong gói 2.14.0 KHÔNG giống phần lớn ví dụ
/// trên mạng — ví dụ cũ dùng <c>OpenAIResponseClient</c>,
/// <c>ResponseCreationOptions</c>, <c>OpenAIResponse</c>. Ở 2.14.0 chúng là
/// <c>ResponsesClient</c>, <c>CreateResponseOptions</c>, <c>ResponseResult</c>.
/// Code dưới đây bám theo gói ĐANG CÀI, không theo ví dụ.
///
/// KHÔNG ném ngoại lệ cho lỗi của nhà cung cấp: trả về <c>Success = false</c>,
/// đúng hợp đồng của <see cref="IAiProvider"/>, để bộ định tuyến thử provider
/// tiếp theo mà không phải bắt ngoại lệ trong vòng lặp.
///
/// KHÔNG LƯU CHUỖI SUY LUẬN RIÊNG của model. SDK trả về
/// <c>ReasoningResponseItem</c>; lớp này chỉ lấy phần TÓM TẮT công khai và bỏ
/// phần còn lại. Nội dung suy luận nội bộ không đi vào log, không đi vào
/// <see cref="AiResponse"/>, không đi ra màn hình.
/// </summary>
public sealed class OpenAiResponsesProvider : IAiProvider
{
    private readonly ProviderOptions _options;
    private readonly SecretResolver _secrets;
    private readonly int _maxResponseCharacters;
    private readonly Func<string, ResponsesClient>? _clientFactory;

    public OpenAiResponsesProvider(
        ProviderOptions options,
        SecretResolver secrets,
        int maxResponseCharacters = 8_000,
        Func<string, ResponsesClient>? clientFactory = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(secrets);

        _options = options;
        _secrets = secrets;
        _maxResponseCharacters = maxResponseCharacters;

        // Cho phép tiêm client để test dựng được tình huống mà không gọi mạng.
        // Không có nó thì mọi test của lớp này đều phải có khoá thật.
        _clientFactory = clientFactory;
    }

    public string ProviderName => _options.Name;

    public IReadOnlyList<string> SupportedModels
    {
        get
        {
            if (_options.Models.Count > 0) return _options.Models;

            var declared = _options.TaskModels.All;

            if (declared.Count > 0) return declared;

            return _options.DefaultModel is null ? [] : [_options.DefaultModel];
        }
    }

    /// <summary>
    /// Sẵn sàng = CÓ khoá. Cố ý KHÔNG gọi thử một lượt sinh chữ.
    ///
    /// Cùng lý lẽ với <c>OpenAiCompatibleProvider</c>: mỗi lần kiểm bằng một
    /// lượt gọi thật là một lượt bị tính tiền, và màn hình chat kiểm mỗi khi
    /// mở. Provider hỏng sẽ lộ ra ở lần gọi thật và
    /// <c>ProviderHealthTracker</c> tạm loại nó ra.
    /// </summary>
    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(_secrets.Has(_options.ApiKeyEnvironmentVariable));

    public async Task<AiResponse> GenerateAsync(AiRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        string? model = request.Model ?? _options.DefaultModel ?? _options.TaskModels.For(request.TaskType);

        if (string.IsNullOrWhiteSpace(model))
        {
            return AiResponse.Fail(ProviderName, null, new AiError(AiErrorKind.Rejected,
                $"provider '{ProviderName}' chưa cấu hình model nào cho loại việc {request.TaskType} " +
                "— đặt DefaultModel hoặc TaskModels"));
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

        var stopwatch = Stopwatch.StartNew();

        try
        {
            var client = CreateClient(key, model);
            var options = BuildOptions(request, model);

            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(_options.Timeout);

            ClientResult<ResponseResult> result = await client
                .CreateResponseAsync(options, timeout.Token)
                .ConfigureAwait(false);

            stopwatch.Stop();

            return Translate(result.Value, model, stopwatch.Elapsed);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            stopwatch.Stop();

            return AiResponse.Fail(ProviderName, model,
                new AiError(AiErrorKind.Timeout, $"quá {_options.Timeout.TotalSeconds:F0}s không có phản hồi"),
                stopwatch.Elapsed);
        }
        catch (ClientResultException ex)
        {
            stopwatch.Stop();

            return AiResponse.Fail(ProviderName, model, OpenAiErrors.FromStatus(ex), stopwatch.Elapsed);
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or TimeoutException)
        {
            stopwatch.Stop();

            return AiResponse.Fail(ProviderName, model,
                new AiError(AiErrorKind.Network, OutboundRedactor.Redact(ex.Message)), stopwatch.Elapsed);
        }
    }

    private ResponsesClient CreateClient(string apiKey, string model)
    {
        if (_clientFactory is not null) return _clientFactory(model);

        var credential = new ApiKeyCredential(apiKey);

        // Endpoint bỏ trống = địa chỉ mặc định của SDK. Khai vào chỉ cần khi
        // trỏ tới Azure hay một proxy nội bộ.
        if (string.IsNullOrWhiteSpace(_options.Endpoint))
        {
            return new ResponsesClient(credential);
        }

        return new ResponsesClient(credential, new ResponsesClientOptions
        {
            Endpoint = new Uri(_options.Endpoint),
        });
    }

    /// <summary>
    /// Dịch <see cref="AiRequest"/> sang <c>CreateResponseOptions</c>.
    ///
    /// Bốn điều đáng chú ý, vì cả bốn đều là quyết định chứ không phải dịch máy:
    ///
    ///   1. CHỈ THỊ HỆ THỐNG đi vào <c>Instructions</c>, KHÔNG thành một lượt
    ///      trong danh sách. Đó là chỗ hợp đồng Responses dành riêng cho nó, và
    ///      nó được đối xử khác với nội dung người dùng.
    ///
    ///   2. TRẦN TOKEN RA lấy giá trị NHỎ HƠN giữa yêu cầu và cấu hình. Người
    ///      vận hành đặt trần, chỗ gọi không vượt được.
    ///
    ///   3. KHÔNG LƯU Ở PHÍA NHÀ CUNG CẤP trừ khi được bật tường minh
    ///      (<c>AllowProviderSideStorage</c>). Mặc định tắt: prompt là dữ liệu
    ///      người dùng.
    ///
    ///   4. SIÊU DỮ LIỆU NỘI BỘ KHÔNG ĐƯỢC GỬI. <see cref="AiRequest.Metadata"/>
    ///      mang id phiên và id lượt của hệ thống; chú thích của chính nó đã nói
    ///      rõ là không gửi ra ngoài.
    /// </summary>
    private CreateResponseOptions BuildOptions(AiRequest request, string model)
    {
        var items = new List<ResponseItem>();

        foreach (var message in request.ConversationHistory)
        {
            items.Add(message.Role switch
            {
                ChatRole.Assistant => ResponseItem.CreateAssistantMessageItem(message.Content),
                ChatRole.System => ResponseItem.CreateDeveloperMessageItem(message.Content),
                _ => ResponseItem.CreateUserMessageItem(message.Content),
            });
        }

        items.Add(ResponseItem.CreateUserMessageItem(request.UserPrompt));

        // Lượt công cụ đã chạy: gửi lại CẢ lời xin lẫn kết quả, theo đúng thứ
        // tự. Thiếu lời xin thì nhà cung cấp từ chối vì kết quả không gắn được
        // vào lượt gọi nào.
        foreach (var exchange in request.ToolHistory)
        {
            items.Add(ResponseItem.CreateFunctionCallItem(
                exchange.Call.CallId,
                exchange.Call.Name,
                BinaryData.FromString(string.IsNullOrWhiteSpace(exchange.Call.ArgumentsJson)
                    ? "{}"
                    : exchange.Call.ArgumentsJson)));

            items.Add(ResponseItem.CreateFunctionCallOutputItem(
                exchange.Call.CallId,
                exchange.Result.Output));
        }

        var options = new CreateResponseOptions(model, items)
        {
            MaxOutputTokenCount = _options.CapOutputTokens(request.MaxTokens),
            StoredOutputEnabled = _options.AllowProviderSideStorage,
        };

        if (!string.IsNullOrWhiteSpace(request.SystemPrompt))
        {
            options.Instructions = request.SystemPrompt;
        }

        // Model suy luận từ chối tham số nhiệt độ. Gửi kèm là đổi một lượt gọi
        // hợp lệ lấy một lỗi 400 — nên chỉ gửi khi KHÔNG đặt mức suy luận.
        string? effort = _options.ReasoningEffort;

        if (effort is null)
        {
            options.Temperature = (float)request.Temperature;
        }
        else
        {
            options.ReasoningOptions = new ResponseReasoningOptions
            {
                ReasoningEffortLevel = new ResponseReasoningEffortLevel(effort),
            };
        }

        foreach (var tool in request.Tools)
        {
            options.Tools.Add(ResponseTool.CreateFunctionTool(
                tool.Name,
                BinaryData.FromString(tool.ParametersJsonSchema),
                strictModeEnabled: false,
                functionDescription: tool.Description));
        }

        // Công cụ tra web PHÍA NHÀ CUNG CẤP — chỉ khi cấu hình bật VÀ loại việc
        // là tra cứu. Bật cho mọi lượt sẽ làm mọi câu hỏi đều có thể phát sinh
        // một lượt tìm kiếm tính tiền, kể cả "2 + 2 bằng mấy".
        if (_options.EnableWebSearch && request.TaskType == AiTaskType.WebResearch)
        {
            options.Tools.Add(ResponseTool.CreateWebSearchTool());
        }

        if (request.ResponseFormat is { } format)
        {
            options.TextOptions = new ResponseTextOptions
            {
                TextFormat = ResponseTextFormat.CreateJsonSchemaFormat(
                    format.Name,
                    BinaryData.FromString(format.JsonSchema),
                    format.Description,
                    format.Strict),
            };
        }

        return options;
    }

    /// <summary>
    /// Dịch <c>ResponseResult</c> về <see cref="AiResponse"/>.
    ///
    /// Duyệt <c>OutputItems</c> và gom hai thứ: CHỮ (từ các lượt message) và
    /// LỜI XIN GỌI CÔNG CỤ. Mọi mục khác — suy luận nội bộ, lượt tra web, lượt
    /// gọi MCP — bị BỎ QUA có chủ ý: chúng không phải câu trả lời, và phần suy
    /// luận thì còn không được phép lưu.
    /// </summary>
    private AiResponse Translate(ResponseResult result, string model, TimeSpan latency)
    {
        if (result.Error is { } error)
        {
            return AiResponse.Fail(ProviderName, model,
                new AiError(AiErrorKind.ProviderError,
                    OutboundRedactor.Redact(error.Message ?? error.Code.ToString())), latency);
        }

        var text = new System.Text.StringBuilder();
        var toolCalls = new List<AiToolCall>();

        foreach (var item in result.OutputItems)
        {
            switch (item)
            {
                case MessageResponseItem message:
                    foreach (var part in message.Content)
                    {
                        if (!string.IsNullOrEmpty(part.Text)) text.Append(part.Text);
                    }

                    break;

                case FunctionCallResponseItem call:
                    toolCalls.Add(new AiToolCall(
                        call.CallId ?? call.Id ?? call.FunctionName,
                        call.FunctionName,
                        call.FunctionArguments?.ToString() ?? "{}"));

                    break;
            }
        }

        var usage = result.Usage is { } tokens
            ? new AiUsage(tokens.InputTokenCount, tokens.OutputTokenCount)
            : AiUsage.Unknown;

        string content = text.ToString().Trim();

        if (content.Length > _maxResponseCharacters)
        {
            content = content[.._maxResponseCharacters];
        }

        if (toolCalls.Count > 0)
        {
            return AiResponse.Tools(ProviderName, result.Model ?? model, toolCalls, latency, usage, content);
        }

        // Chạm trần token là lý do dừng RIÊNG, không phải lỗi: có chữ trả về,
        // chỉ là bị cắt. Gộp nó vào lỗi sẽ khiến bộ định tuyến thử provider
        // khác — và provider khác cũng sẽ bị cắt đúng như vậy.
        bool incomplete = result.Status == ResponseStatus.Incomplete;

        if (content.Length == 0)
        {
            return AiResponse.Fail(ProviderName, result.Model ?? model,
                new AiError(AiErrorKind.InvalidResponse,
                    incomplete ? "phản hồi rỗng và chưa hoàn tất" : "phản hồi không có nội dung chữ"), latency);
        }

        return AiResponse.Ok(ProviderName, result.Model ?? model, content, latency, usage,
            incomplete ? AiFinishReason.Length : AiFinishReason.Stop);
    }
}
