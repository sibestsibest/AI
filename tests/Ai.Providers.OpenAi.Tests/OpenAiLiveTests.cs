using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.OpenAi;

namespace Ai.Providers.OpenAi.Tests;

/// <summary>
/// TEST GỌI API THẬT — mặc định BỊ BỎ QUA.
///
/// ============================================================
/// VÌ SAO TÁCH RA, VÀ VÌ SAO MẶC ĐỊNH TẮT
/// ============================================================
///
/// Một bộ test gọi mạng là một bộ test hỏng: hôm nay xanh, mai đỏ vì mạng chậm
/// hay vì hạn mức, mà chẳng ai sửa gì. Tệ hơn, nó TỐN TIỀN mỗi lần chạy — kể
/// cả những lần chạy tự động mà không ai nhìn.
///
/// Nên những test này chỉ chạy khi có ĐỦ HAI thứ, đặt tường minh:
///
///     OPENAI_API_KEY        khoá thật
///     AI_OPENAI_LIVE_TESTS  đặt bằng "1"
///
/// Thiếu bất kỳ cái nào thì test được BỎ QUA (skip), không phải thất bại.
/// Phân biệt này quan trọng: "chưa cấu hình" không phải là "hỏng", và một bộ
/// test đỏ vì thiếu cấu hình sẽ nhanh chóng bị bỏ qua cả những lần nó đỏ thật.
///
/// Chạy chúng (PowerShell):
///
///     $env:OPENAI_API_KEY = "..."
///     $env:AI_OPENAI_LIVE_TESTS = "1"
///     dotnet test tests/Ai.Providers.OpenAi.Tests
/// </summary>
public class OpenAiLiveTests
{
    private const string KeyVariable = "OPENAI_API_KEY";
    private const string SwitchVariable = "AI_OPENAI_LIVE_TESTS";

    /// <summary>Model dùng cho test thật. Đọc từ cấu hình — KHÔNG ghi cứng tên model nào.</summary>
    private const string ModelVariable = "AI_OPENAI_TEST_MODEL";

    private static string? Skip()
    {
        if (Environment.GetEnvironmentVariable(SwitchVariable) != "1")
        {
            return $"test gọi API thật đang tắt — đặt {SwitchVariable}=1 để bật";
        }

        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(KeyVariable)))
        {
            return $"chưa đặt {KeyVariable}";
        }

        return string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(ModelVariable))
            ? $"chưa đặt {ModelVariable} (tên model để thử)"
            : null;
    }

    private static OpenAiResponsesProvider Provider() => new(
        new ProviderOptions
        {
            Name = "openai-live",
            Type = ProviderKind.OpenAiResponses,
            Endpoint = "",
            ApiKeyEnvironmentVariable = KeyVariable,
            DefaultModel = Environment.GetEnvironmentVariable(ModelVariable),
            TimeoutSeconds = 60,
            MaxRetries = 0,
        },
        new SecretResolver());

    [Fact]
    public async Task GoiThatVaNhanDuocChu()
    {
        // xunit 2.x KHÔNG bỏ qua test lúc chạy được (Assert.Skip là của xunit 3),
        // nên chỗ này thoát sớm. Test hiện là XANH khi chưa cấu hình — đúng ý:
        // "chưa bật" không phải "hỏng", và một bộ test đỏ vì thiếu cấu hình sẽ
        // nhanh chóng bị mọi người bỏ qua cả lúc nó đỏ thật.
        if (Skip() is not null) return;

        var response = await Provider().GenerateAsync(new AiRequest
        {
            UserPrompt = "Trả lời đúng một từ: thủ đô của Việt Nam?",
            MaxTokens = 64,
        });

        Assert.True(response.Success, response.Error?.ToString());
        Assert.NotEmpty(response.Content);
        Assert.True(response.Usage.IsMeasured, "phải đo được token để theo dõi chi phí");
    }

    [Fact]
    public async Task GoiThatCoCongCuThiXinGoiCongCu()
    {
        if (Skip() is not null) return;

        var response = await Provider().GenerateAsync(new AiRequest
        {
            UserPrompt = "Tính 1234 * 5678. Dùng công cụ calculator.",
            MaxTokens = 256,
            Tools =
            [
                new AiToolDefinition("calculator", "tính toán số học chính xác")
                {
                    ParametersJsonSchema =
                        "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}," +
                        "\"required\":[\"query\"],\"additionalProperties\":false}",
                },
            ],
        });

        Assert.True(response.Success, response.Error?.ToString());
        Assert.True(response.WantsTools, "model đáng lẽ phải xin gọi công cụ cho phép nhân này");
        Assert.Equal("calculator", response.ToolName);
    }
}
