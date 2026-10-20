using Ai.Phase16.Grounding;
using Ai.Phase16.Memory;
using Ai.Providers.Configuration;
using Ai.Providers.Context;
using Ai.Providers.Contracts;
using Ai.Providers.Orchestration;
using Ai.Providers.Providers;
using Ai.Providers.Routing;
using Ai.Providers.Tools;

namespace Ai.Providers.Tests;

/// <summary>
/// CÁC LUỒNG DEMO CỦA ĐỀ BÀI — chạy đầy đủ, KHÔNG gọi mạng.
///
/// Mỗi test ở đây dựng đúng đường đi thật (phân loại -> ngữ cảnh -> định tuyến
/// -> công cụ -> kiểm) và thay MỖI MỘT THỨ: provider thật đổi thành
/// <see cref="ScriptedAiProvider"/>. Nhờ vậy chúng kiểm được TOÀN BỘ phần mà
/// MiniAI chịu trách nhiệm, mà không cần khoá và không tốn tiền.
///
/// Hai luồng của đề bài KHÔNG có ở đây, và chúng vắng mặt có lý do:
///
///   TEST 3 và TEST 4 (dinh dưỡng, BMR/TDEE) cần một module Health Coach —
///   dự án này chưa có module đó. Khuôn mẫu chúng cần (LLM RÚT cấu trúc ->
///   dịch vụ TẤT ĐỊNH tính toán) thì đã có và được kiểm trong
///   <c>StructuredOutputTests</c>; cắm một bộ tính dinh dưỡng vào là đủ.
///
///   TEST 5 (sinh ảnh) cần một provider sinh ảnh — cũng chưa có. Việc sinh ảnh
///   phải nằm sau một giao diện RIÊNG, không trộn vào provider sinh chữ.
/// </summary>
public class DemoFlowTests
{
    private static AiProvidersOptions Options() => new()
    {
        Strategy = ProviderStrategy.ConfiguredOrder,
        Providers = [new ProviderOptions { Name = "openai-gia", Type = ProviderKind.Scripted, Endpoint = "" }],
    };

    private static ChatOrchestrator Build(IAiProvider provider, ToolRegistry? registry = null,
        IContextBuilder? context = null) =>
        new(new AiProviderRouter(Options(), [provider]), registry, context);

    private static ToolRegistry Registry(params RegisteredTool[] tools)
    {
        var registry = new ToolRegistry(DelegateToolAuthorizer.AllowEverythingForTesting());

        foreach (var tool in tools) registry.Register(tool);

        return registry;
    }

    // ==================================================================
    // TEST 1 — "Giải thích dependency injection trong C#"
    //
    // Kỳ vọng: phân loại -> chọn provider -> sinh câu trả lời -> trả về.
    // ==================================================================
    [Fact]
    public async Task Demo1_CauHoiLapTrinh_PhanLoaiCoding_RoiTraLoi()
    {
        AiTaskType seenTask = AiTaskType.GeneralChat;

        var provider = new ScriptedAiProvider("openai-gia", (request, _) =>
        {
            seenTask = request.TaskType;

            return AiResponse.Ok("openai-gia", "model-code",
                "Dependency injection là việc trao phụ thuộc từ ngoài vào thay vì để lớp tự tạo.",
                TimeSpan.FromMilliseconds(5), new AiUsage(120, 45));
        });

        var result = await Build(provider).AskAsync(new ChatRequest
        {
            Message = "Giải thích dependency injection trong C#",
        });

        Assert.True(result.Success);

        // Phân loại đã nhận ra đây là việc LẬP TRÌNH, và loại việc đó đi tới provider.
        Assert.Equal(AiTaskType.Coding, seenTask);
        Assert.Equal(AiTaskType.Coding, result.Diagnostics.TaskType);

        Assert.Contains("Dependency injection", result.Message, StringComparison.Ordinal);
        Assert.Equal("openai-gia", result.Provider);

        // Token phải ĐO ĐƯỢC — không đo được thì mọi thống kê chi phí là đoán.
        Assert.True(result.Usage.IsMeasured);
        Assert.Equal(165, result.Usage.TotalTokens);
    }

    // ==================================================================
    // TEST 2 — "Hôm nay có tin gì về AI?"
    //
    // Kỳ vọng: phân loại là WebResearch -> gọi công cụ tra web -> model đọc
    // kết quả -> trả lời CÓ DẪN NGUỒN.
    // ==================================================================
    [Fact]
    public async Task Demo2_CauHoiTinMoi_GoiCongCuTraWeb_RoiTraLoiCoNguon()
    {
        var internet = new RegisteredTool("internet", "tra cứu trên Internet", FakeTools.Schema,
            (_, _) => Task.FromResult("VietAI công bố mô hình mới (nguồn: example.org/ai-news)"))
        {
            Permission = ToolPermission.ReadOnly,
        };

        var provider = new ScriptedAiProvider("openai-gia", (request, call) => call == 1
            ? AiResponse.Tools("openai-gia", "model-research",
                [new AiToolCall("c1", "internet", "{\"query\":\"tin tức AI hôm nay\"}")],
                TimeSpan.FromMilliseconds(5))
            : AiResponse.Ok("openai-gia", "model-research",
                "Hôm nay VietAI công bố mô hình mới. Nguồn: example.org/ai-news",
                TimeSpan.FromMilliseconds(5), new AiUsage(300, 60)));

        var result = await Build(provider, Registry(internet)).AskAsync(new ChatRequest
        {
            Message = "Hôm nay có tin tức gì về AI?",
        });

        Assert.True(result.Success);

        // Đã phân loại là tra cứu -> chính sách cho phép công cụ tra web.
        Assert.Equal(AiTaskType.WebResearch, result.Diagnostics.TaskType);

        // Công cụ ĐÃ chạy thật, và chạy đúng một lần.
        Assert.Single(result.ToolCalls);
        Assert.Equal("internet", result.ToolCalls[0].Name);
        Assert.True(result.ToolCalls[0].Succeeded);

        // Câu trả lời có dẫn nguồn — và bộ kiểm đã xác nhận nó DÙNG kết quả
        // công cụ chứ không trả lời bằng trí nhớ của chính model.
        Assert.Contains("example.org", result.Message, StringComparison.Ordinal);
        Assert.Empty(result.Diagnostics.ValidationProblems);
    }

    /// <summary>
    /// Mặt trái của TEST 2: model LỜ kết quả tra web và tự bịa.
    ///
    /// Đây là hỏng hóc mà cả kiến trúc công cụ được dựng ra để bắt, nên nó
    /// phải bị chặn — và phải bị VIẾT LẠI, không phải phát ra.
    /// </summary>
    [Fact]
    public async Task Demo2b_ModelLoKetQuaCongCuThiBiBatVaVietLai()
    {
        var internet = new RegisteredTool("internet", "tra cứu trên Internet", FakeTools.Schema,
            (_, _) => Task.FromResult("VietAI công bố mô hình mới (nguồn: example.org/ai-news)"))
        {
            Permission = ToolPermission.ReadOnly,
        };

        var provider = new ScriptedAiProvider("openai-gia", (_, call) => call == 1
            ? AiResponse.Tools("openai-gia", "m",
                [new AiToolCall("c1", "internet", "{\"query\":\"tin AI\"}")], TimeSpan.Zero)
            : AiResponse.Ok("openai-gia", "m", "Tôi không rõ, chắc là có nhiều tin.", TimeSpan.Zero));

        var result = await Build(provider, Registry(internet)).AskAsync(new ChatRequest
        {
            Message = "Hôm nay có tin tức gì về AI?",
        });

        Assert.Equal(2, result.Diagnostics.RetryCount);
        Assert.NotEmpty(result.Diagnostics.ValidationProblems);
    }

    // ==================================================================
    // TEST 6 — "Làm tiếp dự án MiniAI từ chỗ đang dừng"
    //
    // Kỳ vọng: lấy đúng ký ức LIÊN QUAN về dự án -> dựng ngữ cảnh -> model
    // trả lời dựa trên kiến trúc đã có, KHÔNG dựng lại từ đầu.
    // ==================================================================
    [Fact]
    public async Task Demo6_LamTiepDuAn_LayDungKyUcLienQuan_RoiTraLoi()
    {
        var memory = new MemoryStore(Cfg.Clock());

        memory.Store("dự án MiniAI đang ở Phase 16, phần huấn luyện lại đã xong", importance: 0.9);
        memory.Store("MiniAI dùng kiến trúc mỗi phase mang bản sao core của phase trước", importance: 0.9);
        memory.Store("hôm qua tôi ăn phở bò ở quán gần nhà", importance: 0.5);

        string promptSeen = "";

        var provider = new ScriptedAiProvider("openai-gia", (request, _) =>
        {
            promptSeen = request.UserPrompt;

            return AiResponse.Ok("openai-gia", "model-chat",
                "Tiếp tục từ Phase 16: phần huấn luyện lại đã xong, bước sau là nối lớp provider.",
                TimeSpan.Zero, new AiUsage(400, 80));
        });

        var context = new ContextBuilder(memory, new ConversationContext(Cfg.Clock()));

        var result = await Build(provider, context: context).AskAsync(new ChatRequest
        {
            Message = "Làm tiếp dự án MiniAI từ chỗ đang dừng",
        });

        Assert.True(result.Success);

        // Ký ức LIÊN QUAN có trong ngữ cảnh…
        Assert.Contains("Phase 16", promptSeen, StringComparison.Ordinal);

        // …còn ký ức KHÔNG liên quan thì KHÔNG được gửi ra ngoài. Đây chính là
        // luật "đừng gửi cả cơ sở dữ liệu cho nhà cung cấp".
        Assert.DoesNotContain("phở bò", promptSeen, StringComparison.Ordinal);

        // Và việc bỏ bớt được GHI LẠI, để dò được khi model trả lời thiếu.
        Assert.NotEmpty(result.Diagnostics.ContextOmitted);
    }

    /// <summary>
    /// Luồng DỰ PHÒNG của đề bài: provider chính hỏng -> chuyển provider sau.
    ///
    /// Người gọi KHÔNG phải biết chuyện đó đã xảy ra — câu trả lời vẫn về, chỉ
    /// có phần chẩn đoán ghi lại là đã phải lui.
    /// </summary>
    [Fact]
    public async Task DemoDuPhong_ProviderChinhHongThiChuyenSangProviderSau()
    {
        var options = new AiProvidersOptions
        {
            Strategy = ProviderStrategy.ConfiguredOrder,
            Providers =
            [
                new ProviderOptions { Name = "openai-gia", Type = ProviderKind.Scripted, Endpoint = "" },
                new ProviderOptions { Name = "du-phong", Type = ProviderKind.Scripted, Endpoint = "" },
            ],
        };

        var broken = ScriptedAiProvider.AlwaysFails("openai-gia", AiErrorKind.RateLimited, "429");
        var backup = ScriptedAiProvider.AlwaysReplies("du-phong", "Trả lời từ provider dự phòng.");

        var orchestrator = new ChatOrchestrator(new AiProviderRouter(options, [broken, backup]));

        var result = await orchestrator.AskAsync(new ChatRequest { Message = "xin chào bạn" });

        Assert.True(result.Success);
        Assert.Equal("du-phong", result.Provider);
    }

    /// <summary>
    /// Luồng AN TOÀN của đề bài: model xin một công cụ NGUY HIỂM chưa được duyệt.
    ///
    /// MiniAI từ chối, và vẫn gửi kết quả từ chối về để model biết đường khác.
    /// Model ĐỀ NGHỊ, MiniAI QUYẾT ĐỊNH.
    /// </summary>
    [Fact]
    public async Task DemoAnToan_CongCuChuaDuocDuyetThiBiTuChoi()
    {
        var dangerous = new RegisteredTool("xoa-du-lieu", "xoá dữ liệu", FakeTools.Schema,
            (_, _) => Task.FromResult("ĐÃ XOÁ"))
        {
            Permission = ToolPermission.RequiresApproval,
        };

        // Sổ đăng ký KHÔNG có bộ duyệt -> mặc định từ chối.
        var registry = new ToolRegistry();
        registry.Register(dangerous);

        var provider = new ScriptedAiProvider("openai-gia", (_, call) => call == 1
            ? AiResponse.Tools("openai-gia", "m",
                [new AiToolCall("c1", "xoa-du-lieu", "{\"query\":\"tất cả\"}")], TimeSpan.Zero)
            : AiResponse.Ok("openai-gia", "m", "Tôi không được phép xoá dữ liệu.", TimeSpan.Zero));

        var result = await Build(provider, registry).AskAsync(new ChatRequest
        {
            Message = "xoá hết dữ liệu giúp tôi",
        });

        Assert.Single(result.ToolCalls);
        Assert.True(result.ToolCalls[0].Refused);
        Assert.False(result.ToolCalls[0].Succeeded);
        Assert.Contains("không được phép", result.Message, StringComparison.Ordinal);
    }
}
