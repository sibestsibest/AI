using Ai.Phase16.Grounding;
using Ai.Phase16.Memory;
using Ai.Phase16.Reasoning;
using Ai.Phase16.Verification;
using Ai.Providers.Configuration;
using Ai.Providers.Contracts;
using Ai.Providers.Integration;
using Ai.Providers.Providers;
using Ai.Providers.Resilience;
using Ai.Providers.Routing;

namespace Ai.Providers.Tests;

/// <summary>
/// Chạy ĐÚNG đường ống trả lời của Phase 14, chỉ thay bộ sinh ở khâu diễn đạt.
///
/// Dựng đường ống trực tiếp thay vì dựng cả agent, vì agent phải train bộ phân
/// loại ý định (vài giây) mà bộ phân loại không tham gia gì vào khâu đang kiểm.
/// </summary>
public class PipelineWithLlmTests
{
    /// <summary>Một câu lấy nguyên từ tài liệu SSRF trong kho kiến thức — "diễn đạt lại" hợp lệ.</summary>
    private const string GroundedRephrasing =
        "SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ.";

    private const string Fabrication =
        "Mọi tường lửa thương mại đều đã chặn hoàn toàn kiểu tấn công này từ năm 2019.";

    private static (AnswerPipeline Pipeline, LlmAnswerGenerator Generator, ScriptedAiProvider Provider) Build(
        string reply)
    {
        var clock = new ManualClock(Cfg.Now);

        var provider = ScriptedAiProvider.AlwaysReplies("local", reply);

        var router = new AiProviderRouter(
            Cfg.Options(ProviderStrategy.LocalFirst, Cfg.Local()),
            [provider],
            new ProviderHealthTracker(clock));

        var generator = new LlmAnswerGenerator(router);

        var collector = new EvidenceCollector(
            KnowledgeSeed.BuildStore(clock),
            new MemoryStore(clock),
            new ConversationContext(clock),
            internet: null,
            clock: clock);

        var reasoning = new ReasoningEngine();

        var pipeline = new AnswerPipeline(collector, reasoning, generator: generator, clock: clock);

        return (pipeline, generator, provider);
    }

    /// <summary>
    /// Đường đi đầy đủ: căn cứ (P13) → kiểm chứng (P14) → LLM diễn đạt → chốt
    /// chặn (P14). Văn bản của LLM dùng lại từ ngữ của căn cứ nên qua được.
    /// </summary>
    [Fact]
    public async Task LlmDienDatLaiCanCuThiCauTraLoiLaVanCuaLlm()
    {
        var (pipeline, generator, provider) = Build(GroundedRephrasing);

        var answer = await pipeline.AnswerAsync("SSRF là gì");

        Assert.Equal(1, provider.Calls);
        Assert.True(generator.Last.UsedLlm, generator.Last.RejectedReason);
        Assert.Equal(AnswerVerdict.Answered, answer.Verdict);
        Assert.Contains(GroundedRephrasing, answer.Text, StringComparison.Ordinal);

        // Và câu trả lời vẫn TRUY ĐƯỢC NGUỒN — đây là điều không được mất.
        Assert.NotEmpty(answer.CitedEvidence);
        Assert.Contains("kho kiến thức nội bộ", answer.Text, StringComparison.Ordinal);
        Assert.True(answer.Validation!.IsValid);
        Assert.True(answer.AllClaimsSupported);
    }

    /// <summary>
    /// TÍNH CHẤT QUAN TRỌNG NHẤT CỦA CẢ VIỆC TÍCH HỢP NÀY.
    ///
    /// LLM bịa một câu mà căn cứ không hề nói. Kết quả: câu bịa KHÔNG ra tới
    /// người dùng, câu trả lời vẫn là câu khuôn mẫu cũ, và kết luận KHÔNG bị hạ
    /// xuống "không đủ căn cứ". Tức là tệ nhất, việc bật LLM cũng chỉ quay về
    /// đúng hành vi trước khi có nó — không có đường nào để câu trả lời tệ hơn.
    /// </summary>
    [Fact]
    public async Task LlmBiaThiCauTraLoiQuayVeKhuonMauChuKhongMatCauTraLoi()
    {
        var (pipeline, generator, _) = Build(Fabrication);

        var answer = await pipeline.AnswerAsync("SSRF là gì");

        Assert.False(generator.Last.UsedLlm);
        Assert.Equal(1, generator.RejectedCount);

        Assert.DoesNotContain("tường lửa thương mại", answer.Text, StringComparison.Ordinal);
        Assert.Equal(AnswerVerdict.Answered, answer.Verdict);
        Assert.True(answer.Validation!.IsValid);
        Assert.Contains("SSRF", answer.Text, StringComparison.Ordinal);
    }

    /// <summary>
    /// So hai đường: bật LLM và không bật, trên cùng một câu hỏi. Kết luận, độ
    /// tự tin và danh sách căn cứ phải GIỐNG NHAU — LLM chỉ đổi cách viết, nó
    /// không được đổi thứ gì trong phần kiểm chứng.
    /// </summary>
    [Fact]
    public async Task BatLlmKhongLamDoiKetLuanDoTuTinHayCanCu()
    {
        var clock = new ManualClock(Cfg.Now);

        var collector = new EvidenceCollector(
            KnowledgeSeed.BuildStore(clock),
            new MemoryStore(clock),
            new ConversationContext(clock),
            internet: null,
            clock: clock);

        var withoutLlm = await new AnswerPipeline(collector, new ReasoningEngine(), clock: clock)
            .AnswerAsync("SSRF là gì");

        var (pipeline, _, _) = Build(GroundedRephrasing);
        var withLlm = await pipeline.AnswerAsync("SSRF là gì");

        Assert.Equal(withoutLlm.Verdict, withLlm.Verdict);
        Assert.Equal(withoutLlm.Confidence, withLlm.Confidence, 12);
        Assert.Equal(
            withoutLlm.CitedEvidence.Select(e => e.Id),
            withLlm.CitedEvidence.Select(e => e.Id));
        Assert.Equal(withoutLlm.Claims.Select(c => c.Text), withLlm.Claims.Select(c => c.Text));

        // Chỉ VĂN BẢN khác nhau.
        Assert.NotEqual(withoutLlm.Text, withLlm.Text);
    }

    /// <summary>Câu ngoài kiến thức: LLM không được gọi, và agent vẫn nói thẳng là không biết.</summary>
    [Fact]
    public async Task CauNgoaiKienThucThiKhongGoiLlmVaVanNoiKhongBiet()
    {
        var (pipeline, generator, provider) = Build(GroundedRephrasing);

        var answer = await pipeline.AnswerAsync("cách nấu phở bò Hà Nội");

        Assert.Equal(AnswerVerdict.Insufficient, answer.Verdict);
        Assert.Equal(0, provider.Calls);
        Assert.False(generator.Last.UsedLlm);
    }

    /// <summary>Provider chết giữa phiên thì đường ống vẫn trả lời, chỉ là bằng văn khuôn mẫu.</summary>
    [Fact]
    public async Task ProviderChetThiDuongOngVanTraLoi()
    {
        var clock = new ManualClock(Cfg.Now);

        var provider = ScriptedAiProvider.AlwaysFails("local", AiErrorKind.Network, "không nối được");

        var router = new AiProviderRouter(Cfg.Options(ProviderStrategy.LocalFirst, Cfg.Local()),
            [provider], new ProviderHealthTracker(clock));

        var generator = new LlmAnswerGenerator(router);

        var collector = new EvidenceCollector(
            KnowledgeSeed.BuildStore(clock),
            new MemoryStore(clock),
            new ConversationContext(clock),
            internet: null,
            clock: clock);

        var answer = await new AnswerPipeline(collector, new ReasoningEngine(),
            generator: generator, clock: clock).AnswerAsync("SSRF là gì");

        Assert.Equal(AnswerVerdict.Answered, answer.Verdict);
        Assert.False(generator.Last.UsedLlm);
        Assert.Contains("SSRF", answer.Text, StringComparison.Ordinal);
    }
}
