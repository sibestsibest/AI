using Ai.Phase16.Agent;
using Ai.Phase16.Grounding;
using Ai.Phase16.Internet;
using Ai.Phase16.Memory;
using Ai.Phase16.Tools;

namespace Ai.Phase16.Tests;

/// <summary>
/// HỒI QUY PHASE 12: agent dựng KHÔNG có tầng evidence phải hành xử đúng như
/// Phase 12 đã hành xử.
///
/// Đây là cách trả lời câu "Phase 13 có phá gì của Phase 12 không" bằng mã
/// chạy được, thay vì bằng lời hứa. Toàn bộ những khẳng định dưới đây được
/// giữ NGUYÊN VĂN từ bộ test của Phase 12.
/// </summary>
[Collection("agent")]
public class Phase12InternetRegressionTests(TrainedAgent fixture)
{
    [Fact]
    public async Task CauHoiCanInternet_ThiGoiCongCuInternet()
    {
        var response = await fixture.BuildPhase12Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.Equal(Intent.SearchInternet, response.Intent);
        Assert.Equal("internet", response.ToolUsed);
        Assert.Contains("owasp.org", response.Answer);
    }

    [Fact]
    public async Task CauTraLoiTuInternet_DuocGanCoChuaKiemChung()
    {
        var response = await fixture.BuildPhase12Style()
            .HandleAsync("tra trên internet xem prompt injection là gì", CancellationToken.None);

        Assert.True(response.UsedUnverifiedInternetData);
        Assert.Contains("CHƯA KIỂM CHỨNG", response.Answer);
        Assert.Contains(response.Decisions, d => d.Question == "Có được coi là sự thật?");
    }

    [Fact]
    public async Task CauTraLoiLuonDanNguon()
    {
        var response = await fixture.BuildPhase12Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.Contains("nguồn:", response.Answer);
        Assert.Contains("https://", response.Answer);
        Assert.NotEmpty(response.InternetReport!.Sources);
    }

    [Fact]
    public async Task KhongTimThayNguonNao_ThiNoiThat()
    {
        var response = await fixture.BuildPhase12Style()
            .HandleAsync("tìm trên mạng cách nấu phở bò Hà Nội", CancellationToken.None);

        Assert.False(response.UsedUnverifiedInternetData);
        Assert.Contains("không tra được", response.Answer);
    }

    [Fact]
    public async Task LuongQuyetDinhGhiLaiBuocInternet()
    {
        var response = await fixture.BuildPhase12Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        var questions = response.Decisions.Select(d => d.Question).ToArray();

        Assert.Contains("Có cần Internet?", questions);
        Assert.Contains("Có công cụ Internet?", questions);
        Assert.Contains("Internet trả về gì?", questions);
        Assert.DoesNotContain("Có cần Tool?", questions);
    }

    [Fact]
    public async Task ChiTietNeuTruyVanDaGui()
    {
        var response = await fixture.BuildPhase12Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.Contains("truy vấn:", response.Detail);
    }

    [Fact]
    public async Task DungCacheKhiHoiLaiCungCau()
    {
        var agent = fixture.BuildPhase12Style();

        await agent.HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);
        var second = await agent.HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.True(second.InternetReport?.FromCache);
        Assert.Contains("cache", second.Detail);
    }

    [Fact]
    public async Task TrangCoTiemChiThi_BiLamCunVaTruDiem()
    {
        var response = await fixture.BuildPhase12Style()
            .HandleAsync("tìm trên mạng mẹo tăng tốc model", CancellationToken.None);

        var evil = response.InternetReport!.Sources
            .FirstOrDefault(s => s.Result.Host.Contains("evil", StringComparison.OrdinalIgnoreCase));

        Assert.NotNull(evil);
        Assert.True(evil.Result.Snippet.LooksLikeInjection);
        Assert.DoesNotContain("Ignore all previous instructions", response.Answer);
        Assert.Contains("CẢNH BÁO", response.Detail);
    }

    [Fact]
    public async Task TraCuuThatBai_ThiNoiKhongCoThongTin()
    {
        var failing = new InternetSearchTool(new SearchService(
            new FailingProvider(),
            clock: new ManualClock(TrainedAgent.FixedNow),
            options: new SearchServiceOptions { FetchPages = false }));

        var agent = fixture.BuildFresh(new ManualClock(TrainedAgent.FixedNow), failing, withEvidence: false);

        var response = await agent.HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.False(response.UsedUnverifiedInternetData);
        Assert.Contains("không có thông tin nào", response.Answer);
    }

    [Fact]
    public async Task NoiDungInternetKhongTuDongThanhKyUc()
    {
        var agent = fixture.BuildPhase12Style();

        await agent.HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);
        await agent.HandleAsync("tra trên internet xem prompt injection là gì", CancellationToken.None);

        Assert.Equal(0, agent.Memory.LongTerm.Count);
        Assert.Equal(0, agent.Memory.ShortTerm.Count);
    }
}

/// <summary>
/// PHASE 13: nhánh Internet giờ đi qua EVIDENCE.
///
/// Khác biệt lớn nhất so với Phase 12 không phải là định dạng câu trả lời, mà
/// là: cùng một câu hỏi giờ được đối chiếu với CẢ kho nội bộ, nên agent biết
/// khi nào nó KHÔNG cần phải cảnh báo nữa.
/// </summary>
[Collection("agent")]
public class Phase13EvidenceRegressionTests(TrainedAgent fixture)
{
    [Fact]
    public async Task NhanhInternetDiQuaEvidence()
    {
        var response = await fixture.BuildPhase13Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.Equal(Intent.SearchInternet, response.Intent);
        Assert.Equal("evidence", response.ToolUsed);
        Assert.NotNull(response.Evidence);
        Assert.True(response.Evidence.HasEvidence);
    }

    /// <summary>
    /// ĐÂY LÀ TEST THỂ HIỆN RÕ NHẤT GIÁ TRỊ CỦA PHASE 13.
    ///
    /// Kho nội bộ CÓ tài liệu về SSRF, web cũng có. Hai kênh độc lập cùng nói
    /// về nó, nên agent không còn phải nói "chưa kiểm chứng" — nó có nguồn đã
    /// kiểm đỡ lưng. Ở Phase 12, cùng câu hỏi này luôn kèm cảnh báo.
    /// </summary>
    [Fact]
    public async Task CoNguonDaKiemThiKhongCanCanhBao()
    {
        var response = await fixture.BuildPhase13Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.True(response.Evidence!.HasVerifiedEvidence);
        Assert.False(response.Evidence.RequiresDisclosure);
        Assert.False(response.UsedUnverifiedInternetData);
        Assert.Contains("có nguồn đã kiểm chứng", response.Answer);

        // Và mẩu căn cứ đứng đầu phải là mẩu ĐÃ KIỂM, không phải mẩu từ web.
        Assert.Equal(SourceKind.Knowledge, response.Evidence.Best!.Kind);
    }

    /// <summary>Ngược lại: chủ đề kho nội bộ KHÔNG có thì vẫn phải cảnh báo.</summary>
    [Fact]
    public async Task KhongCoNguonDaKiemThiVanCanhBao()
    {
        var response = await fixture.BuildPhase13Style()
            .HandleAsync("tìm trên mạng giá bitcoin hiện tại", CancellationToken.None);

        Assert.True(response.Evidence!.HasEvidence);
        Assert.False(response.Evidence.HasVerifiedEvidence);
        Assert.True(response.Evidence.RequiresDisclosure);
        Assert.True(response.UsedUnverifiedInternetData);
        Assert.Contains("CHƯA KIỂM CHỨNG", response.Answer);
    }

    [Fact]
    public async Task MoiCanCuDeuDanDuocNguon()
    {
        var response = await fixture.BuildPhase13Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.All(response.Evidence!.Items, e =>
        {
            Assert.NotEmpty(e.Source.Name);
            Assert.NotEmpty(e.Source.Cite());
            Assert.InRange(e.Reliability, 0.0, 1.0);
            Assert.InRange(e.Relevance, 0.0, 1.0);
        });

        // Trích dẫn phải xuất hiện trong chính câu trả lời, không chỉ trong metadata.
        Assert.Contains("kho kiến thức nội bộ", response.Answer);
    }

    [Fact]
    public async Task GhiLaiBuocThuThapCanCuTrongLuongQuyetDinh()
    {
        var response = await fixture.BuildPhase13Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        var questions = response.Decisions.Select(d => d.Question).ToArray();

        Assert.Contains("Thu thập căn cứ từ đâu?", questions);
        Assert.Contains("Có căn cứ nào không?", questions);
        Assert.Contains("Có nguồn đã kiểm chứng?", questions);
    }

    [Fact]
    public async Task KyUcNguoiDungGopVaoCanCu()
    {
        var clock = new ManualClock(TrainedAgent.FixedNow);
        var agent = fixture.BuildFresh(clock, InternetStackFactory.BuildOffline(clock), withVerification: false);

        await agent.HandleAsync("hãy nhớ rằng tôi đang nghiên cứu SSRF", CancellationToken.None);

        var set = await agent.CollectEvidenceAsync("SSRF là gì", CancellationToken.None);

        Assert.Contains(set.Items, e => e.Kind == SourceKind.Memory);
        Assert.Contains(set.Items, e => e.Kind == SourceKind.Knowledge);
        Assert.True(set.DistinctChannelCount >= 2);
    }

    /// <summary>LUẬT CỨNG: tra mạng bao nhiêu lần cũng không làm kho kiến thức lớn lên.</summary>
    [Fact]
    public async Task TraMangKhongLamTangKhoKienThucHayKyUc()
    {
        var agent = fixture.BuildPhase13Style();
        int knowledgeBefore = agent.Knowledge!.Count;

        await agent.HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);
        await agent.HandleAsync("tra trên internet xem prompt injection là gì", CancellationToken.None);
        await agent.HandleAsync("tìm trên mạng tin tức trí tuệ nhân tạo mới nhất", CancellationToken.None);

        Assert.Equal(knowledgeBefore, agent.Knowledge.Count);
        Assert.Equal(0, agent.Memory.LongTerm.Count);
        Assert.DoesNotContain(agent.Knowledge.Entries, e => e.Origin.Contains("Internet"));
    }

    [Fact]
    public async Task KhongTimDuocCanCu_ThiNoiThat()
    {
        var response = await fixture.BuildPhase13Style()
            .HandleAsync("tìm trên mạng zzzqqq khongtontai wxyz", CancellationToken.None);

        Assert.Contains("không tìm được căn cứ", response.Answer);
        Assert.False(response.UsedUnverifiedInternetData);
    }

    [Fact]
    public async Task ChiTietNeuDauVetChamDiem()
    {
        var response = await fixture.BuildPhase13Style()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.Contains("khớp", response.Detail);
        Assert.Contains("tin ", response.Detail);
        Assert.Contains("kênh", response.Detail);
    }

    // ------------------------------------------------------------------
    // Ngữ cảnh hội thoại
    // ------------------------------------------------------------------

    [Fact]
    public async Task MoiLuotDuocGhiVaoNguCanh()
    {
        var agent = fixture.BuildPhase13Style();

        await agent.HandleAsync("xin chào", CancellationToken.None);
        var second = await agent.HandleAsync("tính 2 + 3 * 4", CancellationToken.None);

        Assert.Equal(2, agent.Context!.Count);
        Assert.Equal(2, second.TurnIndex);
        Assert.Equal("tính 2 + 3 * 4", agent.Context.Last!.UserInput);
        Assert.Equal(Intent.Calculate, agent.Context.Last.Intent);
    }

    /// <summary>
    /// Lượt phải được ghi SAU khi trả lời. Nếu ghi trước, chính câu hỏi đang
    /// xử lý sẽ thành "căn cứ" để trả lời nó — lập luận vòng tròn.
    /// </summary>
    [Fact]
    public async Task LuotHienTaiKhongTuLamCanCuChoChinhNo()
    {
        var clock = new ManualClock(TrainedAgent.FixedNow);
        var agent = fixture.BuildFresh(clock, InternetStackFactory.BuildOffline(clock), withVerification: false);

        var set = await agent.CollectEvidenceAsync("SSRF là gì", CancellationToken.None);

        Assert.DoesNotContain(set.Items, e => e.Kind == SourceKind.Conversation);
    }

    [Fact]
    public async Task NguCanhBoSungChoCauHoiQuaNgan()
    {
        var agent = fixture.BuildPhase13Style();

        await agent.HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        var set = await agent.CollectEvidenceAsync("còn nó", CancellationToken.None);

        Assert.NotNull(set.ExpandedQuestion);
        Assert.Contains("ssrf", set.ExpandedQuestion, StringComparison.OrdinalIgnoreCase);
    }

    // ------------------------------------------------------------------
    // Hồi quy: mọi nhánh cũ vẫn chạy khi tầng evidence đã bật
    // ------------------------------------------------------------------

    [Fact]
    public async Task BatEvidence_KhongLamHongCacNhanhCu()
    {
        var agent = fixture.BuildPhase13Style();

        Assert.Contains("14", (await agent.HandleAsync("tính 2 + 3 * 4", CancellationToken.None)).Answer);
        Assert.Contains("chain rule", (await agent.HandleAsync("backpropagation là gì", CancellationToken.None)).Answer);
        Assert.Contains("4", (await agent.HandleAsync("có bao nhiêu đơn hàng", CancellationToken.None)).Answer);
        Assert.Equal("Tạm biệt.", (await agent.HandleAsync("tạm biệt", CancellationToken.None)).Answer);

        var calc = await agent.HandleAsync("tính 2 + 3 * 4", CancellationToken.None);
        Assert.False(calc.UsedUnverifiedInternetData);
        Assert.Null(calc.Evidence);
    }

    [Fact]
    public async Task KhongCoInternet_ThiNoiThatChuKhongThayThe()
    {
        // Không Internet VÀ không evidence -> phải nói thẳng là không tra được.
        var agent = fixture.BuildFresh(withEvidence: false, withVerification: false);

        var response = await agent.HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.Null(response.ToolUsed);
        Assert.False(response.UsedUnverifiedInternetData);
        Assert.Contains("chưa được cấp quyền truy cập Internet", response.Answer);
    }

    /// <summary>
    /// Có evidence nhưng KHÔNG có Internet: agent vẫn trả lời được từ các kênh
    /// nội bộ, và không hề giả vờ là đã lên mạng.
    /// </summary>
    [Fact]
    public async Task CoEvidenceNhungKhongCoInternet_VanTraLoiTuKenhNoiBo()
    {
        var agent = fixture.BuildFresh(withVerification: false);   // evidence bật, internet + kiểm chứng tắt

        var response = await agent.HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.Equal("evidence", response.ToolUsed);
        Assert.True(response.Evidence!.HasEvidence);
        Assert.Empty(response.Evidence.FromKind(SourceKind.Internet));
        Assert.False(response.UsedUnverifiedInternetData);
    }

    [Fact]
    public async Task AgentTonTrongHuy()
    {
        var clock = new ManualClock(TrainedAgent.FixedNow);
        var slow = new InternetSearchTool(new SearchService(
            new SlowProvider(TimeSpan.FromSeconds(30)),
            clock: clock,
            options: new SearchServiceOptions { FetchPages = false }));

        var agent = fixture.BuildFresh(clock, slow, withVerification: false);
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => agent.HandleAsync("tìm trên mạng thông tin về SSRF", cts.Token));
    }

    [Fact]
    public async Task ChuaCauHinhEvidence_ThiGoiCollectEvidenceSeBaoLoiRoRang()
    {
        var agent = fixture.BuildFresh(withEvidence: false, withVerification: false);

        Assert.False(agent.HasEvidenceCollector);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => agent.CollectEvidenceAsync("SSRF là gì", CancellationToken.None));

        Assert.Contains("EvidenceCollector", ex.Message);
    }
}

public class InternetSearchToolTests
{
    private static readonly DateTimeOffset Start = new(2026, 1, 1, 9, 0, 0, TimeSpan.Zero);

    [Fact]
    public void TenVaMoTaNoiRoDayLaDuLieuChuaKiemChung()
    {
        var tool = InternetStackFactory.BuildOffline(new ManualClock(Start));

        Assert.Equal("internet", tool.Name);
        Assert.Contains("chưa kiểm chứng", tool.Description);
    }

    [Fact]
    public async Task GiuLaiBaoCaoLuotGanNhat()
    {
        var tool = InternetStackFactory.BuildOffline(new ManualClock(Start));

        Assert.Null(tool.LastReport);

        await tool.ExecuteAsync("SSRF là gì", CancellationToken.None);

        Assert.NotNull(tool.LastReport);
        Assert.Equal("SSRF là gì", tool.LastReport.Question);
    }

    [Fact]
    public async Task ChanSoNguonNeuTrongCauTraLoi()
    {
        var tool = new InternetSearchTool(new SearchService(
            SsrfCorpus.Build(5),
            clock: new ManualClock(Start),
            options: new SearchServiceOptions { FetchPages = false, MaxSources = 5 }))
        {
            MaxSourcesInAnswer = 2,
        };

        var result = await tool.ExecuteAsync("ssrf", CancellationToken.None);

        Assert.True(result.Success);
        Assert.Equal(2, result.Output.Split("  • ").Length - 1);
        Assert.Equal(5, tool.LastReport!.Sources.Count);
    }

    [Fact]
    public async Task CanhBaoDuocDuaVaoCauTraLoi()
    {
        var provider = new CompositeSearchProvider(
            new FailingProvider("google"),
            new StaticSearchProvider("chay-duoc")
                .Add("SSRF", "https://owasp.org/ssrf", "ssrf là lỗ hổng máy chủ nội bộ"));

        var tool = new InternetSearchTool(new SearchService(
            provider,
            clock: new ManualClock(Start),
            options: new SearchServiceOptions { FetchPages = false }));

        var result = await tool.ExecuteAsync("ssrf", CancellationToken.None);

        Assert.True(result.Success);
        Assert.Contains("Lưu ý", result.Output);
        Assert.Contains("google", result.Output);
    }

    /// <summary>PHASE 13 công khai dịch vụ bên dưới để EvidenceCollector dùng trực tiếp.</summary>
    [Fact]
    public void CongKhaiDichVuTraCuuChoTangEvidence()
    {
        var tool = InternetStackFactory.BuildOffline(new ManualClock(Start));

        Assert.NotNull(tool.Service);
    }
}
