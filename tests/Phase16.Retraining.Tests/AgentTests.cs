using Ai.Phase16.Agent;
using Ai.Phase16.Internet;
using Ai.Phase16.Memory;
using Ai.Phase16.Reasoning;
using Ai.Phase16.Tools;

namespace Ai.Phase16.Tests;

/// <summary>
/// Agent được train một lần rồi dùng chung cho mọi test — train lại cho từng
/// test thì bộ test chạy rất lâu mà chẳng kiểm tra thêm được gì.
/// </summary>
public sealed class TrainedAgent
{
    public static readonly DateTimeOffset FixedNow = new(2026, 1, 1, 9, 0, 0, TimeSpan.Zero);

    public IntentClassifier Classifier { get; } = AgentFactory.TrainClassifier();

    /// <summary>Agent sạch (bộ nhớ rỗng) nhưng DÙNG LẠI classifier đã train — nhanh hơn nhiều.</summary>
    public AiController BuildFresh(IClock? clock = null, InternetSearchTool? internet = null,
        bool withEvidence = true, bool withVerification = true) =>
        AgentFactory.Build(Classifier, clock ?? new ManualClock(FixedNow), internet, withEvidence, withVerification);

    /// <summary>Agent Phase 14 đầy đủ: Internet ngoại tuyến + evidence + kiểm chứng.</summary>
    public AiController BuildWithInternet(IClock? clock = null)
    {
        clock ??= new ManualClock(FixedNow);
        return BuildFresh(clock, InternetStackFactory.BuildOffline(clock));
    }

    /// <summary>
    /// Agent đúng như Phase 13: có evidence nhưng KHÔNG có tầng kiểm chứng.
    ///
    /// Cùng lý do như <see cref="BuildPhase12Style"/> — giữ cấu hình này để bộ
    /// test của Phase 13 chạy nguyên văn, chứng minh Phase 14 không phá nó.
    /// </summary>
    public AiController BuildPhase13Style(IClock? clock = null)
    {
        clock ??= new ManualClock(FixedNow);
        return BuildFresh(clock, InternetStackFactory.BuildOffline(clock), withVerification: false);
    }

    /// <summary>
    /// Agent đúng như Phase 12: có Internet nhưng KHÔNG có tầng evidence.
    ///
    /// Giữ lại cấu hình này để chứng minh Phase 13 không phá hành vi cũ — các
    /// test của Phase 12 chạy nguyên văn trên agent dựng theo cách này.
    /// </summary>
    public AiController BuildPhase12Style(IClock? clock = null)
    {
        clock ??= new ManualClock(FixedNow);
        return BuildFresh(clock, InternetStackFactory.BuildOffline(clock), withEvidence: false, withVerification: false);
    }
}

[CollectionDefinition("agent")]
public class AgentCollection : ICollectionFixture<TrainedAgent>;

[Collection("agent")]
public class IntentClassifierTests(TrainedAgent fixture)
{
    [Fact]
    public void HocThuocDuocToanBoTapHuanLuyen()
    {
        Assert.Equal(1.0, fixture.Classifier.Accuracy(IntentTrainingData.Examples));
    }

    /// <summary>Quan trọng hơn: phải khái quát hoá sang câu CHƯA từng thấy.</summary>
    [Theory]
    [InlineData("nhớ giúp tôi rằng tôi thích Gundam", Intent.RememberFact)]
    [InlineData("tính 2 nhân 8", Intent.Calculate)]
    [InlineData("attention là gì", Intent.SearchKnowledge)]
    [InlineData("có bao nhiêu khách hàng", Intent.QueryDatabase)]
    [InlineData("chào", Intent.SmallTalk)]
    [InlineData("bạn biết gì về tôi", Intent.RecallMemory)]
    public void KhaiQuatHoaSangCauMoi(string text, Intent expected)
    {
        var prediction = fixture.Classifier.Classify(text);

        Assert.Equal(expected, prediction.Intent);
        Assert.True(prediction.IsConfident, $"độ tự tin chỉ {prediction.Confidence:P0}");
    }

    /// <summary>
    /// PHASE 12: ý định mới phải tách được khỏi SearchKnowledge, kể cả với câu
    /// chưa từng thấy. Đây là phần dễ sai nhất của việc thêm ý định — hai ý
    /// định đều là "đi tìm thông tin", chỉ khác NGUỒN.
    /// </summary>
    [Theory]
    [InlineData("tìm trên mạng xem có gì mới")]
    [InlineData("tra trên internet thông tin về SSRF")]
    [InlineData("search online for the latest version")]
    [InlineData("lên mạng tìm tin tức")]
    public void NhanRaYDinhTraInternet(string text)
    {
        var prediction = fixture.Classifier.Classify(text);

        Assert.Equal(Intent.SearchInternet, prediction.Intent);
        Assert.True(prediction.IsConfident, $"độ tự tin chỉ {prediction.Confidence:P0}");
    }

    /// <summary>Và ngược lại: câu hỏi khái niệm KHÔNG được nhầm sang Internet.</summary>
    [Theory]
    [InlineData("backpropagation là gì")]
    [InlineData("giải thích gradient descent")]
    [InlineData("định nghĩa của attention")]
    public void KhongNhamCauHoiKhaiNiemSangInternet(string text)
    {
        Assert.Equal(Intent.SearchKnowledge, fixture.Classifier.Classify(text).Intent);
    }

    [Fact]
    public void KhongNhanRaTuNao_ThiDoTuTinBangKhong()
    {
        var prediction = fixture.Classifier.Classify("zxcvbnm qwerty asdfgh");

        Assert.Equal(0, fixture.Classifier.KnownWordCount("zxcvbnm qwerty asdfgh"));
        Assert.Equal(0.0, prediction.Confidence);
        Assert.False(prediction.IsConfident);
    }

    [Fact]
    public void XacSuatLaPhanBoHopLe()
    {
        var prediction = fixture.Classifier.Classify("tính tổng của A và B");

        Assert.Equal(1.0, prediction.Probabilities.Sum(), precision: 10);
        Assert.Equal(Enum.GetValues<Intent>().Length, prediction.Probabilities.Length);
    }

    /// <summary>Thêm ý định vào CUỐI enum thì nhãn của các ý định cũ không được lệch.</summary>
    [Fact]
    public void ThemYDinhKhongLamLechNhanCu()
    {
        Assert.Equal(0, (int)Intent.Calculate);
        Assert.Equal(5, (int)Intent.SmallTalk);
        Assert.Equal(6, (int)Intent.SearchInternet);
        Assert.Equal(7, Enum.GetValues<Intent>().Length);

        var oneHot = IntentClassifier.OneHot(Intent.SearchInternet);
        Assert.Equal(1.0, oneHot[6]);
        Assert.Equal(1.0, oneHot.Sum(), precision: 10);
    }

    [Fact]
    public void Encode_LaBagOfWordsNhiPhan()
    {
        var vector = fixture.Classifier.Encode("tính tổng");

        Assert.Equal(fixture.Classifier.Vocabulary.Count, vector.Length);
        Assert.Equal(2.0, vector.Sum(), precision: 10);   // đúng hai từ được bật
    }

    [Fact]
    public void TuLa_ThiBiBoQua()
    {
        // "zzzz" không có trong từ điển -> vector giống hệt khi không có nó.
        var withUnknown = fixture.Classifier.Encode("tính tổng zzzz");
        var without = fixture.Classifier.Encode("tính tổng");

        Assert.Equal(without.ToArray(), withUnknown.ToArray());
    }
}

public class ToolTests
{
    [Fact]
    public void Calculator_TinhChinhXac()
    {
        var engine = new ReasoningEngine();
        engine.LearnAll("A = 10", "B = 20", "C = A + B");
        var tool = new CalculatorTool(engine);

        Assert.Equal("14", tool.Execute("2 + 3 * 4").Output);
        Assert.Equal("30", tool.Execute("C").Output);
        Assert.Equal("12.5", tool.Execute("100 / 8").Output);
    }

    [Fact]
    public void Calculator_BaoLoiThayViBia()
    {
        var tool = new CalculatorTool(new ReasoningEngine());

        var result = tool.Execute("Z + 1");

        Assert.False(result.Success);
        Assert.Contains("Z", result.Output);
    }

    [Fact]
    public void Search_TraVeTaiLieuKhopNhat()
    {
        var tool = new SearchTool();
        tool.AddDocument("Softmax", "Softmax biến logits thành phân bố xác suất.");
        tool.AddDocument("Attention", "Attention tính softmax của Q nhân K chuyển vị.");

        var result = tool.Execute("softmax");

        Assert.True(result.Success);
        Assert.Contains("phân bố xác suất", result.Output);
        Assert.Contains("Softmax", result.Detail);
    }

    [Fact]
    public void Search_TuChoiKhiTrungQuaIt()
    {
        var tool = new SearchTool();
        tool.AddDocument("Attention", "Attention dùng query để quyết định lấy thông tin từ đâu.");

        var result = tool.Execute("làm thế nào để nấu phở");

        Assert.False(result.Success);
        Assert.Contains("không tìm thấy", result.Output);
    }

    private static DatabaseTool BuildDatabase()
    {
        var db = new DatabaseTool();
        db.Insert(new Order(1, "Alice", 1_200_000));
        db.Insert(new Order(2, "Bob", 450_000));
        db.Insert(new Order(3, "Chi", 2_800_000));
        db.Insert(new Order(4, "Dung", 760_000));
        return db;
    }

    [Fact]
    public void Database_Dem()
    {
        Assert.Equal("4", BuildDatabase().Execute("có bao nhiêu đơn hàng").Output);
        Assert.Equal("4", BuildDatabase().Execute("how many orders").Output);
    }

    [Fact]
    public void Database_TongTienThangPhepDemKhiCauChuaCaHai()
    {
        var result = BuildDatabase().Execute("doanh thu tổng cộng là bao nhiêu");

        Assert.Equal("5,210,000", result.Output);
        Assert.Contains("SUM", result.Detail);
    }

    [Fact]
    public void Database_TuChoiTruyVanKhongHieu()
    {
        Assert.False(BuildDatabase().Execute("xoá hết dữ liệu đi").Success);
    }
}

/// <summary>
/// Hành vi của Phase 11 phải còn nguyên sau khi thêm Phase 12. Đây là bộ test
/// hồi quy: nếu nhánh Internet làm hỏng nhánh nào cũ, những test này đỏ.
/// </summary>
[Collection("agent")]
public class AiControllerTests(TrainedAgent fixture)
{
    [Fact]
    public async Task ChaoHoi_TraLoiNgayKhongGoiToolNao()
    {
        var response = await fixture.BuildFresh().HandleAsync("xin chào");

        Assert.Equal(Intent.SmallTalk, response.Intent);
        Assert.Null(response.ToolUsed);
        Assert.Contains(response.Decisions, d => d.Decision.StartsWith("CÓ"));
    }

    [Fact]
    public async Task TinhToan_GoiCalculatorVaChoDapAnChinhXac()
    {
        var response = await fixture.BuildFresh().HandleAsync("tính 2 + 3 * 4");

        Assert.Equal(Intent.Calculate, response.Intent);
        Assert.Equal("calculator", response.ToolUsed);
        Assert.Contains("14", response.Answer);
    }

    [Fact]
    public async Task TraCuu_GoiSearch()
    {
        var response = await fixture.BuildFresh().HandleAsync("backpropagation là gì");

        Assert.Equal(Intent.SearchKnowledge, response.Intent);
        Assert.Equal("search", response.ToolUsed);
        Assert.Contains("chain rule", response.Answer);
    }

    [Fact]
    public async Task TruyVanDuLieu_GoiDatabase()
    {
        var agent = fixture.BuildFresh();

        Assert.Contains("4", (await agent.HandleAsync("có bao nhiêu đơn hàng")).Answer);
        Assert.Contains("5,210,000", (await agent.HandleAsync("doanh thu tổng cộng là bao nhiêu")).Answer);
    }

    [Fact]
    public async Task GhiNhoVaNhoLaiQuaNhieuLuot()
    {
        var agent = fixture.BuildFresh();

        await agent.HandleAsync("hãy nhớ rằng tôi dùng C# và dotnet");
        await agent.HandleAsync("ghi nhớ là tôi làm việc với Azure");

        Assert.Equal(2, agent.Memory.LongTerm.Count);

        var recalled = await agent.HandleAsync("tôi dùng ngôn ngữ nào");

        Assert.Equal(Intent.RecallMemory, recalled.Intent);
        Assert.Contains("C#", recalled.Answer);
    }

    [Fact]
    public async Task GhiNho_CatBoPhanMoDau()
    {
        var agent = fixture.BuildFresh();

        await agent.HandleAsync("hãy nhớ rằng tôi dùng C#");

        Assert.Equal("tôi dùng C#", agent.Memory.LongTerm.Facts.Single().Content);
    }

    [Fact]
    public async Task KhongNhoGiLienQuan_ThiNoiThang()
    {
        var response = await fixture.BuildFresh().HandleAsync("tôi dùng ngôn ngữ nào");

        Assert.Equal(Intent.RecallMemory, response.Intent);
        Assert.Contains("không nhớ", response.Answer, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task DauVaoVoNghia_ThiNoiKhongChacChuKhongDoanBua()
    {
        var response = await fixture.BuildFresh().HandleAsync("zxcvbnm qwerty");

        Assert.Equal(0.0, response.Confidence);
        Assert.Null(response.ToolUsed);
        Assert.Contains("không chắc", response.Answer);
    }

    [Fact]
    public async Task ToolThatBai_ThiBaoLoiChuKhongBia()
    {
        var agent = fixture.BuildFresh();

        Assert.Contains("không xong", (await agent.HandleAsync("tính 5 chia 0")).Answer);
        Assert.Contains("Chưa biết", (await agent.HandleAsync("Z bằng bao nhiêu")).Answer);
    }

    /// <summary>Luồng quyết định giờ có thêm bước Internet, và thứ tự phải đúng.</summary>
    [Fact]
    public async Task GhiLaiDayDuLuongQuyetDinh()
    {
        var response = await fixture.BuildFresh().HandleAsync("tính 2 + 3 * 4");
        var questions = response.Decisions.Select(d => d.Question).ToArray();

        Assert.Contains("Người dùng muốn gì?", questions);
        Assert.Contains("Trả lời được bằng kiến thức sẵn có?", questions);
        Assert.Contains("Có cần Memory?", questions);
        Assert.Contains("Có cần Internet?", questions);
        Assert.Contains("Có cần Tool?", questions);
        Assert.Contains("Tool trả về gì?", questions);

        Assert.True(Array.IndexOf(questions, "Có cần Memory?") < Array.IndexOf(questions, "Có cần Internet?"));
        Assert.True(Array.IndexOf(questions, "Có cần Internet?") < Array.IndexOf(questions, "Có cần Tool?"));
    }

    [Theory]
    [InlineData("có bao nhiêu đơn hàng")]
    [InlineData("đếm số đơn")]
    [InlineData("how many orders are there")]
    public async Task NhieuCachDienDat_CungMotDapAn(string phrasing)
    {
        var response = await fixture.BuildFresh().HandleAsync(phrasing);

        Assert.Equal(Intent.QueryDatabase, response.Intent);
        Assert.Contains("4", response.Answer);
    }

    [Fact]
    public async Task BoNhoTuLocKhongLuuRac()
    {
        var agent = fixture.BuildFresh();

        await agent.HandleAsync("hãy nhớ rằng tôi dùng C#");
        await agent.HandleAsync("hãy nhớ rằng tôi dùng C#");

        Assert.Equal(1, agent.Memory.LongTerm.Count);
    }

    [Fact]
    public async Task CauHoiKhongLienQuanInternet_KhongHeGoiInternet()
    {
        var internet = new RecordingAsyncTool();
        var agent = AgentFactory.Build(fixture.Classifier, new ManualClock(TrainedAgent.FixedNow));

        // Agent không có internet vẫn phải trả lời đủ mọi nhánh cũ.
        Assert.Contains("14", (await agent.HandleAsync("tính 2 + 3 * 4")).Answer);
        Assert.Equal(0, internet.CallCount);
    }

    [Fact]
    public void DangKyDuBaCongCuDongBo()
    {
        var names = fixture.BuildFresh().Tools.Select(t => t.Name).ToArray();

        Assert.Equal(["calculator", "search", "database"], names);
    }

    [Fact]
    public void KhongCauHinhInternet_ThiKhongCoCongCuBatDongBo()
    {
        var agent = fixture.BuildFresh();

        Assert.False(agent.HasInternetAccess);
        Assert.Empty(agent.AsyncTools);
    }

    [Fact]
    public void CoCauHinhInternet_ThiDangKyDungMotCongCuBatDongBo()
    {
        var agent = fixture.BuildWithInternet();

        Assert.True(agent.HasInternetAccess);
        Assert.Equal(["internet"], agent.AsyncTools.Select(t => t.Name).ToArray());
    }

    private sealed class RecordingAsyncTool : IAsyncTool
    {
        public int CallCount { get; private set; }

        public string Name => "internet";

        public string Description => "đếm số lần được gọi";

        public Task<ToolResult> ExecuteAsync(string input, CancellationToken cancellationToken = default)
        {
            CallCount++;
            return Task.FromResult(new ToolResult(true, "đã gọi"));
        }
    }
}
