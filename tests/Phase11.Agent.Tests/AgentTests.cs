using Ai.Phase11.Agent;
using Ai.Phase11.Memory;
using Ai.Phase11.Reasoning;
using Ai.Phase11.Tools;

namespace Ai.Phase11.Tests;

/// <summary>
/// Agent được train một lần rồi dùng chung cho mọi test — train lại cho từng
/// test thì bộ test chạy rất lâu mà chẳng kiểm tra thêm được gì.
/// </summary>
public sealed class TrainedAgent
{
    public IntentClassifier Classifier { get; } = AgentFactory.TrainClassifier();

    /// <summary>Agent sạch (bộ nhớ rỗng) nhưng DÙNG LẠI classifier đã train — nhanh hơn nhiều.</summary>
    public AiController BuildFresh(IClock? clock = null) =>
        AgentFactory.Build(Classifier, clock ?? new ManualClock(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero)));
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
    public void Calculator_KemDauVetSuyLuan()
    {
        var engine = new ReasoningEngine();
        engine.LearnAll("A = 10", "B = 20", "C = A + B");

        var result = new CalculatorTool(engine).Execute("C");

        Assert.True(result.Success);
        Assert.Contains("A + B", result.Detail);
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

    /// <summary>
    /// Trùng một từ vặt không được tính là tìm thấy — nếu không, agent sẽ
    /// trả lời rất tự tin bằng một tài liệu chẳng liên quan.
    /// </summary>
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

    /// <summary>
    /// "doanh thu tổng cộng là bao nhiêu" chứa cả từ của phép TỔNG lẫn phép
    /// ĐẾM. Phép cụ thể hơn phải thắng — đây từng là một bug thật.
    /// </summary>
    [Fact]
    public void Database_TongTienThangPhepDemKhiCauChuaCaHai()
    {
        var result = BuildDatabase().Execute("doanh thu tổng cộng là bao nhiêu");

        Assert.Equal("5,210,000", result.Output);
        Assert.Contains("SUM", result.Detail);
    }

    [Fact]
    public void Database_LonNhat()
    {
        var result = BuildDatabase().Execute("đơn hàng lớn nhất");

        Assert.Contains("Chi", result.Output);
        Assert.Contains("2,800,000", result.Output);
    }

    [Fact]
    public void Database_LietKe()
    {
        var result = BuildDatabase().Execute("liệt kê tất cả đơn hàng");

        Assert.Contains("Alice", result.Output);
        Assert.Contains("Dung", result.Output);
    }

    [Fact]
    public void Database_TuChoiTruyVanKhongHieu()
    {
        Assert.False(BuildDatabase().Execute("xoá hết dữ liệu đi").Success);
    }
}

[Collection("agent")]
public class AiControllerTests(TrainedAgent fixture)
{
    [Fact]
    public void ChaoHoi_TraLoiNgayKhongGoiToolNao()
    {
        var response = fixture.BuildFresh().Handle("xin chào");

        Assert.Equal(Intent.SmallTalk, response.Intent);
        Assert.Null(response.ToolUsed);
        Assert.Contains(response.Decisions, d => d.Decision.StartsWith("CÓ"));
    }

    [Fact]
    public void TinhToan_GoiCalculatorVaChoDapAnChinhXac()
    {
        var response = fixture.BuildFresh().Handle("tính 2 + 3 * 4");

        Assert.Equal(Intent.Calculate, response.Intent);
        Assert.Equal("calculator", response.ToolUsed);
        Assert.Contains("14", response.Answer);
    }

    [Fact]
    public void TraCuu_GoiSearch()
    {
        var response = fixture.BuildFresh().Handle("backpropagation là gì");

        Assert.Equal(Intent.SearchKnowledge, response.Intent);
        Assert.Equal("search", response.ToolUsed);
        Assert.Contains("chain rule", response.Answer);
    }

    [Fact]
    public void TruyVanDuLieu_GoiDatabase()
    {
        var agent = fixture.BuildFresh();

        Assert.Contains("4", agent.Handle("có bao nhiêu đơn hàng").Answer);
        Assert.Contains("5,210,000", agent.Handle("doanh thu tổng cộng là bao nhiêu").Answer);
    }

    [Fact]
    public void GhiNhoVaNhoLaiQuaNhieuLuot()
    {
        var agent = fixture.BuildFresh();

        agent.Handle("hãy nhớ rằng tôi dùng C# và dotnet");
        agent.Handle("ghi nhớ là tôi làm việc với Azure");

        Assert.Equal(2, agent.Memory.LongTerm.Count);

        var recalled = agent.Handle("tôi dùng ngôn ngữ nào");

        Assert.Equal(Intent.RecallMemory, recalled.Intent);
        Assert.Contains("C#", recalled.Answer);
    }

    [Fact]
    public void GhiNho_CatBoPhanMoDau()
    {
        var agent = fixture.BuildFresh();

        agent.Handle("hãy nhớ rằng tôi dùng C#");

        Assert.Equal("tôi dùng C#", agent.Memory.LongTerm.Facts.Single().Content);
    }

    [Fact]
    public void KhongNhoGiLienQuan_ThiNoiThang()
    {
        var response = fixture.BuildFresh().Handle("tôi dùng ngôn ngữ nào");

        Assert.Equal(Intent.RecallMemory, response.Intent);
        Assert.Contains("không nhớ", response.Answer, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DauVaoVoNghia_ThiNoiKhongChacChuKhongDoanBua()
    {
        var response = fixture.BuildFresh().Handle("zxcvbnm qwerty");

        Assert.Equal(0.0, response.Confidence);
        Assert.Null(response.ToolUsed);
        Assert.Contains("không chắc", response.Answer);
    }

    [Fact]
    public void ToolThatBai_ThiBaoLoiChuKhongBia()
    {
        var agent = fixture.BuildFresh();

        var divideByZero = agent.Handle("tính 5 chia 0");
        Assert.Contains("không xong", divideByZero.Answer);

        var unknown = agent.Handle("Z bằng bao nhiêu");
        Assert.Contains("Chưa biết", unknown.Answer);
    }

    /// <summary>Đúng luồng quyết định của đề bài.</summary>
    [Fact]
    public void GhiLaiDayDuLuongQuyetDinh()
    {
        var response = fixture.BuildFresh().Handle("tính 2 + 3 * 4");
        var questions = response.Decisions.Select(d => d.Question).ToArray();

        Assert.Contains("Người dùng muốn gì?", questions);
        Assert.Contains("Trả lời được bằng kiến thức sẵn có?", questions);
        Assert.Contains("Có cần Memory?", questions);
        Assert.Contains("Có cần Tool?", questions);
        Assert.Contains("Tool trả về gì?", questions);
    }

    /// <summary>
    /// Điểm cốt lõi của kiến trúc lai: nhiều cách diễn đạt khác nhau đều
    /// dẫn tới CÙNG một công cụ và CÙNG một đáp án chính xác.
    /// </summary>
    [Theory]
    [InlineData("có bao nhiêu đơn hàng")]
    [InlineData("đếm số đơn")]
    [InlineData("how many orders are there")]
    public void NhieuCachDienDat_CungMotDapAn(string phrasing)
    {
        var response = fixture.BuildFresh().Handle(phrasing);

        Assert.Equal(Intent.QueryDatabase, response.Intent);
        Assert.Contains("4", response.Answer);
    }

    [Fact]
    public void BoNhoTuLocKhongLuuRac()
    {
        var agent = fixture.BuildFresh();

        agent.Handle("hãy nhớ rằng tôi dùng C#");
        agent.Handle("hãy nhớ rằng tôi dùng C#");

        Assert.Equal(1, agent.Memory.LongTerm.Count);
    }

    [Fact]
    public void DangKyDuBaCongCu()
    {
        var names = fixture.BuildFresh().Tools.Select(t => t.Name).ToArray();

        Assert.Equal(["calculator", "search", "database"], names);
    }
}
