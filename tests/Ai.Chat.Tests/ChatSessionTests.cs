using Ai.Chat;
using Ai.Phase16.Agent;
using Ai.Phase16.Feedback;
using Ai.Phase16.Grounding;
using Ai.Phase16.Verification;

namespace Ai.Chat.Tests;

/// <summary>
/// Một phiên chat dùng chung cho các test CHỈ ĐỌC — dựng phiên tốn vài giây vì
/// phải train bộ phân loại, và train lại cho từng test thì bộ test chậm mà
/// chẳng kiểm thêm được gì.
/// </summary>
public sealed class SharedSession
{
    public ChatSession Session { get; } = ChatSession.Start();
}

[CollectionDefinition("chat")]
public class ChatCollection : ICollectionFixture<SharedSession>;

[Collection("chat")]
public class ChatRoutingTests(SharedSession fixture)
{
    private ChatSession Session => fixture.Session;

    [Fact]
    public async Task CauHoiDuocDinhTuyenThiDiDuongCongCu()
    {
        var turn = await Session.AskAsync("hãy nhớ tôi dùng C#");

        Assert.False(turn.UsedFallback);
        Assert.Equal(Intent.RememberFact, turn.Response.Intent);
        Assert.True(turn.RoutedAs!.IsConfident);
    }

    /// <summary>
    /// "SSRF là gì" chỉ còn một từ khoá sau khi bỏ từ dừng, và từ đó không có
    /// trong từ điển 164 từ — nên bộ định tuyến mù hoàn toàn. Hệ thống vẫn phải
    /// trả lời được, vì kho kiến thức có tài liệu về SSRF.
    /// </summary>
    [Fact]
    public async Task KhongDinhTuyenDuocThiVanTraLoiDuocQuaDuongOngKiemChung()
    {
        var turn = await Session.AskAsync("SSRF là gì");

        Assert.True(turn.UsedFallback);
        Assert.Equal(0, Session.Classifier.KnownWordCount("SSRF là gì"));
        Assert.NotNull(turn.Response.Grounded);
        Assert.True(turn.Response.Grounded!.IsAnswer);
        Assert.Contains(turn.Response.Grounded.CitedEvidence, e => e.Kind == SourceKind.Knowledge);
    }

    /// <summary>
    /// Đường lui KHÔNG được bịa: câu ngoài phạm vi kiến thức phải ra
    /// <see cref="AnswerVerdict.Insufficient"/>, vì đường ống Phase 14 tự có
    /// chốt riêng.
    /// </summary>
    [Fact]
    public async Task DuongLuiKhongBiaCauTraLoiKhiKhongDuCanCu()
    {
        var turn = await Session.AskAsync("cách nấu phở bò Hà Nội");

        Assert.True(turn.UsedFallback);
        Assert.Equal(AnswerVerdict.Insufficient, turn.Response.Grounded!.Verdict);
        Assert.False(turn.Response.Grounded.IsAnswer);
    }

    /// <summary>Biểu thức số học không còn từ khoá nào, nhưng engine ký hiệu của Phase 9 vẫn tính được.</summary>
    [Fact]
    public async Task BieuThucSoHocDiQuaDuongLuiVaVanTinhDung()
    {
        var turn = await Session.AskAsync("2 + 3 * 4");

        Assert.True(turn.UsedFallback);
        Assert.Contains("14", turn.Response.Answer, StringComparison.Ordinal);
    }

    [Fact]
    public async Task MoiLuotDeuDuocCapIdDeChamDuoc()
    {
        var turn = await Session.AskAsync("có bao nhiêu đơn hàng");

        Assert.NotNull(turn.Record.Id);
        Assert.Same(turn, Session.LastTurn);
        Assert.Equal(turn.Question, turn.Record.Question);
    }
}

/// <summary>
/// Test làm ĐỔI trạng thái (phản hồi, tập dữ liệu, model) nên mỗi test tự dựng
/// phiên riêng — dùng chung thì thứ tự chạy quyết định kết quả.
/// </summary>
public class ChatFeedbackLoopTests
{
    /// <summary>
    /// TEST CHỐNG TÁI DIỄN cho một lỗi thật.
    ///
    /// Bản đầu của màn hình chat gọi <c>HandleAsync</c> trước rồi mới đi đường
    /// lui. <c>HandleAsync</c> ghi lượt hội thoại vào ngữ cảnh TRƯỚC khi đường
    /// lui chạy, nên câu hỏi vừa gõ trở thành căn cứ cho chính nó:
    ///
    ///     [SỰ THẬT] SSRF là gì
    ///       ↳ bạn vừa nói ở lượt #1
    ///
    /// Đúng kiểu lập luận vòng tròn mà Phase 14 đã chặn. Test này khẳng định
    /// câu trả lời không dẫn CHÍNH câu hỏi của lượt đó làm căn cứ.
    ///
    /// Phiên phải SẠCH: nếu câu đó đã được hỏi ở lượt trước thì ngữ cảnh chứa
    /// nó một cách hợp lệ, và test sẽ đỏ vì một lý do không phải lỗi. (Chuyện
    /// này đã xảy ra thật khi test dùng chung phiên với các test khác.)
    /// </summary>
    [Fact]
    public async Task CauTraLoiKhongDanChinhCauHoiVuaGoLamCanCu()
    {
        const string question = "SSRF là gì";

        var session = ChatSession.Start();
        var turn = await session.AskAsync(question);

        Assert.True(turn.UsedFallback);
        Assert.DoesNotContain(turn.Response.Grounded!.CitedEvidence,
            e => e.Kind == SourceKind.Conversation
                 && e.Content.Contains(question, StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ChuaHoiGiThiKhongChamDuoc()
    {
        var session = ChatSession.Start();

        var result = session.Rate(FeedbackRating.Correct);

        Assert.False(result.Accepted);
        Assert.Contains("chưa có câu trả lời", result.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public async Task MotLuotChiChamDuocMotLan()
    {
        var session = ChatSession.Start();
        await session.AskAsync("chào bạn");

        Assert.True(session.Rate(FeedbackRating.Correct).Accepted);
        Assert.True(session.LastTurnRated);
        Assert.False(session.Rate(FeedbackRating.Incorrect).Accepted);
    }

    /// <summary>
    /// Vòng đầy đủ ngay trong một phiên chat: hỏi → chấm sai kèm nhãn đúng →
    /// /hoc sinh ra phiên bản dữ liệu mới. Và trọng số CHƯA đổi ở bước này.
    /// </summary>
    [Fact]
    public async Task PhanHoiSinhRaPhienBanDuLieuMoiMaKhongDoiTrongSo()
    {
        var session = ChatSession.Start();

        var turn = await session.AskAsync("cho tôi biết số liệu đơn hàng");
        var before = session.Classifier.Classify(turn.Question);

        session.Rate(FeedbackRating.Incorrect, Intent.SearchInternet);

        var report = session.Learn();

        Assert.True(report.ProducedNewVersion);
        Assert.Equal(2, report.NewVersion!.Version);
        Assert.Single(report.NewVersion.Added);

        var after = session.Classifier.Classify(turn.Question);

        Assert.Equal(before.Intent, after.Intent);
        Assert.Equal(before.Confidence, after.Confidence, 12);
        Assert.Null(session.Registry.Active);
    }

    [Fact]
    public async Task ChamDungThiKhongSinhRaCaLoiNao()
    {
        var session = ChatSession.Start();
        await session.AskAsync("chào bạn");

        session.Rate(FeedbackRating.Correct);

        var report = session.Learn();

        Assert.Empty(report.ErrorCases);
        Assert.False(report.ProducedNewVersion);
    }

    /// <summary>
    /// Huấn luyện lại từ màn hình chat: model mới chỉ được áp dụng khi vượt cửa
    /// đánh giá, và sổ đăng ký ghi lại đủ để truy về phiên bản dữ liệu.
    /// </summary>
    [Fact]
    public async Task TrainLaiChiApDungModelKhiVuotCuaDanhGia()
    {
        var session = ChatSession.Start();
        await session.AskAsync("chào bạn");

        var report = await session.RetrainAsync();

        Assert.True(report.Deployed, $"cửa đánh giá từ chối: {report.Decision}");
        Assert.NotNull(session.Registry.Active);
        Assert.Equal(report.Candidate.Version, session.Registry.Active!.Version);
        Assert.Equal(session.Status().DatasetFingerprint, report.Candidate.DatasetFingerprint);
        Assert.Equal(report.Golden.Total, report.Golden.Correct);
    }

    /// <summary>Cấu hình tồi bị từ chối, và agent giữ nguyên model đang chạy.</summary>
    [Fact]
    public async Task CauHinhToiBiTuChoiVaAgentGiuNguyenModel()
    {
        var session = ChatSession.Start();
        await session.AskAsync("chào bạn");

        var good = await session.RetrainAsync();
        Assert.True(good.Deployed);

        var bad = await session.RetrainAsync(new Phase16.Retraining.TrainingConfig { Epochs = 1 });

        Assert.False(bad.Deployed);
        Assert.Equal(good.Candidate.Version, session.Registry.Active!.Version);
        Assert.NotEmpty(bad.Decision.Blockers);
    }

    [Fact]
    public async Task BatDauHoiThoaiMoiThiQuenKyUcMaGiuModel()
    {
        var session = ChatSession.Start();
        await session.AskAsync("hãy nhớ tôi dùng C#");

        Assert.Equal(1, session.Status().MemoryCount);

        session.ResetConversation();

        Assert.Equal(0, session.Status().MemoryCount);
        Assert.Empty(session.Turns);
        Assert.Equal(1, session.Status().DatasetVersion);
    }

    [Fact]
    public async Task TrangThaiPhanAnhDungSoDaDo()
    {
        var session = ChatSession.Start();
        await session.AskAsync("chào bạn");
        session.Rate(FeedbackRating.Correct);

        var status = session.Status();

        Assert.Equal(1, status.TurnCount);
        Assert.Equal(1, status.FeedbackCount);
        Assert.Equal(2759, status.ParameterCount);
        Assert.True(status.KnowledgeCount > 0);
    }
}
