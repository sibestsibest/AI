using Ai.Phase15.Agent;
using Ai.Phase15.Feedback;
using Ai.Phase15.Memory;
using Ai.Phase15.Verification;

namespace Ai.Phase15.Tests;

/// <summary>Dựng sẵn câu trả lời của Phase 14 để test vòng phản hồi tách khỏi khâu sinh câu trả lời.</summary>
internal static class Ans
{
    public static readonly DateTimeOffset Now = new(2026, 1, 1, 9, 0, 0, TimeSpan.Zero);

    public static GroundedAnswer Grounded(string question = "SSRF là gì",
        AnswerVerdict verdict = AnswerVerdict.Answered, double confidence = 0.85,
        string? text = null, params string[] evidenceIds) =>
        new()
        {
            Question = question,
            Text = text ?? $"trả lời cho \"{question}\"",
            Verdict = verdict,
            Confidence = confidence,
            AnsweredAt = Now,
            Summary = new ReasoningSummary
            {
                Decisions = ["gom được 2 căn cứ từ 2 kênh", $"kết luận: {verdict}"],
                EvidenceIds = evidenceIds,
            },
        };

    public static ManualClock Clock() => new(Now);
}

public class AnswerJournalTests
{
    private readonly AnswerJournal _journal = new(Ans.Clock());

    [Fact]
    public void CapIdTangDanVaTraCuuDuoc()
    {
        var first = _journal.Record(Ans.Grounded());
        var second = _journal.Record(Ans.Grounded("softmax là gì"));

        Assert.Equal("a1", first.Id);
        Assert.Equal("a2", second.Id);
        Assert.Same(second, _journal.Find("a2"));
        Assert.Null(_journal.Find("a99"));
    }

    [Fact]
    public void GiuNguyenKetLuanVaDoTuTinCuaPhase14()
    {
        var record = _journal.Record(Ans.Grounded(verdict: AnswerVerdict.Uncertain, confidence: 0.33));

        Assert.True(record.WentThroughVerification);
        Assert.Equal(AnswerVerdict.Uncertain, record.Verdict);
        Assert.Equal(0.33, record.Confidence);
        Assert.Equal(ConfidenceBand.Low, record.Band);
    }

    /// <summary>
    /// Lượt trả lời bằng công cụ KHÔNG có độ tự tin — và sổ phải để null chứ
    /// không điền 0, vì null nghĩa là "không có lời khai nào", còn 0 nghĩa là
    /// "agent nói thẳng nó không có căn cứ".
    /// </summary>
    [Fact]
    public void LuotTraLoiBangCongCuThiKhongCoDoTuTin()
    {
        var response = new AgentResponse("2 + 3 = 5", Intent.Calculate, 0.97, "calculator", []);
        var record = _journal.Record("tính 2 cộng 3", response);

        Assert.False(record.WentThroughVerification);
        Assert.Null(record.Verdict);
        Assert.Null(record.Confidence);
        Assert.Null(record.Band);
        Assert.Equal(Intent.Calculate, record.PredictedIntent);
        Assert.Equal(0.97, record.IntentConfidence);
    }

    [Fact]
    public void QuaDungLuongThiCauTraLoiCuNhatBiDayRa()
    {
        var journal = new AnswerJournal(Ans.Clock(), capacity: 2);

        journal.Record(Ans.Grounded("câu 1"));
        journal.Record(Ans.Grounded("câu 2"));
        journal.Record(Ans.Grounded("câu 3"));

        Assert.Equal(2, journal.Count);
        Assert.Equal(1, journal.EvictedCount);
        Assert.Null(journal.Find("a1"));
        Assert.NotNull(journal.Find("a3"));
    }

    [Fact]
    public void DungLuongPhaiDuong()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AnswerJournal(Ans.Clock(), capacity: 0));
    }
}

public class FeedbackStoreTests
{
    private readonly AnswerJournal _journal = new(Ans.Clock());
    private readonly FeedbackStore _store;

    public FeedbackStoreTests() => _store = new FeedbackStore(_journal, Ans.Clock());

    private AnswerRecord Record(AnswerVerdict verdict = AnswerVerdict.Answered, double confidence = 0.85) =>
        _journal.Record(Ans.Grounded(verdict: verdict, confidence: confidence));

    [Fact]
    public void NhanPhanHoiGanVaoCauTraLoiCoThat()
    {
        var record = Record();

        var result = _store.Submit(record.Id, FeedbackRating.Correct);

        Assert.True(result.Accepted);
        Assert.Equal(record.Id, result.Feedback!.AnswerId);
        Assert.Equal(1, _store.Count);
    }

    /// <summary>
    /// Phản hồi về một câu trả lời không tồn tại thì không truy được về căn cứ
    /// nào — nhận nó vào chỉ tạo ra số liệu không có gì đối chiếu.
    /// </summary>
    [Fact]
    public void TuChoiPhanHoiVeCauTraLoiKhongTonTai()
    {
        var result = _store.Submit("a999", FeedbackRating.Incorrect);

        Assert.False(result.Accepted);
        Assert.Contains("a999", result.Reason, StringComparison.Ordinal);
        Assert.Equal(0, _store.Count);
    }

    /// <summary>Hiệu chuẩn là tỉ lệ trên số CÂU TRẢ LỜI, không phải trên số lần bấm.</summary>
    [Fact]
    public void MotCauTraLoiChiDuocChamMotLan()
    {
        var record = Record();

        Assert.True(_store.Submit(record.Id, FeedbackRating.Correct).Accepted);

        var again = _store.Submit(record.Id, FeedbackRating.Incorrect);

        Assert.False(again.Accepted);
        Assert.Contains("đã được chấm", again.Reason, StringComparison.Ordinal);
        Assert.Equal(1, _store.Count);
    }

    [Fact]
    public void TuChoiYDinhKhongTonTai()
    {
        var record = Record();

        var result = _store.Submit(record.Id, FeedbackRating.Incorrect, expectedIntent: (Intent)99);

        Assert.False(result.Accepted);
        Assert.Contains("không nằm trong tập ý định", result.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void DemDuocPhanHoiTieuCucMaKhongSuaDuoc()
    {
        var a = Record();
        var b = Record();
        var c = Record();

        _store.Submit(a.Id, FeedbackRating.Incorrect);                                      // không sửa được
        _store.Submit(b.Id, FeedbackRating.Incorrect, expectedIntent: Intent.SearchInternet); // có nhãn
        _store.Submit(c.Id, FeedbackRating.Correct);                                        // tích cực

        Assert.Equal(1, _store.UnactionableCount);
        Assert.Equal(2, _store.Negative.Count);
    }

    [Fact]
    public void DemDungSaiTachTheoKetLuan()
    {
        var answered = Record(AnswerVerdict.Answered);
        var uncertain = Record(AnswerVerdict.Uncertain, 0.3);
        var insufficient = Record(AnswerVerdict.Insufficient, 0.0);

        _store.Submit(answered.Id, FeedbackRating.Correct);
        _store.Submit(uncertain.Id, FeedbackRating.Incorrect);
        _store.Submit(insufficient.Id, FeedbackRating.Unhelpful);

        var tallies = _store.TallyByVerdict();

        Assert.Equal(1, tallies.Single(t => t.Verdict == AnswerVerdict.Answered).Correct);
        Assert.Equal(1, tallies.Single(t => t.Verdict == AnswerVerdict.Uncertain).Wrong);
        Assert.Equal(0, tallies.Single(t => t.Verdict == AnswerVerdict.Insufficient).Correct);
        Assert.DoesNotContain(tallies, t => t.Verdict == AnswerVerdict.Conflicted);
    }

    /// <summary>
    /// Đây là số liệu duy nhất trong cả dự án mà hệ thống không tự đo được:
    /// độ tự tin agent tự nhận có khớp với thực tế hay không.
    /// </summary>
    [Fact]
    public void HieuChuanSoDoTuTinTuNhanVoiTiLeDungDoDuoc()
    {
        // Bốn câu trả lời dải High (>= 0.70), chỉ một câu đúng.
        for (int i = 0; i < 4; i++)
        {
            var record = Record(confidence: 0.80);
            _store.Submit(record.Id, i == 0 ? FeedbackRating.Correct : FeedbackRating.Incorrect);
        }

        var report = _store.Calibrate();
        var high = report.Buckets.Single(b => b.Band == ConfidenceBand.High);

        Assert.Equal(0.80, high.Claimed, 3);
        Assert.Equal(0.25, high.Observed, 3);
        Assert.True(high.IsOverconfident);
        Assert.Equal(3, report.OverconfidentErrors);
        Assert.Equal(0.25, report.OverallCorrectRate, 3);
    }

    [Fact]
    public void HieuChuanDemCaLanBoLo()
    {
        var missed = Record(AnswerVerdict.Insufficient, 0.0);
        _store.Submit(missed.Id, FeedbackRating.Unhelpful, correction: "câu trả lời đúng là X");

        Assert.Equal(1, _store.Calibrate().MissedAnswers);
    }

    /// <summary>
    /// Lượt trả lời bằng công cụ không tự nhận độ tự tin nào, nên không có gì
    /// để hiệu chuẩn. Đếm nó vào thì phải bịa ra một con số.
    /// </summary>
    [Fact]
    public void HieuChuanBoQuaLuotKhongQuaKiemChung()
    {
        var response = new AgentResponse("2 + 3 = 5", Intent.Calculate, 0.97, "calculator", []);
        var toolRecord = _journal.Record("tính 2 cộng 3", response);
        var groundedRecord = Record(confidence: 0.80);

        _store.Submit(toolRecord.Id, FeedbackRating.Correct);
        _store.Submit(groundedRecord.Id, FeedbackRating.Correct);

        var report = _store.Calibrate();

        Assert.Equal(1, report.TotalRated);
        Assert.Single(report.Buckets);
    }

    [Fact]
    public void KhongCoPhanHoiThiBaoCaoKhongBiaSo()
    {
        var report = _store.Calibrate();

        Assert.Equal(0, report.TotalRated);
        Assert.Empty(report.Buckets);
        Assert.Equal(0, report.OverallCorrectRate);
    }

    [Fact]
    public void TruyDuocTuPhanHoiVeCauTraLoi()
    {
        var record = Record();
        var feedback = _store.Submit(record.Id, FeedbackRating.Correct).Feedback!;

        Assert.Same(record, _store.AnswerFor(feedback));
    }
}

public class ErrorCaseCollectorTests
{
    private readonly AnswerJournal _journal = new(Ans.Clock());
    private readonly FeedbackStore _store;
    private readonly ErrorCaseCollector _collector = new(Ans.Clock());

    public ErrorCaseCollectorTests() => _store = new FeedbackStore(_journal, Ans.Clock());

    private (UserFeedback Feedback, AnswerRecord Record) Case(FeedbackRating rating,
        AnswerVerdict verdict = AnswerVerdict.Answered, double confidence = 0.85,
        Intent? predicted = null, Intent? expected = null, string? correction = null,
        double intentConfidence = 0.9)
    {
        var answer = Ans.Grounded(verdict: verdict, confidence: confidence, evidenceIds: ["k1", "w2"]);
        var record = predicted is null
            ? _journal.Record(answer)
            : _journal.Record(answer, predicted, intentConfidence);

        var feedback = _store.Submit(record.Id, rating, correction, expected).Feedback!;

        return (feedback, record);
    }

    [Fact]
    public void PhanHoiTichCucThiKhongCoCaLoiNao()
    {
        var (feedback, record) = Case(FeedbackRating.Correct);

        Assert.Null(_collector.Collect(feedback, record));
    }

    /// <summary>
    /// Ý định sai là lỗi THƯỢNG NGUỒN nên phải xét trước: hiểu sai câu hỏi thì
    /// mọi khâu sau đều chạy đúng trên một câu hỏi sai. Xét theo thứ tự khác
    /// thì ca này bị dán nhãn Overconfident và sẽ được đem đi sửa ngưỡng độ tự
    /// tin — trong khi ngưỡng không có lỗi gì.
    /// </summary>
    [Fact]
    public void YDinhSaiDuocXetTruocCaSaiMaNoiChac()
    {
        var (feedback, record) = Case(FeedbackRating.Incorrect, confidence: 0.95,
            predicted: Intent.RecallMemory, expected: Intent.SearchInternet);

        var errorCase = _collector.Collect(feedback, record)!;

        Assert.Equal(ErrorKind.WrongIntent, errorCase.Kind);
        Assert.True(errorCase.IsTrainable);
        Assert.Equal(Intent.SearchInternet, errorCase.ExpectedIntent);
        Assert.Equal(Intent.RecallMemory, errorCase.PredictedIntent);
    }

    /// <summary>Bộ phân loại càng tự tin về cái nhãn sai thì ca lỗi càng nặng.</summary>
    [Fact]
    public void YDinhSaiVoiDoTuTinCaoThiNangHon()
    {
        var confident = Case(FeedbackRating.Incorrect, predicted: Intent.RecallMemory,
            expected: Intent.SearchInternet, intentConfidence: 0.99);
        var unsure = Case(FeedbackRating.Incorrect, predicted: Intent.RecallMemory,
            expected: Intent.SearchInternet, intentConfidence: 0.51);

        double high = _collector.Collect(confident.Feedback, confident.Record)!.Severity;
        double low = _collector.Collect(unsure.Feedback, unsure.Record)!.Severity;

        Assert.True(high > low, $"{high} phải nặng hơn {low}");
    }

    [Fact]
    public void SaiVoiDoTuTinCaoLaLoiNoiQua()
    {
        var (feedback, record) = Case(FeedbackRating.Incorrect, confidence: 0.90);

        var errorCase = _collector.Collect(feedback, record)!;

        Assert.Equal(ErrorKind.Overconfident, errorCase.Kind);
        Assert.Equal(0.90, errorCase.Severity, 3);
        Assert.False(errorCase.IsTrainable);
    }

    [Fact]
    public void SaiVoiDoTuTinThapChiLaSaiNoiDung()
    {
        var (feedback, record) = Case(FeedbackRating.Incorrect, AnswerVerdict.Uncertain, confidence: 0.30);

        Assert.Equal(ErrorKind.WrongFact, _collector.Collect(feedback, record)!.Kind);
    }

    /// <summary>
    /// Lượt trả lời bằng công cụ không tự nhận độ tự tin nào, nên nó không thể
    /// "nói quá" — phải rơi xuống WrongFact, không phải Overconfident.
    /// </summary>
    [Fact]
    public void KhongTuNhanDoTuTinThiKhongTheLaNoiQua()
    {
        var response = new AgentResponse("2 + 3 = 6", Intent.Calculate, 0.99, "calculator", []);
        var record = _journal.Record("tính 2 cộng 3", response);
        var feedback = _store.Submit(record.Id, FeedbackRating.Incorrect).Feedback!;

        var errorCase = _collector.Collect(feedback, record)!;

        Assert.Equal(ErrorKind.WrongFact, errorCase.Kind);
        Assert.Null(errorCase.Confidence);
        Assert.Null(errorCase.Verdict);
    }

    [Fact]
    public void NoiKhongDuCanCuMaNguoiDungCoCauTraLoiLaCaBoLo()
    {
        var withFix = Case(FeedbackRating.Unhelpful, AnswerVerdict.Insufficient, 0.0,
            correction: "câu trả lời đúng là X");
        var withoutFix = Case(FeedbackRating.Unhelpful, AnswerVerdict.Insufficient, 0.0);

        var withCase = _collector.Collect(withFix.Feedback, withFix.Record)!;
        var withoutCase = _collector.Collect(withoutFix.Feedback, withoutFix.Record)!;

        Assert.Equal(ErrorKind.MissedAnswer, withCase.Kind);
        Assert.Equal(ErrorKind.MissedAnswer, withoutCase.Kind);
        Assert.True(withCase.Severity > withoutCase.Severity);
    }

    [Fact]
    public void DungNhungThieuLaLoaiRieng()
    {
        var incomplete = Case(FeedbackRating.Incomplete);
        var unhelpful = Case(FeedbackRating.Unhelpful);

        Assert.Equal(ErrorKind.Incomplete, _collector.Collect(incomplete.Feedback, incomplete.Record)!.Kind);
        Assert.Equal(ErrorKind.Unhelpful, _collector.Collect(unhelpful.Feedback, unhelpful.Record)!.Kind);
    }

    /// <summary>Ca lỗi phải dựng lại được đủ ngữ cảnh, không chỉ là một dòng than phiền.</summary>
    [Fact]
    public void CaLoiGiuLaiCanCuVaCacQuyetDinh()
    {
        var (feedback, record) = Case(FeedbackRating.Incorrect, confidence: 0.9);

        var errorCase = _collector.Collect(feedback, record)!;

        Assert.Equal(["k1", "w2"], errorCase.EvidenceIds);
        Assert.NotEmpty(errorCase.Decisions);
        Assert.Equal(record.Question, errorCase.Question);
        Assert.Equal(feedback.Id, errorCase.FeedbackId);
    }

    [Fact]
    public void KhongChoPhepGhepPhanHoiVoiCauTraLoiKhac()
    {
        var first = Case(FeedbackRating.Incorrect);
        var second = Case(FeedbackRating.Correct);

        Assert.Throws<ArgumentException>(() => _collector.Collect(first.Feedback, second.Record));
    }

    /// <summary>
    /// Ý định sai nhưng người dùng nói đúng nhãn agent đã đoán thì KHÔNG phải
    /// ca hiểu sai ý định — nó là lỗi ở khâu khác.
    /// </summary>
    [Fact]
    public void NhanTrungVoiDuDoanThiKhongPhaiLoiYDinh()
    {
        var (feedback, record) = Case(FeedbackRating.Incorrect, confidence: 0.9,
            predicted: Intent.SearchKnowledge, expected: Intent.SearchKnowledge);

        Assert.Equal(ErrorKind.Overconfident, _collector.Collect(feedback, record)!.Kind);
    }
}
