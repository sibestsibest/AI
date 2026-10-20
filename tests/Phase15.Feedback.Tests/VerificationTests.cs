using Ai.Phase15.Agent;
using Ai.Phase15.Grounding;
using Ai.Phase15.Internet;
using Ai.Phase15.Memory;
using Ai.Phase15.Reasoning;
using Ai.Phase15.Verification;

namespace Ai.Phase15.Tests;

/// <summary>Dựng sẵn căn cứ đã chấm điểm — Phase 14 nhận evidence đã qua ranker.</summary>
internal static class Ev
{
    public static readonly DateTimeOffset Now = new(2026, 1, 1, 9, 0, 0, TimeSpan.Zero);

    public static Evidence Knowledge(string topic, string content,
        double reliability = 0.95, double relevance = 1.0) =>
        Evidence.FromKnowledge(new KnowledgeEntry
        {
            Id = $"k-{topic}",
            Topic = topic,
            Content = content,
            TopicKeywords = Keywords.Extract(topic),
            AllKeywords = Keywords.Extract($"{topic} {content}"),
            AddedAt = Now,
        }) with { Reliability = reliability, Relevance = relevance };

    public static Evidence Web(string content, double reliability = 0.6,
        double relevance = 1.0, string url = "https://blog.example.com/x") =>
        Evidence.FromWeb(
            UntrustedText.FromWeb(content),
            Source.FromInternet("Trang web", new Uri(url), Now, reliability))
        with { Reliability = reliability, Relevance = relevance };

    public static Evidence Memory(string content, double reliability = 0.8, double relevance = 1.0) =>
        Evidence.FromMemory(new MemoryFact
        {
            Id = "m1",
            Content = content,
            Keywords = Keywords.Extract(content),
            Importance = 0.7,
            CreatedAt = Now,
            LastAccessedAt = Now,
        }) with { Reliability = reliability, Relevance = relevance };

    public static Evidence Reasoning(string statement, string result) =>
        Evidence.FromReasoning(statement, result) with { Reliability = 0.9, Relevance = 1.0 };

    /// <summary>Một EvidenceSet dựng tay, để test khâu kiểm chứng tách khỏi khâu thu thập.</summary>
    public static EvidenceSet Set(params Evidence[] items) =>
        new() { Question = "câu hỏi thử", Items = items, CollectedAt = Now };
}

public class ClaimTests
{
    private static Claim Build(double support, params string[] evidenceIds) => new()
    {
        Id = "c1",
        Text = "một khẳng định nào đó",
        Kind = ClaimKind.Fact,
        EvidenceIds = evidenceIds,
        Support = support,
    };

    /// <summary>Khẳng định không dẫn được căn cứ nào là khẳng định agent tự nghĩ ra.</summary>
    [Fact]
    public void KhongDanCanCuThiKhongDuocCoiLaDuocDo()
    {
        Assert.False(Build(1.0).IsSupported);
    }

    [Fact]
    public void DuoiNguongDoThiKhongDuocCoiLaDuocDo()
    {
        Assert.False(Build(0.49, "e1").IsSupported);
        Assert.True(Build(0.5, "e1").IsSupported);
        Assert.True(Build(1.0, "e1").IsSupported);
    }

    [Fact]
    public void NhanSuThatHaySuyRaLaHaiKieuKhacNhau()
    {
        Assert.NotEqual(ClaimKind.Fact, ClaimKind.Inference);
    }
}

public class ContradictionDetectorTests
{
    private readonly ContradictionDetector _detector = new();

    [Fact]
    public void PhatHienPhuDinhLechNhau()
    {
        var contradiction = _detector.Compare(
            Ev.Knowledge("SSRF", "SSRF là lỗ hổng cho phép máy chủ gửi request tới địa chỉ nội bộ"),
            Ev.Web("SSRF không phải lỗ hổng cho phép máy chủ gửi request tới địa chỉ nội bộ"));

        Assert.NotNull(contradiction);
        Assert.Equal(ContradictionKind.Negation, contradiction.Kind);
        Assert.Contains("phủ định", contradiction.Explanation);
    }

    [Fact]
    public void PhatHienLechSo()
    {
        var contradiction = _detector.Compare(
            Ev.Knowledge("Timeout", "Giới hạn thời gian mặc định của bộ tải trang là 10 giây"),
            Ev.Web("Giới hạn thời gian mặc định của bộ tải trang là 30 giây"));

        Assert.NotNull(contradiction);
        Assert.Equal(ContradictionKind.NumericMismatch, contradiction.Kind);
        Assert.Contains("10", contradiction.Explanation);
        Assert.Contains("30", contradiction.Explanation);
    }

    [Fact]
    public void PhatHienCapTuDoiNghich()
    {
        var contradiction = _detector.Compare(
            Ev.Knowledge("Cache", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu"),
            Ev.Web("Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu"));

        Assert.NotNull(contradiction);
        Assert.Equal(ContradictionKind.Opposite, contradiction.Kind);
    }

    /// <summary>
    /// ĐÂY LÀ TEST QUAN TRỌNG NHẤT CỦA FILE.
    ///
    /// Hai câu chẳng liên quan gì đến nhau, một câu chứa "không". Nếu bỏ điều
    /// kiện cùng chủ đề thì chúng bị báo mâu thuẫn — báo động giả làm agent từ
    /// chối trả lời những câu nó trả lời được, và dạy người dùng bỏ qua cảnh báo.
    /// </summary>
    [Fact]
    public void KhacChuDeThiKHONGBaoMauThuan()
    {
        var contradiction = _detector.Compare(
            Ev.Knowledge("Softmax", "Softmax biến logits thành phân bố xác suất"),
            Ev.Knowledge("Overfitting", "Overfitting là khi model không học được quy luật"));

        Assert.Null(contradiction);
    }

    [Fact]
    public void CungChuDeVaCungY_ThiKhongBaoMauThuan()
    {
        var contradiction = _detector.Compare(
            Ev.Knowledge("SSRF", "SSRF là lỗ hổng khi máy chủ bị lừa gửi request nội bộ"),
            Ev.Web("SSRF là lỗ hổng khi máy chủ bị lừa gửi request nội bộ"));

        Assert.Null(contradiction);
    }

    /// <summary>Số lệch chút ít là làm tròn khác nhau, không phải mâu thuẫn.</summary>
    [Fact]
    public void LechSoTrongSaiSoChoPhepThiBoQua()
    {
        var contradiction = _detector.Compare(
            Ev.Knowledge("Tỉ lệ", "Tỉ lệ nén trung bình của thuật toán này đạt khoảng 100 phần trăm"),
            Ev.Web("Tỉ lệ nén trung bình của thuật toán này đạt khoảng 102 phần trăm"));

        Assert.Null(contradiction);   // lệch 2% < NumericTolerance 5%
    }

    /// <summary>
    /// Mức nghiêm trọng lấy theo bên YẾU hơn: nguồn xịn mâu thuẫn với rác thì
    /// gần như không phải tranh chấp.
    /// </summary>
    [Fact]
    public void MucNghiemTrongTheoBenYeuHon()
    {
        var strongVsWeak = _detector.Compare(
            Ev.Knowledge("Cache", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu"),
            Ev.Web("Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu", reliability: 0.1));

        var strongVsStrong = _detector.Compare(
            Ev.Knowledge("Cache", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu"),
            Ev.Knowledge("Cache2", "Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu"));

        Assert.NotNull(strongVsWeak);
        Assert.NotNull(strongVsStrong);
        Assert.True(strongVsStrong.Severity > strongVsWeak.Severity);
        Assert.True(strongVsStrong.BetweenVerifiedSources);
        Assert.False(strongVsWeak.BetweenVerifiedSources);
    }

    [Fact]
    public void QuetHetCacCapVaXepTheoMucNghiemTrong()
    {
        var found = _detector.Detect([
            Ev.Knowledge("Cache", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu"),
            Ev.Web("Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu"),
            Ev.Knowledge("Softmax", "Softmax biến logits thành phân bố xác suất"),
        ]);

        Assert.Single(found);
        Assert.Equal(ContradictionKind.Opposite, found[0].Kind);
    }

    [Fact]
    public void KhongCoGiThiTraVeRong()
    {
        Assert.Empty(_detector.Detect([]));
        Assert.Empty(_detector.Detect([Ev.Knowledge("X", "một nội dung đơn lẻ")]));
    }

    [Theory]
    [InlineData("giá 1200000 đồng", 1200000.0)]
    [InlineData("khoảng 12.5 giây", 12.5)]
    [InlineData("tổng 1,200,000 đồng", 1200000.0)]
    [InlineData("không có số nào")]
    public void RutDuocSoTrongVanBan(string text, params double[] expected)
    {
        Assert.Equal(expected, ContradictionDetector.ExtractNumbers(text));
    }
}

public class ConfidenceEstimatorTests
{
    private readonly ConfidenceEstimator _estimator = new();

    private static Claim Claim(ClaimKind kind = ClaimKind.Fact, double support = 1.0) => new()
    {
        Id = "c1",
        Text = "x",
        Kind = kind,
        EvidenceIds = ["e1"],
        Support = support,
    };

    [Fact]
    public void KhongCoCanCuThiDoTuTinBangKhong()
    {
        var assessment = _estimator.Estimate(Ev.Set(), [], []);

        Assert.Equal(0, assessment.Confidence);
        Assert.Contains("không có căn cứ", assessment.Explanation);
    }

    [Fact]
    public void NguonDaKiemChoDoTuTinCaoHonNguonWeb()
    {
        var verified = _estimator.Estimate(
            Ev.Set(Ev.Knowledge("SSRF", "SSRF là lỗ hổng")), [Claim()], []);

        var web = _estimator.Estimate(
            Ev.Set(Ev.Web("SSRF là lỗ hổng")), [Claim()], []);

        Assert.True(verified.Confidence > web.Confidence);
        Assert.Contains("không có nguồn đã kiểm", web.Explanation);
    }

    /// <summary>Phạt theo TỈ LỆ (nhân), nên mâu thuẫn làm giảm mạnh dù điểm gốc bao nhiêu.</summary>
    [Fact]
    public void MauThuanLamGiamDoTuTin()
    {
        var evidence = Ev.Set(
            Ev.Knowledge("Cache", "Bật cache làm tăng tốc độ phản hồi"),
            Ev.Knowledge("Cache2", "Bật cache làm giảm tốc độ phản hồi"));

        var clean = _estimator.Estimate(evidence, [Claim()], []);

        var contradictions = new ContradictionDetector().Detect(evidence.Items);
        var conflicted = _estimator.Estimate(evidence, [Claim()], contradictions);

        Assert.NotEmpty(contradictions);
        Assert.True(conflicted.Confidence < clean.Confidence);
        Assert.Contains("mâu thuẫn", conflicted.Explanation);
    }

    [Fact]
    public void NhieuKenhChoDoTuTinCaoHon()
    {
        var single = _estimator.Estimate(
            Ev.Set(Ev.Knowledge("SSRF", "SSRF là lỗ hổng")), [Claim()], []);

        var multi = _estimator.Estimate(
            Ev.Set(Ev.Knowledge("SSRF", "SSRF là lỗ hổng"), Ev.Memory("tôi nghiên cứu SSRF")),
            [Claim()], []);

        Assert.True(multi.Confidence > single.Confidence);
    }

    [Fact]
    public void ChuYeuLaSuyRaThiBiTru()
    {
        var evidence = Ev.Set(Ev.Knowledge("X", "nội dung"));

        var facts = _estimator.Estimate(evidence, [Claim(ClaimKind.Fact), Claim(ClaimKind.Fact)], []);
        var inferences = _estimator.Estimate(evidence,
            [Claim(ClaimKind.Inference), Claim(ClaimKind.Inference)], []);

        Assert.True(inferences.Confidence < facts.Confidence);
        Assert.Contains("suy ra", inferences.Explanation);
    }

    [Fact]
    public void DoTuTinLuonTrongKhoang0Den1()
    {
        var assessment = _estimator.Estimate(
            Ev.Set(Ev.Knowledge("X", "nội dung", reliability: 1.0)), [Claim()], []);

        Assert.InRange(assessment.Confidence, 0.0, 1.0);
    }

    // ------------------------------------------------------------------
    // Chốt kết luận
    // ------------------------------------------------------------------

    [Fact]
    public void KhongCoCanCuThiKetLuanLaInsufficient()
    {
        Assert.Equal(AnswerVerdict.Insufficient, _estimator.Decide(0.9, [], hasEvidence: false));
    }

    [Theory]
    [InlineData(0.05, AnswerVerdict.Insufficient)]
    [InlineData(0.30, AnswerVerdict.Uncertain)]
    [InlineData(0.80, AnswerVerdict.Answered)]
    public void KetLuanTheoDaiDoTuTin(double confidence, AnswerVerdict expected)
    {
        Assert.Equal(expected, _estimator.Decide(confidence, [], hasEvidence: true));
    }

    /// <summary>
    /// Mâu thuẫn xét TRƯỚC độ tự tin. Hai nguồn đã kiểm nói ngược nhau thì
    /// điểm vẫn cao — mà đó đúng là lúc không được chọn bừa một phía.
    /// </summary>
    [Fact]
    public void MauThuanThangDoTuTinCao()
    {
        var evidence = Ev.Set(
            Ev.Knowledge("Cache", "Bật cache làm tăng tốc độ phản hồi"),
            Ev.Knowledge("Cache2", "Bật cache làm giảm tốc độ phản hồi"));

        var contradictions = new ContradictionDetector().Detect(evidence.Items);

        Assert.Equal(AnswerVerdict.Conflicted, _estimator.Decide(0.95, contradictions, hasEvidence: true));
    }
}

public class AnswerValidatorTests
{
    private readonly AnswerValidator _validator = new();

    private static readonly Evidence Source =
        Ev.Knowledge("SSRF", "SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ");

    private static Claim GoodClaim() => new()
    {
        Id = "c1",
        Text = Source.Content,
        Kind = ClaimKind.Fact,
        EvidenceIds = [Source.Id],
        Support = 1.0,
    };

    [Fact]
    public void CauTraLoiDungTuCanCuThiHopLe()
    {
        var report = _validator.Validate(
            "SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ.",
            [GoodClaim()], [Source]);

        Assert.True(report.IsValid);
        Assert.Empty(report.UnsupportedClaims);
        Assert.Equal(1.0, report.GroundedRatio);
    }

    /// <summary>
    /// ĐÂY LÀ LÝ DO VALIDATOR TỒN TẠI: nó bắt được câu bịa, dù bộ sinh hôm nay
    /// không bịa. Nó không tin bộ sinh — kể cả bộ sinh viết sau này.
    /// </summary>
    [Fact]
    public void BatDuocCauBiaThemKhongCoTrongCanCu()
    {
        string answer = Source.Content +
            ". Ngoài ra, mọi tường lửa thương mại đều đã chặn hoàn toàn kiểu tấn công này từ năm 2019.";

        var report = _validator.Validate(answer, [GoodClaim()], [Source]);

        Assert.False(report.IsValid);
        Assert.Single(report.UnsupportedClaims);
        Assert.Contains("tường lửa", report.UnsupportedClaims[0].Text);
        Assert.Contains("không truy được về căn cứ", string.Join(";", report.Problems));
    }

    [Fact]
    public void BatDuocKhangDinhKhongDanCanCu()
    {
        var orphan = GoodClaim() with { EvidenceIds = [] };

        var report = _validator.Validate(Source.Content + ".", [orphan], [Source]);

        Assert.False(report.IsValid);
        Assert.Contains("không dẫn căn cứ nào", string.Join(";", report.Problems));
    }

    [Fact]
    public void BatDuocKhangDinhDoQuaYeu()
    {
        var weak = GoodClaim() with { Support = 0.2 };

        var report = _validator.Validate(Source.Content + ".", [weak], [Source]);

        Assert.False(report.IsValid);
        Assert.Contains("dưới ngưỡng", string.Join(";", report.Problems));
    }

    /// <summary>Dẫn một id căn cứ không tồn tại — dấu hiệu bộ sinh đang bịa cả trích dẫn.</summary>
    [Fact]
    public void BatDuocTrichDanCanCuKhongTonTai()
    {
        var fake = GoodClaim() with { EvidenceIds = ["e-khong-ton-tai"] };

        var report = _validator.Validate(Source.Content + ".", [fake], [Source]);

        Assert.False(report.IsValid);
        Assert.Contains("không có trong tập đã dẫn", string.Join(";", report.Problems));
    }

    [Fact]
    public void CauQuaNganThiBoQuaKhongKiem()
    {
        var report = _validator.Validate("Xong. OK.", [GoodClaim()], [Source]);

        Assert.Equal(0, report.SentencesChecked);
        Assert.True(report.IsValid);
    }

    [Fact]
    public void TachCauTheoCaDauChamVaDongMoi()
    {
        var sentences = AnswerValidator.SplitSentences("Câu một. Câu hai\n• Câu ba; Câu bốn");

        Assert.Equal(4, sentences.Count);
        Assert.Equal("Câu ba", sentences[2]);
    }

    /// <summary>
    /// URL bị BỎ trước khi tách câu.
    ///
    /// Đây là test hồi quy cho một lỗi thật: URL chứa dấu chấm nên bị băm thành
    /// những mảnh vô nghĩa ("com/@someone/bitcoin-price-today"), rồi mảnh đó bị
    /// kiểm như một câu khẳng định và tất nhiên không có căn cứ nào đỡ. Kết quả
    /// là agent im lặng về một thông tin nó thực sự có.
    /// </summary>
    [Fact]
    public void BoUrlTruocKhiTachCau()
    {
        var sentences = AnswerValidator.SplitSentences(
            "Nội dung thật. nguồn: https://medium.com/@someone/bitcoin-price-today  [tin cậy 0.28]");

        Assert.DoesNotContain(sentences, s => s.Contains("medium"));
        Assert.DoesNotContain(sentences, s => s.Contains("bitcoin-price-today"));
        Assert.Contains(sentences, s => s.Contains("Nội dung thật"));
    }

    /// <summary>Và cả ở mức toàn bộ báo cáo: URL trong trích dẫn không làm câu trả lời bị coi là bịa.</summary>
    [Fact]
    public void UrlTrongTrichDanKhongLamCauTraLoiBiCoiLaBia()
    {
        var web = Ev.Web("SSRF là lỗ hổng khi máy chủ bị lừa gửi request nội bộ",
            url: "https://medium.com/@someone/some-very-long-slug-here");

        var claim = new Claim
        {
            Id = "c1",
            Text = web.Content,
            Kind = ClaimKind.Fact,
            EvidenceIds = [web.Id],
            Support = 1.0,
        };

        string answer = $"{web.Content}.\n    ↳ {web.Source.Cite()}  [tin cậy 0.28]";

        var report = _validator.Validate(answer, [claim], [web]);

        Assert.True(report.IsValid, string.Join("; ", report.Problems));
    }
}

public class AnswerPipelineTests
{
    private static ManualClock Clock() => new(Ev.Now);

    private static AnswerPipeline Build(
        KnowledgeStore? knowledge = null,
        ISearchService? internet = null,
        ReasoningEngine? reasoning = null,
        MemoryStore? memory = null,
        IClock? clock = null,
        AnswerPipelineOptions? options = null)
    {
        clock ??= Clock();

        var collector = new EvidenceCollector(knowledge, memory, internet: internet, clock: clock);

        return new AnswerPipeline(collector, reasoning, clock: clock, options: options);
    }

    [Fact]
    public async Task CoCanCuDuThiTraLoi()
    {
        var clock = Clock();

        var answer = await Build(KnowledgeSeed.BuildStore(clock), clock: clock)
            .AnswerAsync("SSRF là gì", CancellationToken.None);

        Assert.Equal(AnswerVerdict.Answered, answer.Verdict);
        Assert.True(answer.IsAnswer);
        Assert.NotEmpty(answer.Claims);
        Assert.True(answer.AllClaimsSupported);
        Assert.Contains("SSRF", answer.Text);
    }

    /// <summary>
    /// NHÁNH QUAN TRỌNG NHẤT CỦA PHASE 14: không đủ căn cứ thì KHÔNG trả lời.
    /// </summary>
    [Fact]
    public async Task KhongCoCanCuThiNoiThangLaKhongBiet()
    {
        var clock = Clock();

        var answer = await Build(KnowledgeSeed.BuildStore(clock), clock: clock)
            .AnswerAsync("thủ đô của nước Wakanda là gì", CancellationToken.None);

        Assert.Equal(AnswerVerdict.Insufficient, answer.Verdict);
        Assert.False(answer.IsAnswer);
        Assert.Equal(0, answer.Confidence);
        Assert.Contains("KHÔNG ĐỦ CĂN CỨ", answer.Text);
    }

    [Fact]
    public async Task CanCuYeuThiTraLoiKemNoiRoLaChuaChac()
    {
        var clock = Clock();

        // Chỉ có một nguồn web độ tin cậy thấp -> điểm tụt xuống dải Uncertain.
        var weakWeb = new StaticSearchProvider("web")
            .Add("Giá bitcoin", "https://medium.com/@a/b",
                "Bài blog cá nhân bàn về giá bitcoin hiện tại, không dẫn nguồn dữ liệu nào");

        var answer = await Build(
                internet: new SearchService(weakWeb, clock: clock,
                    options: new SearchServiceOptions { FetchPages = false }),
                clock: clock)
            .AnswerAsync("giá bitcoin", CancellationToken.None);

        Assert.Equal(AnswerVerdict.Uncertain, answer.Verdict);
        Assert.InRange(answer.Confidence, 0.20, 0.45);
        Assert.Contains("KHÔNG CHẮC", answer.Text);
        Assert.True(answer.RequiresDisclosure);

        // Vẫn phải qua được kiểm tra cuối — "không chắc" khác "không có căn cứ".
        Assert.True(answer.Validation!.IsValid, string.Join("; ", answer.Validation.Problems));
    }

    /// <summary>
    /// HAI nguồn ĐÃ KIỂM nói ngược nhau -> tranh chấp thật -> trình bày CẢ HAI,
    /// không chọn bên.
    /// </summary>
    [Fact]
    public async Task HaiNguonDaKiemNoiNguocNhauThiKhongChonBen()
    {
        var clock = Clock();

        var knowledge = new KnowledgeStore(clock);
        knowledge.Add("Cache tăng tốc", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu");
        knowledge.Add("Cache giảm tốc", "Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu");

        var answer = await Build(knowledge, clock: clock)
            .AnswerAsync("bật cache ảnh hưởng tốc độ phản hồi thế nào", CancellationToken.None);

        Assert.Equal(AnswerVerdict.Conflicted, answer.Verdict);
        Assert.NotEmpty(answer.Contradictions);
        Assert.True(answer.Contradictions[0].BetweenVerifiedSources);
        Assert.Contains("NÓI NGƯỢC NHAU", answer.Text);
        Assert.Contains("Phía A", answer.Text);
        Assert.Contains("Phía B", answer.Text);
        Assert.Contains("cả hai đều là nguồn đã kiểm chứng", answer.Text);
    }

    /// <summary>
    /// Mâu thuẫn NHẸ (một bên kém tin cậy hơn hẳn) -> vẫn trả lời theo bên
    /// mạnh, NHƯNG phải nêu rõ là có nguồn nói ngược lại.
    ///
    /// Giấu phía kia đi chính là hành vi "chọn bên rồi im lặng" mà Phase 14
    /// phê — chỉ khác là lần này nó xảy ra âm thầm qua ngưỡng điểm.
    /// </summary>
    [Fact]
    public async Task MauThuanNhe_VanTraLoiNhungPhaiNeuNguonNoiNguoc()
    {
        var clock = Clock();

        var knowledge = new KnowledgeStore(clock);
        knowledge.Add("Cache", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu");

        var web = new StaticSearchProvider("web")
            .Add("Cache", "https://blog.example.com/cache",
                "Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu");

        var answer = await Build(knowledge,
                internet: new SearchService(web, clock: clock,
                    options: new SearchServiceOptions { FetchPages = false }),
                clock: clock)
            .AnswerAsync("bật cache ảnh hưởng tốc độ phản hồi thế nào", CancellationToken.None);

        // Kết luận vẫn là trả lời được — nguồn đã kiểm thắng blog.
        Assert.True(answer.IsAnswer, $"kết luận thực tế: {answer.Verdict}");
        Assert.NotEmpty(answer.Contradictions);

        // Nhưng phía nói ngược PHẢI xuất hiện trong câu trả lời.
        Assert.Contains("CÓ NGUỒN NÓI NGƯỢC LẠI", answer.Text);
        Assert.Contains("giảm", answer.Text);
        Assert.Contains("tăng", answer.Text);
    }

    /// <summary>
    /// Bộ bỏ trùng KHÔNG được gộp hai mẩu nói ngược chiều nhau.
    ///
    /// Hai câu chỉ khác nhau đúng một từ ("tăng"/"giảm") nên Jaccard của chúng
    /// rất cao — gộp lại là làm mâu thuẫn biến mất trước khi có ai kịp thấy,
    /// và còn đẩy Agreement lên 2 như thể hai kênh cùng xác nhận.
    /// </summary>
    [Fact]
    public void BoTrungKhongGopHaiMauNoiNguocChieu()
    {
        var report = new EvidenceDeduplicator().Deduplicate([
            Ev.Knowledge("Cache", "Bật cache làm tăng tốc độ phản hồi của dịch vụ tra cứu"),
            Ev.Web("Bật cache làm giảm tốc độ phản hồi của dịch vụ tra cứu"),
        ]);

        Assert.Equal(2, report.Unique.Count);
        Assert.Equal(0, report.DuplicatesRemoved);
    }

    /// <summary>Và cả với phủ định — "không" là stop word nên đây là trường hợp còn dễ lọt hơn.</summary>
    [Fact]
    public void BoTrungKhongGopHaiMauPhuDinhNhau()
    {
        var report = new EvidenceDeduplicator().Deduplicate([
            Ev.Knowledge("SSRF", "SSRF là lỗ hổng cho phép máy chủ gửi request tới địa chỉ nội bộ"),
            Ev.Web("SSRF không phải lỗ hổng cho phép máy chủ gửi request tới địa chỉ nội bộ"),
        ]);

        Assert.Equal(2, report.Unique.Count);
    }

    [Fact]
    public async Task DungSuyLuanKyHieuChoCauHoiSoHoc()
    {
        var clock = Clock();
        var reasoning = new ReasoningEngine();
        reasoning.LearnAll("A = 10", "B = 20", "C = A + B");

        var answer = await Build(KnowledgeSeed.BuildStore(clock), reasoning: reasoning, clock: clock)
            .AnswerAsync("2 + 3 * 4", CancellationToken.None);

        Assert.Contains("14", answer.Text);
        Assert.Contains(answer.Claims, c => c.Kind == ClaimKind.Inference);
        Assert.True(answer.Summary.InferenceCount >= 1);
    }

    /// <summary>Kết quả suy luận được gắn nhãn SUY RA, không phải SỰ THẬT.</summary>
    [Fact]
    public async Task KetQuaSuyLuanLaSuyRaChuKhongPhaiSuThat()
    {
        var clock = Clock();
        var reasoning = new ReasoningEngine();

        var answer = await Build(reasoning: reasoning, clock: clock)
            .AnswerAsync("100 / 8", CancellationToken.None);

        var derived = answer.Claims.FirstOrDefault(c => c.Text.Contains("12.5"));

        Assert.NotNull(derived);
        Assert.Equal(ClaimKind.Inference, derived.Kind);
    }

    [Fact]
    public async Task CauHoiKhaiNiemThiBuocSuyLuanBoQuaKhongLoi()
    {
        var clock = Clock();

        var answer = await Build(KnowledgeSeed.BuildStore(clock),
                reasoning: new ReasoningEngine(), clock: clock)
            .AnswerAsync("softmax là gì", CancellationToken.None);

        Assert.Equal(AnswerVerdict.Answered, answer.Verdict);
        Assert.Contains(answer.Claims, c => c.Kind == ClaimKind.Fact);
    }

    [Fact]
    public async Task MoiKhangDinhDuocDanCanCu()
    {
        var clock = Clock();

        var answer = await Build(KnowledgeSeed.BuildStore(clock), clock: clock)
            .AnswerAsync("softmax là gì", CancellationToken.None);

        Assert.All(answer.Claims, c =>
        {
            Assert.NotEmpty(c.EvidenceIds);
            Assert.True(c.IsSupported);
        });

        // Và id được dẫn phải thực sự nằm trong tập căn cứ đã dẫn.
        var ids = answer.CitedEvidence.Select(e => e.Id).ToHashSet(StringComparer.Ordinal);
        Assert.All(answer.Claims, c => Assert.All(c.EvidenceIds, id => Assert.Contains(id, ids)));
    }

    [Fact]
    public async Task QuaDuocKiemTraCuoi()
    {
        var clock = Clock();

        var answer = await Build(KnowledgeSeed.BuildStore(clock), clock: clock)
            .AnswerAsync("SSRF là gì", CancellationToken.None);

        Assert.NotNull(answer.Validation);
        Assert.True(answer.Validation.IsValid, string.Join("; ", answer.Validation.Problems));
        Assert.Empty(answer.Validation.UnsupportedClaims);
    }

    /// <summary>
    /// Bộ sinh bịa -> đường ống HẠ kết luận xuống Insufficient và viết lại,
    /// chứ không phát ra văn bản mà chính nó vừa kết luận là không có căn cứ.
    /// </summary>
    [Fact]
    public async Task BoSinhBia_ThiDuongOngHaKetLuanVaVietLai()
    {
        var clock = Clock();

        var collector = new EvidenceCollector(KnowledgeSeed.BuildStore(clock), clock: clock);

        var pipeline = new AnswerPipeline(collector,
            generator: new FabricatingGenerator(),
            clock: clock);

        var answer = await pipeline.AnswerAsync("SSRF là gì", CancellationToken.None);

        Assert.Equal(AnswerVerdict.Insufficient, answer.Verdict);
        Assert.False(answer.Validation!.IsValid);
        Assert.DoesNotContain("người ngoài hành tinh", answer.Text);
        Assert.Contains(answer.Summary.Decisions, d => d.Contains("hạ kết luận"));
    }

    [Fact]
    public async Task TatKiemTraThiQuanSatDuocCauTraLoiXau()
    {
        var clock = Clock();

        var collector = new EvidenceCollector(KnowledgeSeed.BuildStore(clock), clock: clock);

        var pipeline = new AnswerPipeline(collector,
            generator: new FabricatingGenerator(),
            clock: clock,
            options: new AnswerPipelineOptions { EnforceValidation = false });

        var answer = await pipeline.AnswerAsync("SSRF là gì", CancellationToken.None);

        Assert.False(answer.Validation!.IsValid);
        Assert.Contains("người ngoài hành tinh", answer.Text);   // lọt ra vì đã tắt chốt chặn
    }

    // ------------------------------------------------------------------
    // Tóm tắt suy luận
    // ------------------------------------------------------------------

    /// <summary>
    /// Tóm tắt phải là QUYẾT ĐỊNH + THAM CHIẾU, không phải dòng suy nghĩ.
    /// </summary>
    [Fact]
    public async Task TomTatSuyLuanLaQuyetDinhVaThamChieu()
    {
        var clock = Clock();

        var answer = await Build(KnowledgeSeed.BuildStore(clock), clock: clock)
            .AnswerAsync("SSRF là gì", CancellationToken.None);

        var summary = answer.Summary;

        Assert.NotEmpty(summary.Decisions);
        Assert.NotEmpty(summary.EvidenceIds);
        Assert.True(summary.ChannelCount >= 1);
        Assert.Equal(answer.Claims.Count(c => c.Kind == ClaimKind.Fact), summary.FactCount);

        // Mỗi quyết định là một câu ngắn, không phải một đoạn độc thoại.
        Assert.All(summary.Decisions, d => Assert.True(d.Length < 200, $"quá dài: {d}"));

        // Và mọi id được tham chiếu phải truy lại được.
        var ids = answer.CitedEvidence.Select(e => e.Id).ToHashSet(StringComparer.Ordinal);
        Assert.All(summary.EvidenceIds, id => Assert.Contains(id, ids));
    }

    [Fact]
    public async Task DaiDoTuTinKhopVoiConSo()
    {
        var clock = Clock();

        var answer = await Build(KnowledgeSeed.BuildStore(clock), clock: clock)
            .AnswerAsync("SSRF là gì", CancellationToken.None);

        Assert.Equal(answer.Confidence switch
        {
            >= 0.70 => ConfidenceBand.High,
            >= 0.45 => ConfidenceBand.Medium,
            >= 0.20 => ConfidenceBand.Low,
            _ => ConfidenceBand.VeryLow,
        }, answer.Band);
    }

    [Fact]
    public async Task TonTrongHuy()
    {
        var clock = Clock();

        var slow = new SearchService(new SlowProvider(TimeSpan.FromSeconds(30)), clock: clock,
            options: new SearchServiceOptions { FetchPages = false });

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => Build(internet: slow, clock: clock).AnswerAsync("SSRF là gì", cts.Token));
    }

    [Fact]
    public async Task KhongGhiGiVaoKnowledgeHayMemory()
    {
        var clock = Clock();
        var knowledge = KnowledgeSeed.BuildStore(clock);
        var memory = new MemoryStore(clock);

        int knowledgeBefore = knowledge.Count;

        var pipeline = Build(knowledge, memory: memory,
            internet: InternetStackFactory.BuildOffline(clock).Service, clock: clock);

        await pipeline.AnswerAsync("SSRF là gì", CancellationToken.None);
        await pipeline.AnswerAsync("prompt injection là gì", CancellationToken.None);

        Assert.Equal(knowledgeBefore, knowledge.Count);
        Assert.Equal(0, memory.LongTerm.Count);
    }

    /// <summary>Bộ sinh cố tình bịa — để test chốt chặn cuối có thật sự chặn.</summary>
    private sealed class FabricatingGenerator : IAnswerGenerator
    {
        private readonly AnswerGenerator _real = new();

        public IReadOnlyList<Claim> ExtractClaims(EvidenceSet evidence, IReadOnlyList<Evidence> cited) =>
            _real.ExtractClaims(evidence, cited);

        public string Compose(string question, AnswerVerdict verdict, double confidence,
            IReadOnlyList<Claim> claims, IReadOnlyList<Evidence> cited,
            IReadOnlyList<Contradiction> contradictions, bool requiresDisclosure)
        {
            // Kết luận Insufficient thì trả về bản thật, để test thấy được văn
            // bản đã được VIẾT LẠI sạch sau khi bị hạ cấp.
            if (verdict == AnswerVerdict.Insufficient)
            {
                return _real.Compose(question, verdict, confidence, claims, cited, contradictions, requiresDisclosure);
            }

            return "Lỗ hổng này do người ngoài hành tinh cài vào từ thế kỷ trước và không ai sửa được.";
        }
    }
}

[Collection("agent")]
public class VerificationAgentTests(TrainedAgent fixture)
{
    [Fact]
    public async Task NhanhInternetDiQuaKiemChung()
    {
        var response = await fixture.BuildWithInternet()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        Assert.Equal("verification", response.ToolUsed);
        Assert.NotNull(response.Grounded);
        Assert.Equal(AnswerVerdict.Answered, response.Grounded.Verdict);
    }

    [Fact]
    public async Task LuongQuyetDinhGhiLaiBuocKiemChung()
    {
        var response = await fixture.BuildWithInternet()
            .HandleAsync("tìm trên mạng thông tin về SSRF", CancellationToken.None);

        var questions = response.Decisions.Select(d => d.Question).ToArray();

        Assert.Contains("Kiểm chứng ra kết luận gì?", questions);
        Assert.Contains("Kiểm tra cuối có đạt?", questions);
    }

    [Fact]
    public async Task AnswerAsyncTraVeCauTraLoiCoCoSo()
    {
        var answer = await fixture.BuildWithInternet()
            .AnswerAsync("SSRF là gì", CancellationToken.None);

        Assert.Equal(AnswerVerdict.Answered, answer.Verdict);
        Assert.True(answer.Validation!.IsValid);
        Assert.NotEmpty(answer.Claims);
    }

    [Fact]
    public async Task AnswerAsyncGhiLaiLuotVaoNguCanh()
    {
        var agent = fixture.BuildWithInternet();

        await agent.AnswerAsync("SSRF là gì", CancellationToken.None);

        Assert.Equal(1, agent.Context!.Count);
        Assert.Equal("SSRF là gì", agent.Context.Last!.UserInput);
    }

    [Fact]
    public async Task ChuaCauHinhKiemChung_ThiBaoLoiRoRang()
    {
        var agent = fixture.BuildPhase13Style();

        Assert.False(agent.HasVerification);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => agent.AnswerAsync("SSRF là gì", CancellationToken.None));

        Assert.Contains("AnswerPipeline", ex.Message);
    }

    [Fact]
    public async Task KhongDuCanCu_ThiAgentNoiThang()
    {
        var response = await fixture.BuildWithInternet()
            .HandleAsync("tìm trên mạng zzzqqq khongtontai wxyz", CancellationToken.None);

        Assert.Equal(AnswerVerdict.Insufficient, response.Grounded!.Verdict);
        Assert.Contains("KHÔNG ĐỦ CĂN CỨ", response.Answer);
    }

    /// <summary>Mọi nhánh cũ vẫn chạy khi tầng kiểm chứng đã bật.</summary>
    [Fact]
    public async Task BatKiemChung_KhongLamHongCacNhanhCu()
    {
        var agent = fixture.BuildWithInternet();

        Assert.Contains("14", (await agent.HandleAsync("tính 2 + 3 * 4", CancellationToken.None)).Answer);
        Assert.Contains("chain rule", (await agent.HandleAsync("backpropagation là gì", CancellationToken.None)).Answer);
        Assert.Contains("4", (await agent.HandleAsync("có bao nhiêu đơn hàng", CancellationToken.None)).Answer);
        Assert.Equal("Tạm biệt.", (await agent.HandleAsync("tạm biệt", CancellationToken.None)).Answer);

        var calc = await agent.HandleAsync("tính 2 + 3 * 4", CancellationToken.None);
        Assert.Null(calc.Grounded);
    }

    [Fact]
    public async Task TraKiemChungKhongLamTangKhoKienThucHayKyUc()
    {
        var agent = fixture.BuildWithInternet();
        int knowledgeBefore = agent.Knowledge!.Count;

        await agent.AnswerAsync("SSRF là gì", CancellationToken.None);
        await agent.AnswerAsync("prompt injection là gì", CancellationToken.None);

        Assert.Equal(knowledgeBefore, agent.Knowledge.Count);
        Assert.Equal(0, agent.Memory.LongTerm.Count);
    }
}
