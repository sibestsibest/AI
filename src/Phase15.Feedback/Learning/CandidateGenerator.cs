using Ai.Phase15.Agent;
using Ai.Phase15.Feedback;
using Ai.Phase15.Grounding;
using Ai.Phase15.Memory;

namespace Ai.Phase15.Learning;

/// <summary>Một ứng viên được sinh ra hay không, và vì sao.</summary>
public sealed record CandidateResult(bool Accepted, string Reason, TrainingCandidate? Candidate = null);

/// <summary>
/// BỘ SINH ỨNG VIÊN — bốn cửa vào, và ba trong bốn cửa đó là cửa TỪ CHỐI.
///
///   <see cref="FromErrorCase"/>       ca lỗi WrongIntent + người dùng có nói nhãn  -> NHẬN
///   <see cref="FromCurated"/>         người viết hệ thống tự gán nhãn              -> NHẬN
///   <see cref="FromConfirmedAnswer"/> người dùng bấm "đúng" trên câu trả lời       -> TỪ CHỐI
///   <see cref="FromWebEvidence"/>     nội dung web                                 -> TỪ CHỐI
///
/// Hai cửa từ chối kia được viết ra THÀNH HÀM, chứ không phải "chỉ là không có
/// hàm nào làm việc đó". Đây là cùng một quyết định thiết kế như
/// <c>KnowledgeStore.TryPromote</c> của Phase 13, vì cùng một lý do: cái không
/// tồn tại thì không test được. Một hàm trả về lý do từ chối thì viết được
/// unit test khẳng định luật vẫn còn đó, và test sẽ đỏ ngay hôm có ai nới luật
/// ra — kể cả khi người nới luật thật lòng nghĩ mình đang cải tiến hệ thống.
/// </summary>
public sealed class CandidateGenerator
{
    private readonly IClock _clock;
    private int _nextId = 1;

    public CandidateGenerator(IClock? clock = null) => _clock = clock ?? new SystemClock();

    /// <summary>
    /// Đường CHÍNH: một ca lỗi "hiểu sai ý định" kèm nhãn do người dùng nói ra.
    ///
    /// Mọi loại lỗi khác đều bị từ chối kèm lý do chỉ rõ khâu cần sửa. Đây
    /// không phải sự khắt khe hình thức: bốn trong sáu loại lỗi của
    /// <see cref="ErrorKind"/> nằm ở khâu truy hồi, kho kiến thức hay công thức
    /// độ tự tin. Đổ chúng vào dữ liệu huấn luyện thì model học một mối liên
    /// hệ không có thật, còn nguyên nhân thật vẫn nguyên đó — và lần sau nó
    /// hỏng y như vậy, chỉ khác là bây giờ tập dữ liệu cũng bị nhiễu.
    /// </summary>
    public CandidateResult FromErrorCase(ErrorCase errorCase)
    {
        ArgumentNullException.ThrowIfNull(errorCase);

        if (errorCase.Kind != ErrorKind.WrongIntent)
        {
            return new CandidateResult(false,
                $"ca lỗi loại {errorCase.Kind} không sửa được bằng huấn luyện — {WhereToFix(errorCase.Kind)}");
        }

        if (errorCase.ExpectedIntent is null)
        {
            return new CandidateResult(false,
                "người dùng nói agent hiểu sai nhưng không nói lẽ ra phải hiểu thế nào — không có nhãn thì không có dữ liệu");
        }

        return new CandidateResult(true, "nhãn do người dùng nói ra, ca lỗi thuộc khâu phân loại ý định",
            new TrainingCandidate
            {
                Id = $"c{_nextId++}",
                Input = errorCase.Question,
                Label = errorCase.ExpectedIntent.Value,
                Provenance = CandidateProvenance.UserCorrection,
                FeedbackId = errorCase.FeedbackId,
                ErrorCaseId = errorCase.Id,
                CreatedAt = _clock.Now,
            });
    }

    /// <summary>Đường dành cho NGƯỜI: tự tay gán nhãn. Không đi qua phản hồi nào.</summary>
    public CandidateResult FromCurated(string input, Intent label)
    {
        ArgumentNullException.ThrowIfNull(input);

        return new CandidateResult(true, "người viết hệ thống tự gán nhãn",
            new TrainingCandidate
            {
                Id = $"c{_nextId++}",
                Input = input,
                Label = label,
                Provenance = CandidateProvenance.CuratedSeed,
                CreatedAt = _clock.Now,
            });
    }

    /// <summary>
    /// CỬA TỪ CHỐI 1 — câu trả lời được người dùng bấm "đúng".
    ///
    /// Đây là cửa dễ bị nới nhất trong cả hệ thống, vì nó trông vô hại đến mức
    /// khó tin: người dùng đã xác nhận là đúng rồi, dữ liệu miễn phí, và càng
    /// dùng càng nhiều dữ liệu. Chính vì thế nó được viết ra tường minh kèm lý
    /// do, chứ không để im lặng.
    ///
    /// Nhãn ở đây do MODEL tự đoán. Người dùng chấm câu trả lời — họ không đọc,
    /// không thấy và không xác nhận cái nhãn ý định bên trong. Nhận vào thì
    /// model được dạy bằng chính đầu ra của nó: mọi thiên lệch ban đầu được
    /// khẳng định lại mỗi vòng, và không có thông tin nào từ thế giới thật đi
    /// vào. Tệ hơn nữa là nó trông rất ổn trong khi đang xảy ra — accuracy trên
    /// tập tự gán nhãn chỉ có thể tăng.
    ///
    /// Muốn dùng được lượt "đúng" này thì phải hỏi người dùng đúng câu hỏi
    /// khác: "lẽ ra tôi phải hiểu câu này theo ý nào?". Lúc đó nhãn là của
    /// người, và nó vào bằng <see cref="FromErrorCase"/>.
    /// </summary>
    public CandidateResult FromConfirmedAnswer(AnswerRecord record, UserFeedback feedback)
    {
        ArgumentNullException.ThrowIfNull(record);
        ArgumentNullException.ThrowIfNull(feedback);

        return new CandidateResult(false,
            $"nhãn '{record.PredictedIntent}' do chính model dự đoán, không phải do người dùng gán — " +
            "người dùng chấm nội dung câu trả lời, không xác nhận nhãn ý định bên trong; " +
            "huấn luyện từ đầu ra của chính mình là vòng tự khuếch đại");
    }

    /// <summary>
    /// CỬA TỪ CHỐI 2 — nội dung Internet.
    ///
    /// Không có ngoại lệ, kể cả nguồn <c>Reliability = 1.0</c>, kể cả khi nó
    /// trùng khớp với kho kiến thức nội bộ. Nếu web mở được đường vào trọng số
    /// thì bất kỳ ai đăng được một trang web cũng dạy được model — và không
    /// cần chờ ai duyệt.
    /// </summary>
    public CandidateResult FromWebEvidence(Evidence evidence, Intent label)
    {
        ArgumentNullException.ThrowIfNull(evidence);

        return new CandidateResult(false,
            $"căn cứ kênh {evidence.Kind} có mức tin cậy {evidence.Trust} — " +
            "nội dung Internet không bao giờ được thành dữ liệu huấn luyện");
    }

    /// <summary>Gom nhiều ca lỗi một lượt, giữ lại cả lý do của những ca bị bỏ.</summary>
    public (IReadOnlyList<TrainingCandidate> Candidates, IReadOnlyList<string> Skipped) FromErrorCases(
        IEnumerable<ErrorCase> errorCases)
    {
        ArgumentNullException.ThrowIfNull(errorCases);

        var candidates = new List<TrainingCandidate>();
        var skipped = new List<string>();

        foreach (var errorCase in errorCases)
        {
            var result = FromErrorCase(errorCase);

            if (result is { Accepted: true, Candidate: not null })
            {
                candidates.Add(result.Candidate);
            }
            else
            {
                skipped.Add($"{errorCase.Id}: {result.Reason}");
            }
        }

        return (candidates, skipped);
    }

    private static string WhereToFix(ErrorKind kind) => kind switch
    {
        ErrorKind.Overconfident => "phải sửa công thức/ngưỡng độ tự tin",
        ErrorKind.WrongFact => "phải sửa kho kiến thức hoặc khâu chọn căn cứ",
        ErrorKind.MissedAnswer => "phải bổ sung kiến thức hoặc mở thêm kênh truy hồi",
        ErrorKind.Incomplete => "phải nới số căn cứ được dẫn trong câu trả lời",
        ErrorKind.Unhelpful => "phải sửa khâu diễn đạt câu trả lời",
        _ => "không thuộc khâu có trọng số",
    };
}
