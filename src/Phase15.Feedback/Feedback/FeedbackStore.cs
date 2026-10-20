using System.Globalization;
using Ai.Phase15.Agent;
using Ai.Phase15.Memory;
using Ai.Phase15.Verification;

namespace Ai.Phase15.Feedback;

/// <summary>Phản hồi được nhận hay bị từ chối, và vì sao.</summary>
public sealed record FeedbackResult(bool Accepted, string Reason, UserFeedback? Feedback = null);

/// <summary>Đếm đúng/sai theo từng kết luận của Phase 14.</summary>
public sealed record VerdictTally(AnswerVerdict Verdict, int Correct, int Wrong)
{
    public int Total => Correct + Wrong;

    public double CorrectRate => Total == 0 ? 0 : (double)Correct / Total;

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture, "{0}: {1}/{2} đúng ({3:P0})",
            Verdict, Correct, Total, CorrectRate);
}

/// <summary>
/// Một dải độ tự tin, kèm con số người dùng thực sự xác nhận.
///
/// <see cref="Claimed"/> là điều agent TỰ NÓI về mình, <see cref="Observed"/>
/// là điều đo được từ phản hồi. Khoảng cách giữa hai con số đó là thứ đáng
/// quan tâm nhất trong cả Phase 15.
/// </summary>
public sealed record CalibrationBucket(ConfidenceBand Band, double Claimed, double Observed, int Count)
{
    /// <summary>Dương = agent nói chắc hơn thực tế. Âm = nó tự hạ mình quá mức.</summary>
    public double Gap => Claimed - Observed;

    public bool IsOverconfident => Gap > 0.15;

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "{0,-6} tự nhận {1:P0} — thực tế {2:P0} ({3} mẫu), lệch {4:+0.##;-0.##;0}",
            Band, Claimed, Observed, Count, Gap);
}

/// <summary>
/// BÁO CÁO HIỆU CHUẨN — câu hỏi "độ tự tin của agent có đáng tin không".
///
/// Phase 14 in ra một con số tự tin và nói thẳng rằng nó KHÔNG phải xác suất
/// đúng — vì không có cách nào tính được xác suất đó mà không biết sự thật.
/// Phản hồi người dùng chính là mảnh sự thật còn thiếu. Có nó rồi thì con số
/// kia kiểm chứng được:
///
///     agent nói 80% -> người dùng xác nhận đúng 8/10 lần?  hiệu chuẩn tốt
///     agent nói 80% -> người dùng xác nhận đúng 4/10 lần?  nó đang nói quá
///
/// Đây là số liệu duy nhất trong dự án này mà HỆ THỐNG KHÔNG TỰ ĐO ĐƯỢC. Mọi
/// con số khác (loss, accuracy trên tập nhãn, gradient check) đều tính từ dữ
/// liệu có sẵn. Còn "câu trả lời này có đúng không" thì phải có người ngoài
/// nói — và đó là lý do vòng phản hồi tồn tại.
/// </summary>
public sealed record CalibrationReport
{
    public IReadOnlyList<CalibrationBucket> Buckets { get; init; } = [];

    public int TotalRated { get; init; }

    /// <summary>Trả lời với độ tự tin CAO nhưng người dùng nói sai. Loại lỗi tệ nhất.</summary>
    public int OverconfidentErrors { get; init; }

    /// <summary>Nói "không đủ căn cứ" trong khi người dùng có câu trả lời. Lỗi ngược lại.</summary>
    public int MissedAnswers { get; init; }

    public double OverallCorrectRate { get; init; }

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "{0} lượt chấm, đúng {1:P0}, {2} lần tự tin quá, {3} lần bỏ lỡ",
            TotalRated, OverallCorrectRate, OverconfidentErrors, MissedAnswers);
}

/// <summary>
/// KHO PHẢN HỒI — nhận, kiểm và đếm phản hồi của người dùng.
///
/// HAI LUẬT NHẬN VÀO, và cả hai đều tồn tại để những con số sau này không bịa:
///
///   1. Phản hồi phải gắn vào một câu trả lời CÓ TRONG SỔ. Id lạ thì từ chối
///      kèm lý do. Nhận bừa thì thống kê có mẫu mà không có gì đối chiếu.
///
///   2. MỘT câu trả lời được chấm NHIỀU NHẤT MỘT LẦN. Hiệu chuẩn là tỉ lệ
///      trên số CÂU TRẢ LỜI, không phải trên số lần bấm. Cho bấm nhiều lần
///      thì một người dùng nhiệt tình đủ sức tự tay đổi bức tranh hiệu chuẩn
///      của cả hệ thống — mà không hề nói dối câu nào.
///
/// Kho này CHỈ ĐẾM. Nó không sinh dữ liệu huấn luyện, không sửa kiến thức,
/// không chạm tới trọng số. Những việc đó là của <see cref="LearningPipeline"/>,
/// và tách ra để chỗ nhận dữ liệu bên ngoài không đồng thời là chỗ có quyền
/// thay đổi model.
/// </summary>
public sealed class FeedbackStore
{
    private readonly List<UserFeedback> _feedback = [];
    private readonly AnswerJournal _journal;
    private readonly IClock _clock;
    private int _nextId = 1;

    public FeedbackStore(AnswerJournal journal, IClock? clock = null)
    {
        ArgumentNullException.ThrowIfNull(journal);

        _journal = journal;
        _clock = clock ?? new SystemClock();
    }

    public int Count => _feedback.Count;

    public IReadOnlyList<UserFeedback> All => _feedback;

    public IReadOnlyList<UserFeedback> Negative => [.. _feedback.Where(f => f.IsNegative)];

    /// <summary>Phản hồi tiêu cực nhưng người dùng không nói đúng là gì — không sửa được.</summary>
    public int UnactionableCount =>
        _feedback.Count(f => f.IsNegative && f.ExpectedIntent is null && string.IsNullOrWhiteSpace(f.Correction));

    public FeedbackResult Submit(string answerId, FeedbackRating rating,
        string? correction = null, Intent? expectedIntent = null, string? comment = null)
    {
        ArgumentNullException.ThrowIfNull(answerId);

        // --- Luật 1: phải gắn được vào một câu trả lời có thật ---
        if (!_journal.Contains(answerId))
        {
            return new FeedbackResult(false,
                $"không có câu trả lời nào mang id '{answerId}' trong sổ — phản hồi không truy được về căn cứ nào");
        }

        // --- Luật 2: một câu trả lời chỉ được chấm một lần ---
        if (_feedback.Any(f => f.AnswerId == answerId))
        {
            return new FeedbackResult(false,
                $"câu trả lời '{answerId}' đã được chấm rồi — đếm hai lần sẽ làm lệch hiệu chuẩn");
        }

        if (expectedIntent is not null && !Enum.IsDefined(expectedIntent.Value))
        {
            return new FeedbackResult(false, $"ý định '{expectedIntent}' không nằm trong tập ý định đã biết");
        }

        var feedback = new UserFeedback
        {
            Id = $"f{_nextId++}",
            AnswerId = answerId,
            Rating = rating,
            Correction = string.IsNullOrWhiteSpace(correction) ? null : correction.Trim(),
            ExpectedIntent = expectedIntent,
            Comment = string.IsNullOrWhiteSpace(comment) ? null : comment.Trim(),
            SubmittedAt = _clock.Now,
        };

        _feedback.Add(feedback);

        return new FeedbackResult(true, "đã nhận", feedback);
    }

    public UserFeedback? Find(string feedbackId) => _feedback.FirstOrDefault(f => f.Id == feedbackId);

    /// <summary>Phản hồi về câu trả lời nào — dùng để dựng lại đủ ngữ cảnh của một lỗi.</summary>
    public AnswerRecord? AnswerFor(UserFeedback feedback)
    {
        ArgumentNullException.ThrowIfNull(feedback);

        return _journal.Find(feedback.AnswerId);
    }

    /// <summary>Đếm đúng/sai tách theo kết luận của Phase 14.</summary>
    public IReadOnlyList<VerdictTally> TallyByVerdict()
    {
        var tallies = new List<VerdictTally>();

        foreach (var verdict in Enum.GetValues<AnswerVerdict>())
        {
            var rated = _feedback
                .Select(f => (Feedback: f, Record: _journal.Find(f.AnswerId)))
                .Where(x => x.Record?.Verdict == verdict)
                .ToList();

            if (rated.Count == 0) continue;

            int correct = rated.Count(x => x.Feedback.Rating == FeedbackRating.Correct);
            tallies.Add(new VerdictTally(verdict, correct, rated.Count - correct));
        }

        return tallies;
    }

    /// <summary>
    /// Dựng báo cáo hiệu chuẩn.
    ///
    /// Gộp theo <see cref="ConfidenceBand"/> chứ không theo từng giá trị thập
    /// phân, vì mỗi giá trị riêng lẻ hầu như chỉ có một mẫu — và một mẫu thì
    /// tỉ lệ đúng chỉ có thể là 0% hoặc 100%, tức là vô nghĩa. Dải thì gộp
    /// được đủ mẫu để con số nói lên điều gì.
    ///
    /// CHỈ TÍNH những lượt ĐÃ ĐI QUA ĐƯỜNG ỐNG KIỂM CHỨNG. Lượt trả lời bằng
    /// công cụ không tự nhận một độ tự tin nào, nên không có gì để đối chiếu.
    /// Đếm chúng vào đây thì hoặc phải bịa ra một con số, hoặc phải coi chúng
    /// là 0% — cả hai đều làm báo cáo nói sai về chính nó.
    /// </summary>
    public CalibrationReport Calibrate()
    {
        var rated = _feedback
            .Select(f => (Feedback: f, Record: _journal.Find(f.AnswerId)))
            .Where(x => x.Record?.Grounded is not null)
            .Select(x => (x.Feedback, Record: x.Record!, Grounded: x.Record!.Grounded!))
            .ToList();

        var buckets = new List<CalibrationBucket>();

        foreach (var band in Enum.GetValues<ConfidenceBand>())
        {
            var inBand = rated.Where(x => x.Grounded.Band == band).ToList();
            if (inBand.Count == 0) continue;

            double claimed = inBand.Average(x => x.Grounded.Confidence);
            double observed = (double)inBand.Count(x => x.Feedback.Rating == FeedbackRating.Correct) / inBand.Count;

            buckets.Add(new CalibrationBucket(band, claimed, observed, inBand.Count));
        }

        return new CalibrationReport
        {
            Buckets = buckets,
            TotalRated = rated.Count,
            OverallCorrectRate = rated.Count == 0
                ? 0
                : (double)rated.Count(x => x.Feedback.Rating == FeedbackRating.Correct) / rated.Count,
            OverconfidentErrors = rated.Count(x =>
                x.Feedback.Rating == FeedbackRating.Incorrect
                && x.Grounded.Band is ConfidenceBand.High),
            MissedAnswers = rated.Count(x =>
                x.Feedback.IsNegative
                && x.Grounded.Verdict == AnswerVerdict.Insufficient),
        };
    }

    public void Clear()
    {
        _feedback.Clear();
        _nextId = 1;
    }
}
