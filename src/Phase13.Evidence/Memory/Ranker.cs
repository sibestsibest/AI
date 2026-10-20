namespace Ai.Phase13.Memory;

/// <summary>Một ký ức kèm điểm số và lý do được chấm điểm như vậy.</summary>
public sealed record ScoredMemory(
    MemoryFact Fact,
    double Score,
    double Relevance,
    double Recency,
    double Frequency)
{
    public string Explain() =>
        $"score {Score:F3} = relevance {Relevance:F2} · 0.55 + importance {Fact.Importance:F2} · 0.20 " +
        $"+ recency {Recency:F2} · 0.15 + frequency {Frequency:F2} · 0.10";
}

/// <summary>
/// Chấm điểm ký ức — trái tim của việc "nhớ có chọn lọc".
///
/// Bốn tín hiệu:
///
///   RELEVANCE  ký ức này có liên quan đến câu hỏi không?   (trọng số 0.55)
///   IMPORTANCE nó quan trọng đến đâu?                       (0.20)
///   RECENCY    lần cuối dùng đến cách đây bao lâu?          (0.15)
///   FREQUENCY  đã dùng bao nhiêu lần?                       (0.10)
///
/// RELEVANCE nặng nhất vì một ký ức không liên quan thì dù quan trọng đến
/// đâu cũng vô dụng cho câu hỏi đang xét. Thực tế, relevance = 0 thì điểm
/// bằng 0 luôn — không lôi ra thứ chẳng dính dáng gì.
///
/// RECENCY suy giảm theo hàm mũ với chu kỳ bán rã:
///
///     recency = 0.5 ^ (số giờ kể từ lần dùng cuối / halfLife)
///
/// Sau đúng một chu kỳ bán rã thì còn 0.5, hai chu kỳ còn 0.25... Dạng
/// đường cong này khớp khá tốt với đường quên của Ebbinghaus ở người.
///
/// FREQUENCY dùng log để một ký ức được truy cập 100 lần không nuốt chửng
/// mọi thứ khác: log(1+100)/log(1+100) chứ không phải 100 lần điểm.
/// </summary>
public sealed class Ranker(double recencyHalfLifeHours = 72)
{
    public const double RelevanceWeight = 0.55;
    public const double ImportanceWeight = 0.20;
    public const double RecencyWeight = 0.15;
    public const double FrequencyWeight = 0.10;

    public double RecencyHalfLifeHours { get; } = recencyHalfLifeHours > 0
        ? recencyHalfLifeHours
        : throw new ArgumentOutOfRangeException(nameof(recencyHalfLifeHours));

    public double Recency(MemoryFact fact, DateTimeOffset now) =>
        Math.Pow(0.5, Math.Max(0, fact.HoursSinceLastAccess(now)) / RecencyHalfLifeHours);

    public static double Frequency(MemoryFact fact) =>
        Math.Min(1.0, Math.Log(1 + fact.AccessCount) / Math.Log(11));   // 10 lần dùng -> 1.0

    /// <summary>Điểm khi CÓ câu hỏi — dùng cho việc lấy ký ức ra.</summary>
    public ScoredMemory Score(MemoryFact fact, IReadOnlySet<string> queryKeywords, DateTimeOffset now)
    {
        ArgumentNullException.ThrowIfNull(fact);

        double relevance = Keywords.Relevance(queryKeywords, fact.Keywords);
        double recency = Recency(fact, now);
        double frequency = Frequency(fact);

        // Không liên quan thì không lấy, bất kể các tín hiệu khác.
        double score = relevance <= 0
            ? 0
            : relevance * RelevanceWeight
              + fact.Importance * ImportanceWeight
              + recency * RecencyWeight
              + frequency * FrequencyWeight;

        return new ScoredMemory(fact, score, relevance, recency, frequency);
    }

    /// <summary>
    /// Điểm khi KHÔNG có câu hỏi — dùng để quyết định nên quên cái gì.
    /// Ở đây chỉ còn ba tín hiệu: quan trọng, gần đây, hay dùng.
    /// </summary>
    public double RetentionScore(MemoryFact fact, DateTimeOffset now)
    {
        ArgumentNullException.ThrowIfNull(fact);

        return fact.Importance * 0.5 + Recency(fact, now) * 0.3 + Frequency(fact) * 0.2;
    }
}
