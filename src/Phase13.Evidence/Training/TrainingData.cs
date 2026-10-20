using System.Collections;
using Ai.Phase13.LinAlg;

namespace Ai.Phase13.Training;

/// <summary>Một mẫu huấn luyện: vector đầu vào và vector đáp án đúng.</summary>
public readonly record struct TrainingSample(Vec Input, Vec Target);

/// <summary>
/// Tập dữ liệu huấn luyện. Ở Phase 1 dữ liệu là cặp hai số; giờ mỗi mẫu là
/// cặp hai VECTOR, vì mạng có thể có nhiều đầu vào và nhiều đầu ra.
/// </summary>
public sealed class TrainingData : IReadOnlyList<TrainingSample>
{
    private readonly TrainingSample[] _samples;

    public TrainingData(params TrainingSample[] samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Length == 0) throw new ArgumentException("Cần ít nhất 1 mẫu.", nameof(samples));

        _samples = samples;
    }

    public int Count => _samples.Length;

    public TrainingSample this[int index] => _samples[index];

    public IEnumerator<TrainingSample> GetEnumerator() => ((IEnumerable<TrainingSample>)_samples).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Tạo dữ liệu từ các cặp số vô hướng (1 đầu vào, 1 đầu ra).</summary>
    public static TrainingData FromScalars(params (double X, double Y)[] pairs)
    {
        ArgumentNullException.ThrowIfNull(pairs);

        var samples = pairs
            .Select(p => new TrainingSample(Vec.Of(p.X), Vec.Of(p.Y)))
            .ToArray();

        return new TrainingData(samples);
    }

    /// <summary>
    /// Đúng dataset của Phase 1 (y = 2x + 1). Dùng để chứng minh mạng neural
    /// tổng quát hoá được Linear Regression, không mâu thuẫn với nó.
    /// </summary>
    public static TrainingData Phase1Linear() =>
        FromScalars((1, 3), (2, 5), (3, 7), (4, 9), (5, 11));

    /// <summary>
    /// y = x² trên [−1, 1]. Đây là quan hệ PHI TUYẾN: một model kiểu Phase 1
    /// (đường thẳng) không thể khớp được, nên nó chứng minh layer ẩn +
    /// activation thực sự đem lại năng lực mới.
    /// </summary>
    public static TrainingData Square(int pointCount = 21)
    {
        if (pointCount < 2) throw new ArgumentOutOfRangeException(nameof(pointCount));

        var pairs = new (double, double)[pointCount];
        for (int i = 0; i < pointCount; i++)
        {
            double x = -1.0 + 2.0 * i / (pointCount - 1);
            pairs[i] = (x, x * x);
        }

        return FromScalars(pairs);
    }
}
