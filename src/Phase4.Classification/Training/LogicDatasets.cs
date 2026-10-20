using Ai.Phase4.LinAlg;

namespace Ai.Phase4.Training;

/// <summary>
/// Các hàm logic nhị phân dùng làm bài toán phân loại.
///
/// AND và OR là "khả phân tuyến tính": vẽ 4 điểm lên mặt phẳng, có thể kẻ
/// MỘT đường thẳng tách lớp 0 khỏi lớp 1.
///
///   OR                      XOR
///   (0,1)● ─── ●(1,1)       (0,1)● ─── ○(1,1)
///     │    ╱    │             │         │
///     │  ╱      │             │         │      ○ = lớp 0   ● = lớp 1
///   (0,0)○ ─── ●(1,0)       (0,0)○ ─── ●(1,0)
///   kẻ được 1 đường         KHÔNG kẻ được đường nào
///
/// XOR thì không. Hai điểm lớp 1 nằm ở hai góc đối diện, hai điểm lớp 0 cũng
/// vậy — không đường thẳng nào tách được. Đây chính là bài toán đã khiến
/// nghiên cứu mạng neural đóng băng suốt thập niên 1970 (Minsky &amp; Papert,
/// 1969), cho đến khi người ta hiểu rằng layer ẩn giải được nó.
/// </summary>
public static class LogicDatasets
{
    public static TrainingData Xor() => Build([0, 1, 1, 0]);

    public static TrainingData And() => Build([0, 0, 0, 1]);

    public static TrainingData Or() => Build([0, 1, 1, 1]);

    /// <summary>Bốn tổ hợp đầu vào theo thứ tự (0,0), (0,1), (1,0), (1,1).</summary>
    public static readonly (double A, double B)[] Inputs =
    [
        (0, 0), (0, 1), (1, 0), (1, 1),
    ];

    private static TrainingData Build(double[] outputs)
    {
        var samples = new TrainingSample[4];
        for (int i = 0; i < 4; i++)
        {
            samples[i] = new TrainingSample(
                Vec.Of(Inputs[i].A, Inputs[i].B),
                Vec.Of(outputs[i]));
        }

        return new TrainingData(samples);
    }
}
