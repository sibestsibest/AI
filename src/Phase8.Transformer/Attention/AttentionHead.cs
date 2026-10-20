using Ai.Phase8.LinAlg;

namespace Ai.Phase8.Attention;

/// <summary>
/// Một "đầu" attention có tham số học được.
///
/// <see cref="ScaledDotProductAttention"/> không có tham số nào — nó chỉ là
/// công thức. Cái được HỌC nằm ở ba ma trận chiếu:
///
///     Q = X · Wq        "tôi đang tìm gì?"
///     K = X · Wk        "tôi cung cấp được gì?"
///     V = X · Wv        "nội dung tôi mang theo là gì?"
///
/// VÌ SAO PHẢI CÓ BA MA TRẬN RIÊNG, KHÔNG DÙNG THẲNG X?
/// Nếu Q = K = X thì điểm tương đồng luôn là x·x — mỗi token chỉ chú ý mạnh
/// nhất tới CHÍNH NÓ, và quan hệ giữa các token luôn đối xứng. Nhưng ngôn ngữ
/// không đối xứng: tính từ bổ nghĩa cho danh từ chứ không ngược lại. Ba phép
/// chiếu riêng cho phép "cái tôi tìm" khác "cái tôi có", nhờ vậy attention
/// biểu diễn được quan hệ có hướng.
///
/// Wq, Wk, Wv chính là tri thức của đầu attention, và được học bằng
/// backpropagation như mọi trọng số khác.
/// </summary>
public sealed class AttentionHead
{
    public Mat Wq { get; }

    public Mat Wk { get; }

    public Mat Wv { get; }

    public int ModelDimensions { get; }

    public int HeadDimensions { get; }

    public bool CausalMask { get; }

    public AttentionHead(int modelDimensions, int headDimensions, Random rng, bool causalMask = false)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (modelDimensions <= 0) throw new ArgumentOutOfRangeException(nameof(modelDimensions));
        if (headDimensions <= 0) throw new ArgumentOutOfRangeException(nameof(headDimensions));

        ModelDimensions = modelDimensions;
        HeadDimensions = headDimensions;
        CausalMask = causalMask;

        Wq = RandomMatrix(modelDimensions, headDimensions, rng);
        Wk = RandomMatrix(modelDimensions, headDimensions, rng);
        Wv = RandomMatrix(modelDimensions, headDimensions, rng);
    }

    /// <summary>Tạo đầu attention với trọng số xác định — cho test và demo tính tay.</summary>
    public AttentionHead(Mat wq, Mat wk, Mat wv, bool causalMask = false)
    {
        ArgumentNullException.ThrowIfNull(wq);
        ArgumentNullException.ThrowIfNull(wk);
        ArgumentNullException.ThrowIfNull(wv);

        if (wq.Rows != wk.Rows || wk.Rows != wv.Rows)
        {
            throw new ArgumentException("Wq, Wk, Wv phải có cùng số hàng (= số chiều model).");
        }

        if (wq.Cols != wk.Cols)
        {
            throw new ArgumentException("Wq và Wk phải cùng số cột để tính được Q·Kᵀ.");
        }

        Wq = wq;
        Wk = wk;
        Wv = wv;
        ModelDimensions = wq.Rows;
        HeadDimensions = wq.Cols;
        CausalMask = causalMask;
    }

    /// <param name="x">Chuỗi đầu vào: n hàng (n token) × ModelDimensions cột.</param>
    public AttentionResult Forward(Mat x)
    {
        ArgumentNullException.ThrowIfNull(x);

        if (x.Cols != ModelDimensions)
        {
            throw new ArgumentException(
                $"Đầu vào có {x.Cols} chiều nhưng head cần {ModelDimensions}.", nameof(x));
        }

        return ScaledDotProductAttention.Compute(
            x.Multiply(Wq),
            x.Multiply(Wk),
            x.Multiply(Wv),
            CausalMask);
    }

    public int ParameterCount => 3 * ModelDimensions * HeadDimensions;

    /// <summary>Ma trận đơn vị — chiếu "không làm gì", tiện cho demo.</summary>
    public static Mat Identity(int size)
    {
        var m = new Mat(size, size);
        for (int i = 0; i < size; i++) m[i, i] = 1;
        return m;
    }

    private static Mat RandomMatrix(int rows, int cols, Random rng)
    {
        double limit = Math.Sqrt(6.0 / (rows + cols));
        var m = new Mat(rows, cols);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                m[r, c] = (rng.NextDouble() * 2 - 1) * limit;
            }
        }

        return m;
    }
}
