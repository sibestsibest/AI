using Ai.Phase8.Activations;
using Ai.Phase8.LinAlg;

namespace Ai.Phase8.Transformer;

/// <summary>
/// Một khối transformer, kiểu pre-norm (chuẩn hoá TRƯỚC mỗi khối con):
///
///     h = x + Attention( LayerNorm₁(x) )
///     y = h + FeedForward( LayerNorm₂(h) )
///
/// HAI Ý TƯỞNG GHÉP LẠI:
///
/// 1. ATTENTION = "trao đổi thông tin giữa các token".
///    Mỗi token nhìn sang các token khác và hút về thứ nó cần.
///
/// 2. FEED-FORWARD = "xử lý riêng trong từng token".
///    Cùng một mạng 2 layer áp lên TỪNG token độc lập, không token nào nhìn
///    token nào. Đây là nơi mô hình "suy nghĩ" về thông tin vừa thu thập.
///    Layer ẩn thường rộng gấp 4 lần (d -> 4d -> d).
///
/// Xen kẽ hai bước này nhiều lần chính là toàn bộ kiến trúc GPT.
///
/// RESIDUAL (dấu cộng "x +"):
///    Đường tắt cho gradient. Đạo hàm của x + f(x) theo x là 1 + f'(x) —
///    số 1 bảo đảm gradient luôn có một lối đi thẳng về layer trước, dù f'
///    có teo đến đâu. Không có residual thì mạng sâu không train nổi.
///    Backward của residual đơn giản đến bất ngờ: CỘNG gradient của cả hai nhánh.
///
/// PRE-NORM hay POST-NORM? Bài báo gốc dùng post-norm (chuẩn hoá SAU khi
/// cộng residual). Thực tế pre-norm dễ train hơn hẳn vì đường residual sạch
/// hoàn toàn, không bị LayerNorm chen vào giữa. Mọi mô hình hiện đại đều
/// dùng pre-norm.
/// </summary>
public sealed class TransformerBlock
{
    public LayerNorm Norm1 { get; }

    public SelfAttentionLayer Attention { get; }

    public LayerNorm Norm2 { get; }

    public Dense FeedForwardIn { get; }

    public Dense FeedForwardOut { get; }

    public int ModelDimensions { get; }

    public TransformerBlock(
        string name,
        int modelDimensions,
        int headDimensions,
        int feedForwardSize,
        Random rng,
        bool causalMask = true)
    {
        ArgumentNullException.ThrowIfNull(rng);

        ModelDimensions = modelDimensions;

        Norm1 = new LayerNorm($"{name}.ln1", modelDimensions);
        Attention = new SelfAttentionLayer($"{name}.attn", modelDimensions, headDimensions, rng, causalMask);
        Norm2 = new LayerNorm($"{name}.ln2", modelDimensions);
        FeedForwardIn = new Dense($"{name}.ff1", modelDimensions, feedForwardSize, new ReLU(), rng);
        FeedForwardOut = new Dense($"{name}.ff2", feedForwardSize, modelDimensions, new Identity(), rng);
    }

    public Mat Forward(Mat x)
    {
        ArgumentNullException.ThrowIfNull(x);

        // Khối con 1: giao tiếp giữa các token
        var h = x.Add(Attention.Forward(Norm1.Forward(x)));

        // Khối con 2: xử lý trong từng token
        return h.Add(FeedForwardOut.Forward(FeedForwardIn.Forward(Norm2.Forward(h))));
    }

    public Mat Backward(Mat gradientWrtOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientWrtOutput);

        // Ngược khối con 2. Gradient đi theo HAI đường: qua FFN và qua residual.
        var throughFeedForward = Norm2.Backward(
            FeedForwardIn.Backward(FeedForwardOut.Backward(gradientWrtOutput)));
        var dh = gradientWrtOutput.Add(throughFeedForward);

        // Ngược khối con 1, cũng hai đường.
        var throughAttention = Norm1.Backward(Attention.Backward(dh));
        return dh.Add(throughAttention);
    }

    public IEnumerable<Param> Parameters() =>
    [
        .. Norm1.Parameters(),
        .. Attention.Parameters(),
        .. Norm2.Parameters(),
        .. FeedForwardIn.Parameters(),
        .. FeedForwardOut.Parameters(),
    ];
}
