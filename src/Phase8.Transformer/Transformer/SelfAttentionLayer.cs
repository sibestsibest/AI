using Ai.Phase8.LinAlg;
using Ai.Phase8.Training;

namespace Ai.Phase8.Transformer;

/// <summary>
/// Self-attention có tham số học được VÀ có backward pass.
///
/// Phase 7 chỉ làm forward. Phase 8 phải đi ngược qua nó, và đây là phần
/// khó nhất của cả dự án — nên dẫn kỹ từng bước.
///
/// FORWARD:
///     Q = X·Wq , K = X·Wk , V = X·Wv
///     S = Q·Kᵀ / √dₖ          (che tam giác trên nếu causal)
///     A = softmax theo hàng của S
///     C = A·V
///     Y = C·Wo                 (chiếu ngược về số chiều của model)
///
/// BACKWARD, cho trước dY:
///     dWo = Cᵀ·dY              dC = dY·Woᵀ
///     dA  = dC·Vᵀ              dV = Aᵀ·dC
///
///     Qua softmax theo từng hàng i — Jacobian của softmax là
///         ∂Aᵢⱼ/∂Sᵢₖ = Aᵢⱼ(δⱼₖ − Aᵢₖ)
///     nhân ra và rút gọn được:
///         dSᵢⱼ = Aᵢⱼ · ( dAᵢⱼ − Σₖ Aᵢₖ·dAᵢₖ )
///     Số hạng tổng chính là "trung bình có trọng số" — nó bảo đảm tổng dS
///     theo hàng bằng 0, đúng như ràng buộc tổng A bằng 1.
///
///     dQ = (dS·K)/√dₖ          dK = (dSᵀ·Q)/√dₖ
///     dWq = Xᵀ·dQ , dWk = Xᵀ·dK , dWv = Xᵀ·dV
///     dX  = dQ·Wqᵀ + dK·Wkᵀ + dV·Wvᵀ      (X đi vào theo BA đường, cộng cả ba)
///
/// Dòng cuối là chỗ dễ sai nhất: X được dùng ba lần nên gradient phải cộng
/// dồn từ cả ba nhánh. Gradient check sẽ bắt ngay nếu thiếu một nhánh.
/// </summary>
public sealed class SelfAttentionLayer
{
    private readonly bool _causalMask;

    private Mat? _x;
    private Mat? _q;
    private Mat? _k;
    private Mat? _v;
    private Mat? _attention;      // A
    private Mat? _context;        // C = A·V

    public Param Wq { get; }

    public Param Wk { get; }

    public Param Wv { get; }

    public Param Wo { get; }

    public int ModelDimensions => Wq.Value.Rows;

    public int HeadDimensions => Wq.Value.Cols;

    /// <summary>Trọng số attention của lần forward gần nhất — để quan sát.</summary>
    public Mat? LastAttentionWeights => _attention;

    public SelfAttentionLayer(string name, int modelDimensions, int headDimensions, Random rng, bool causalMask)
    {
        ArgumentNullException.ThrowIfNull(rng);

        _causalMask = causalMask;
        Wq = Param.Random($"{name}.Wq", modelDimensions, headDimensions, rng);
        Wk = Param.Random($"{name}.Wk", modelDimensions, headDimensions, rng);
        Wv = Param.Random($"{name}.Wv", modelDimensions, headDimensions, rng);
        Wo = Param.Random($"{name}.Wo", headDimensions, modelDimensions, rng);
    }

    public Mat Forward(Mat x)
    {
        ArgumentNullException.ThrowIfNull(x);

        if (x.Cols != ModelDimensions)
        {
            throw new ArgumentException($"Attention cần {ModelDimensions} chiều nhưng nhận {x.Cols}.", nameof(x));
        }

        _x = x;
        _q = x.Multiply(Wq.Value);
        _k = x.Multiply(Wk.Value);
        _v = x.Multiply(Wv.Value);

        double scale = 1.0 / Math.Sqrt(HeadDimensions);
        var scores = _q.Multiply(_k.Transpose()).Scale(scale);

        if (_causalMask)
        {
            for (int i = 0; i < scores.Rows; i++)
            {
                for (int j = i + 1; j < scores.Cols; j++)
                {
                    scores[i, j] = double.NegativeInfinity;
                }
            }
        }

        _attention = new Mat(scores.Rows, scores.Cols);
        for (int i = 0; i < scores.Rows; i++)
        {
            _attention.SetRow(i, Softmax.Apply(scores.Row(i)));
        }

        _context = _attention.Multiply(_v);
        return _context.Multiply(Wo.Value);
    }

    public Mat Backward(Mat gradientWrtOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientWrtOutput);

        if (_x is null || _q is null || _k is null || _v is null || _attention is null || _context is null)
        {
            throw new InvalidOperationException("Phải gọi Forward trước Backward.");
        }

        // --- Qua phép chiếu đầu ra ---
        Wo.AccumulateGrad(_context.Transpose().Multiply(gradientWrtOutput));
        var dContext = gradientWrtOutput.Multiply(Wo.Value.Transpose());

        // --- Qua C = A·V ---
        var dAttention = dContext.Multiply(_v.Transpose());
        var dv = _attention.Transpose().Multiply(dContext);

        // --- Qua softmax theo từng hàng ---
        int n = _attention.Rows;
        var dScores = new Mat(n, n);

        for (int i = 0; i < n; i++)
        {
            double weightedSum = 0;
            for (int j = 0; j < n; j++)
            {
                weightedSum += _attention[i, j] * dAttention[i, j];
            }

            for (int j = 0; j < n; j++)
            {
                dScores[i, j] = _attention[i, j] * (dAttention[i, j] - weightedSum);
            }
        }

        // --- Qua S = Q·Kᵀ/√d ---
        double scale = 1.0 / Math.Sqrt(HeadDimensions);
        var dq = dScores.Multiply(_k).Scale(scale);
        var dk = dScores.Transpose().Multiply(_q).Scale(scale);

        // --- Qua ba phép chiếu Q, K, V ---
        Wq.AccumulateGrad(_x.Transpose().Multiply(dq));
        Wk.AccumulateGrad(_x.Transpose().Multiply(dk));
        Wv.AccumulateGrad(_x.Transpose().Multiply(dv));

        // X đi vào theo ba đường -> gradient là TỔNG của ba nhánh.
        return dq.Multiply(Wq.Value.Transpose())
            .Add(dk.Multiply(Wk.Value.Transpose()))
            .Add(dv.Multiply(Wv.Value.Transpose()));
    }

    public IEnumerable<Param> Parameters() => [Wq, Wk, Wv, Wo];
}
