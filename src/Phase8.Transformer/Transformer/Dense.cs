using Ai.Phase8.Activations;
using Ai.Phase8.LinAlg;

namespace Ai.Phase8.Transformer;

/// <summary>
/// Layer tuyến tính làm việc trên cả chuỗi token cùng lúc:
///
///     Y = f( X · W + b )
///
/// X là n × in (n token), W là in × out, b broadcast lên mọi hàng.
///
/// Khác gì <c>Layer</c> của Phase 2–3? Không khác gì về toán. Chỉ khác ở
/// chỗ nó xử lý MỘT MA TRẬN token thay vì một vector, nên cache được đầu
/// vào của cả chuỗi — điều bắt buộc vì transformer phải forward hết mọi
/// token rồi mới backward.
///
/// BACKWARD (chain rule cho phép nhân ma trận):
///     dZ = dY ⊙ f'(Z)
///     dW = Xᵀ · dZ          db = tổng theo cột của dZ
///     dX = dZ · Wᵀ
/// </summary>
public sealed class Dense
{
    private readonly IActivation _activation;
    private Mat? _lastInput;
    private Mat? _lastPreActivation;

    public Param W { get; }

    public Param B { get; }

    public int InputSize => W.Value.Rows;

    public int OutputSize => W.Value.Cols;

    public Dense(string name, int inputSize, int outputSize, IActivation activation, Random rng)
    {
        ArgumentNullException.ThrowIfNull(activation);
        ArgumentNullException.ThrowIfNull(rng);

        _activation = activation;
        W = Param.Random($"{name}.W", inputSize, outputSize, rng);
        B = Param.Constant($"{name}.b", 1, outputSize, 0);
    }

    public Mat Forward(Mat x)
    {
        ArgumentNullException.ThrowIfNull(x);

        if (x.Cols != InputSize)
        {
            throw new ArgumentException($"Dense cần {InputSize} chiều nhưng nhận {x.Cols}.", nameof(x));
        }

        _lastInput = x;
        _lastPreActivation = x.Multiply(W.Value).AddRowVector(B.Value.Row(0));

        return _lastPreActivation.Map(_activation.Forward);
    }

    public Mat Backward(Mat gradientWrtOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientWrtOutput);

        if (_lastInput is null || _lastPreActivation is null)
        {
            throw new InvalidOperationException("Phải gọi Forward trước Backward.");
        }

        var dz = gradientWrtOutput.Hadamard(_lastPreActivation.Map(_activation.Derivative));

        W.AccumulateGrad(_lastInput.Transpose().Multiply(dz));
        B.AccumulateGrad(dz.ColumnSums());

        return dz.Multiply(W.Value.Transpose());
    }

    public IEnumerable<Param> Parameters() => [W, B];
}
