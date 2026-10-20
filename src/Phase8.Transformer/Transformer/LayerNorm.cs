using Ai.Phase8.LinAlg;

namespace Ai.Phase8.Transformer;

/// <summary>
/// Layer Normalization — chuẩn hoá TỪNG TOKEN theo các chiều của chính nó.
///
///     μ  = trung bình của d giá trị trong một token
///     σ² = phương sai của chúng
///     x̂  = (x − μ) / √(σ² + ε)          -> trung bình 0, phương sai 1
///     y  = γ ⊙ x̂ + β                    -> γ, β học được
///
/// VÌ SAO CẦN?
/// Transformer cộng dồn rất nhiều thứ vào cùng một vector (residual, attention,
/// FFN). Không chuẩn hoá thì độ lớn của vector trôi dần qua từng layer: hoặc
/// nổ lên vô cực, hoặc teo về 0, và gradient đi theo. LayerNorm kéo mọi token
/// về cùng một thang đo sau mỗi bước.
///
/// VÌ SAO CÓ γ VÀ β? Ép cứng về trung bình 0 / phương sai 1 là quá thô — có
/// khi layer CẦN đầu ra lệch đi. γ và β cho mạng quyền tự điều chỉnh lại thang
/// đo, kể cả huỷ bỏ hoàn toàn việc chuẩn hoá nếu điều đó có lợi.
///
/// KHÁC BATCH NORM Ở ĐÂU? BatchNorm chuẩn hoá theo chiều "các mẫu trong batch",
/// nên phụ thuộc batch size và rắc rối khi suy luận từng câu một. LayerNorm
/// chuẩn hoá trong nội bộ MỘT token, hoàn toàn độc lập với các token khác và
/// với batch — hợp với dữ liệu chuỗi độ dài thay đổi.
///
/// BACKWARD (dẫn từ chain rule, chú ý μ và σ đều PHỤ THUỘC x):
///     dγ  = Σ_tokens dy ⊙ x̂           dβ = Σ_tokens dy
///     dx̂  = dy ⊙ γ
///     dx  = (1/σ)·( dx̂ − mean(dx̂) − x̂ · mean(dx̂ ⊙ x̂) )
/// Hai số hạng trừ đi chính là đóng góp gián tiếp qua μ và σ.
/// </summary>
public sealed class LayerNorm
{
    private const double Epsilon = 1e-5;

    private Mat? _normalized;      // x̂
    private double[]? _stdDev;     // σ của từng token

    public Param Gamma { get; }

    public Param Beta { get; }

    public int Dimensions => Gamma.Value.Cols;

    public LayerNorm(string name, int dimensions)
    {
        // γ = 1, β = 0 -> lúc đầu LayerNorm chỉ chuẩn hoá, không bóp méo gì.
        Gamma = Param.Constant($"{name}.gamma", 1, dimensions, 1);
        Beta = Param.Constant($"{name}.beta", 1, dimensions, 0);
    }

    public Mat Forward(Mat x)
    {
        ArgumentNullException.ThrowIfNull(x);

        if (x.Cols != Dimensions)
        {
            throw new ArgumentException($"LayerNorm cần {Dimensions} chiều nhưng nhận {x.Cols}.", nameof(x));
        }

        int n = x.Rows;
        int d = x.Cols;

        _normalized = new Mat(n, d);
        _stdDev = new double[n];
        var output = new Mat(n, d);

        for (int i = 0; i < n; i++)
        {
            double mean = 0;
            for (int j = 0; j < d; j++) mean += x[i, j];
            mean /= d;

            double variance = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = x[i, j] - mean;
                variance += diff * diff;
            }

            variance /= d;

            double std = Math.Sqrt(variance + Epsilon);
            _stdDev[i] = std;

            for (int j = 0; j < d; j++)
            {
                _normalized[i, j] = (x[i, j] - mean) / std;
                output[i, j] = Gamma.Value[0, j] * _normalized[i, j] + Beta.Value[0, j];
            }
        }

        return output;
    }

    public Mat Backward(Mat gradientWrtOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientWrtOutput);

        if (_normalized is null || _stdDev is null)
        {
            throw new InvalidOperationException("Phải gọi Forward trước Backward.");
        }

        int n = gradientWrtOutput.Rows;
        int d = gradientWrtOutput.Cols;
        var dx = new Mat(n, d);

        for (int i = 0; i < n; i++)
        {
            // Gradient của γ và β: cộng dồn qua mọi token.
            for (int j = 0; j < d; j++)
            {
                Gamma.Grad[0, j] += gradientWrtOutput[i, j] * _normalized[i, j];
                Beta.Grad[0, j] += gradientWrtOutput[i, j];
            }

            // dx̂ = dy ⊙ γ
            var dNorm = new double[d];
            for (int j = 0; j < d; j++)
            {
                dNorm[j] = gradientWrtOutput[i, j] * Gamma.Value[0, j];
            }

            double meanDNorm = dNorm.Average();

            double meanDNormTimesNorm = 0;
            for (int j = 0; j < d; j++) meanDNormTimesNorm += dNorm[j] * _normalized[i, j];
            meanDNormTimesNorm /= d;

            for (int j = 0; j < d; j++)
            {
                dx[i, j] = (dNorm[j] - meanDNorm - _normalized[i, j] * meanDNormTimesNorm) / _stdDev[i];
            }
        }

        return dx;
    }

    public IEnumerable<Param> Parameters() => [Gamma, Beta];
}
