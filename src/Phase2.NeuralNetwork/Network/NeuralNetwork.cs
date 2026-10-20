using System.Text;
using Ai.Phase2.LinAlg;

namespace Ai.Phase2.Network;

/// <summary>
/// Mạng neural: một chuỗi các layer nối tiếp nhau.
/// Đầu ra của layer trước là đầu vào của layer sau.
///
///     x -> [Layer 1] -> h -> [Layer 2] -> ŷ
///
/// Ngoài Forward, class này còn cho phép đọc/ghi toàn bộ tham số dưới dạng
/// MỘT mảng phẳng. Nhờ vậy optimizer và numerical gradient không cần biết gì
/// về cấu trúc mạng — chúng chỉ thấy một vector tham số duy nhất.
/// </summary>
public sealed class NeuralNetwork
{
    public Layer[] Layers { get; }

    public int InputSize => Layers[0].InputSize;

    public int OutputSize => Layers[^1].OutputSize;

    public NeuralNetwork(params Layer[] layers)
    {
        ArgumentNullException.ThrowIfNull(layers);
        if (layers.Length == 0) throw new ArgumentException("Mạng phải có ít nhất 1 layer.", nameof(layers));

        // Kiểm tra các layer khớp kích thước với nhau — bắt lỗi ngay lúc dựng
        // mạng thay vì để nó nổ ở giữa lúc train.
        for (int i = 1; i < layers.Length; i++)
        {
            if (layers[i].InputSize != layers[i - 1].OutputSize)
            {
                throw new ArgumentException(
                    $"Layer {i} cần {layers[i].InputSize} đầu vào nhưng layer {i - 1} " +
                    $"chỉ sinh ra {layers[i - 1].OutputSize}.", nameof(layers));
            }
        }

        Layers = layers;
    }

    /// <summary>
    /// Forward pass toàn mạng: đẩy dữ liệu qua từng layer theo thứ tự.
    /// </summary>
    public Vec Forward(Vec input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>Suy luận — giống Forward, tên khác vì ý nghĩa khác (xem Phase 1).</summary>
    public Vec Predict(Vec input) => Forward(input);

    /// <summary>Tiện ích cho mạng 1 đầu vào / 1 đầu ra.</summary>
    public double PredictScalar(double x) => Forward(Vec.Of(x))[0];

    // ------------------------------------------------------------------
    // Truy cập tham số dưới dạng mảng phẳng
    //
    // Thứ tự: theo layer, trong mỗi layer theo neuron, trong mỗi neuron là
    // các trọng số rồi đến bias. Thứ tự này cố định, nên GetParameters và
    // SetParameters luôn khớp nhau.
    // ------------------------------------------------------------------

    public int ParameterCount => Layers.Sum(l => l.ParameterCount);

    public double[] GetParameters()
    {
        var p = new double[ParameterCount];
        int k = 0;

        foreach (var layer in Layers)
        {
            foreach (var neuron in layer.Neurons)
            {
                for (int i = 0; i < neuron.InputSize; i++)
                {
                    p[k++] = neuron.Weights[i];
                }

                p[k++] = neuron.Bias;
            }
        }

        return p;
    }

    public void SetParameters(double[] parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);

        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Mạng có {ParameterCount} tham số nhưng nhận {parameters.Length}.",
                nameof(parameters));
        }

        int k = 0;
        foreach (var layer in Layers)
        {
            foreach (var neuron in layer.Neurons)
            {
                for (int i = 0; i < neuron.InputSize; i++)
                {
                    neuron.Weights[i] = parameters[k++];
                }

                neuron.Bias = parameters[k++];
            }
        }
    }

    /// <summary>Đọc gradient đang lưu trong các neuron ra mảng phẳng (cùng thứ tự).</summary>
    public double[] GetGradients()
    {
        var g = new double[ParameterCount];
        int k = 0;

        foreach (var layer in Layers)
        {
            foreach (var neuron in layer.Neurons)
            {
                for (int i = 0; i < neuron.InputSize; i++)
                {
                    g[k++] = neuron.WeightGradients[i];
                }

                g[k++] = neuron.BiasGradient;
            }
        }

        return g;
    }

    /// <summary>Ghi mảng gradient phẳng ngược vào từng neuron (cùng thứ tự).</summary>
    public void SetGradients(double[] gradients)
    {
        ArgumentNullException.ThrowIfNull(gradients);

        if (gradients.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Mạng có {ParameterCount} tham số nhưng nhận {gradients.Length} gradient.",
                nameof(gradients));
        }

        int k = 0;
        foreach (var layer in Layers)
        {
            foreach (var neuron in layer.Neurons)
            {
                for (int i = 0; i < neuron.InputSize; i++)
                {
                    neuron.WeightGradients[i] = gradients[k++];
                }

                neuron.BiasGradient = gradients[k++];
            }
        }
    }

    public string Describe()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"NeuralNetwork: {InputSize} đầu vào -> {OutputSize} đầu ra, " +
                      $"{Layers.Length} layer, {ParameterCount} tham số");

        for (int i = 0; i < Layers.Length; i++)
        {
            sb.AppendLine($"  [{i}] {Layers[i]}");
        }

        return sb.ToString().TrimEnd();
    }

    public override string ToString() => Describe();
}
