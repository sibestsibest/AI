using Ai.Phase11.LinAlg;

namespace Ai.Phase11.Embeddings;

/// <summary>
/// Bảng embedding: mỗi từ trong từ điển giữ MỘT vector số thực học được.
///
///             dim 0    dim 1    dim 2   ...
///     cat   [ 0.42 ,  -0.18 ,   0.91 , ... ]
///     dog   [ 0.39 ,  -0.21 ,   0.88 , ... ]   <- gần cat
///     car   [-0.77 ,   0.55 ,  -0.12 , ... ]   <- xa cat
///
/// VÌ SAO KHÔNG DÙNG ONE-HOT?
/// One-hot biểu diễn "cat" thành [1,0,0,...,0], "dog" thành [0,1,0,...,0].
/// Tích vô hướng của hai vector one-hot KHÁC NHAU luôn bằng 0, nên cosine
/// similarity giữa MỌI cặp từ đều bằng 0. One-hot không thể diễn đạt được
/// "cat gần dog hơn car" — nó chỉ nói được "cat khác dog", không hơn.
/// Thêm nữa, one-hot cần V chiều (V = cỡ từ điển, có thể 50.000).
///
/// Embedding nén xuống d chiều (thường 50–1000) và cho phép các từ ở GẦN
/// nhau trong không gian. Khoảng cách trở thành thước đo ngữ nghĩa.
///
/// Bảng này chính là một ma trận V × d, và nó được HỌC bằng gradient descent
/// giống hệt mọi trọng số khác — không ai gán tay các con số này.
/// </summary>
public sealed class EmbeddingTable
{
    private readonly Vec[] _vectors;
    private readonly Vec[] _gradients;

    public Vocabulary Vocabulary { get; }

    public int Dimensions { get; }

    public int Count => _vectors.Length;

    public EmbeddingTable(Vocabulary vocabulary, int dimensions, Random rng)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        ArgumentNullException.ThrowIfNull(rng);
        if (dimensions <= 0) throw new ArgumentOutOfRangeException(nameof(dimensions));

        Vocabulary = vocabulary;
        Dimensions = dimensions;

        _vectors = new Vec[vocabulary.Count];
        _gradients = new Vec[vocabulary.Count];

        // Khởi tạo nhỏ và ngẫu nhiên: mọi từ bắt đầu ở những vị trí gần gốc
        // toạ độ, chưa có quan hệ gì với nhau. Quá trình học sẽ đẩy chúng ra.
        double limit = 0.5 / Math.Sqrt(dimensions);
        for (int i = 0; i < vocabulary.Count; i++)
        {
            _vectors[i] = new Vec(dimensions);
            _gradients[i] = new Vec(dimensions);

            for (int d = 0; d < dimensions; d++)
            {
                _vectors[i][d] = (rng.NextDouble() * 2 - 1) * limit;
            }
        }
    }

    /// <summary>
    /// Tra cứu vector của một từ. Đây chính là "forward pass" của embedding:
    /// không nhân ma trận gì cả, chỉ đọc một hàng của bảng.
    ///
    /// (Về mặt toán học, tra cứu hàng i tương đương nhân ma trận với vector
    /// one-hot vị trí i — nhưng nhân như vậy tốn V·d phép tính để lấy ra đúng
    /// d con số, nên ai cũng cài bằng tra cứu.)
    /// </summary>
    public Vec Lookup(int index) => _vectors[index];

    public Vec Lookup(string word) => _vectors[Vocabulary.IndexOf(word)];

    public Vec GradientOf(int index) => _gradients[index];

    public void ZeroGradients()
    {
        foreach (var gradient in _gradients)
        {
            for (int d = 0; d < Dimensions; d++)
            {
                gradient[d] = 0;
            }
        }
    }

    /// <summary>Cộng dồn gradient cho một từ (nhiều cặp huấn luyện cùng chạm tới một từ).</summary>
    public void AccumulateGradient(int index, Vec delta)
    {
        ArgumentNullException.ThrowIfNull(delta);

        var gradient = _gradients[index];
        for (int d = 0; d < Dimensions; d++)
        {
            gradient[d] += delta[d];
        }
    }

    /// <summary>Toàn bộ bảng dưới dạng mảng phẳng — để optimizer xử lý như mọi tham số khác.</summary>
    public double[] GetParameters()
    {
        var p = new double[Count * Dimensions];
        int k = 0;

        foreach (var vector in _vectors)
        {
            for (int d = 0; d < Dimensions; d++) p[k++] = vector[d];
        }

        return p;
    }

    public void SetParameters(double[] parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);

        if (parameters.Length != Count * Dimensions)
        {
            throw new ArgumentException(
                $"Bảng có {Count * Dimensions} tham số nhưng nhận {parameters.Length}.");
        }

        int k = 0;
        foreach (var vector in _vectors)
        {
            for (int d = 0; d < Dimensions; d++) vector[d] = parameters[k++];
        }
    }

    public double[] GetGradients()
    {
        var g = new double[Count * Dimensions];
        int k = 0;

        foreach (var gradient in _gradients)
        {
            for (int d = 0; d < Dimensions; d++) g[k++] = gradient[d];
        }

        return g;
    }
}
