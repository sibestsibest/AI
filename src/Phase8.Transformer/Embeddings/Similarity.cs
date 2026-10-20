using Ai.Phase8.LinAlg;

namespace Ai.Phase8.Embeddings;

/// <summary>
/// Đo độ giống nhau giữa hai vector.
/// </summary>
public static class Similarity
{
    /// <summary>
    /// Cosine similarity — cos của góc giữa hai vector:
    ///
    ///     cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
    ///
    /// Giá trị từ −1 đến 1:
    ///      1  : cùng hướng hoàn toàn      (rất giống nhau)
    ///      0  : vuông góc                 (không liên quan)
    ///     −1  : ngược hướng hoàn toàn     (đối nghịch)
    ///
    /// Ví dụ số:
    ///     a = [1, 0]      b = [1, 0]   ->  1/(1·1)     = 1.0
    ///     a = [1, 0]      b = [0, 1]   ->  0/(1·1)     = 0.0
    ///     a = [1, 0]      b = [-1, 0]  -> −1/(1·1)     = −1.0
    ///     a = [3, 4]      b = [6, 8]   ->  50/(5·10)   = 1.0   (cùng hướng, khác độ dài)
    ///
    /// VÌ SAO CHIA CHO ĐỘ DÀI? Để chỉ so HƯỚNG, bỏ qua độ lớn. Từ xuất hiện
    /// nhiều thường có vector dài hơn; nếu dùng tích vô hướng thô thì các từ
    /// phổ biến sẽ luôn "giống" mọi thứ. Chuẩn hoá loại bỏ thiên lệch đó.
    /// </summary>
    public static double Cosine(Vec a, Vec b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Kích thước không khớp: {a.Length} và {b.Length}.");
        }

        double normProduct = a.Norm() * b.Norm();

        // Vector 0 không có hướng -> quy ước similarity = 0.
        return normProduct < 1e-12 ? 0 : a.Dot(b) / normProduct;
    }

    /// <summary>Khoảng cách Euclid: ‖a − b‖. Càng nhỏ càng giống.</summary>
    public static double EuclideanDistance(Vec a, Vec b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        return a.Subtract(b).Norm();
    }

    /// <summary>Tạo vector one-hot độ dài <paramref name="size"/> với số 1 ở vị trí index.</summary>
    public static Vec OneHot(int index, int size)
    {
        var v = new Vec(size);
        v[index] = 1;
        return v;
    }

    /// <summary>Các từ gần nhất với một từ cho trước, xếp theo cosine giảm dần.</summary>
    public static (string Word, double Score)[] NearestNeighbours(
        EmbeddingTable table, string word, int take = 3)
    {
        ArgumentNullException.ThrowIfNull(table);

        int self = table.Vocabulary.IndexOf(word);
        var target = table.Lookup(self);

        return Enumerable.Range(0, table.Count)
            .Where(i => i != self)
            .Select(i => (Word: table.Vocabulary.WordAt(i), Score: Cosine(target, table.Lookup(i))))
            .OrderByDescending(x => x.Score)
            .Take(take)
            .ToArray();
    }
}
