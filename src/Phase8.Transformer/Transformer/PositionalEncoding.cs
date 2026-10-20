using Ai.Phase8.LinAlg;

namespace Ai.Phase8.Transformer;

/// <summary>
/// Mã hoá vị trí bằng sin/cos (Vaswani 2017).
///
/// VẤN ĐỀ: attention hoàn toàn KHÔNG BIẾT thứ tự. Nó tính tích vô hướng giữa
/// mọi cặp token; đảo thứ tự các token thì tập hợp các tích vô hướng vẫn y
/// nguyên, chỉ hoán vị hàng. Với attention thuần, "mèo đuổi chuột" và
/// "chuột đuổi mèo" là một.
///
/// GIẢI PHÁP: cộng thêm vào mỗi embedding một "dấu vân tay vị trí":
///
///     PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
///     PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
///
/// Mỗi cặp chiều (2i, 2i+1) là một cây kim đồng hồ quay với tốc độ riêng:
/// chiều đầu quay rất nhanh (phân biệt vị trí liền kề), chiều cuối quay rất
/// chậm (phân biệt đầu câu với cuối câu). Gộp lại, mỗi vị trí có một mẫu
/// hoàn toàn riêng biệt.
///
/// VÌ SAO KHÔNG ĐƠN GIẢN DÙNG SỐ 0, 1, 2, 3...? Vì giá trị sẽ lớn không giới
/// hạn với câu dài, lấn át hoàn toàn tín hiệu embedding. Sin/cos luôn nằm
/// trong [−1, 1].
///
/// VÌ SAO KHÔNG HỌC LUÔN VECTOR VỊ TRÍ? Cũng được — GPT làm vậy. Ưu điểm của
/// sin/cos là không tốn tham số và ngoại suy được sang độ dài chưa từng thấy,
/// vì PE(pos+k) biểu diễn được tuyến tính qua PE(pos).
///
/// Đây là bảng CỐ ĐỊNH, không có tham số, nên backward chỉ việc cho gradient
/// đi thẳng qua (đạo hàm của phép cộng hằng số bằng 1).
/// </summary>
public static class PositionalEncoding
{
    public static Mat Build(int maxLength, int dimensions)
    {
        if (maxLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxLength));
        if (dimensions <= 0) throw new ArgumentOutOfRangeException(nameof(dimensions));

        var encoding = new Mat(maxLength, dimensions);

        for (int pos = 0; pos < maxLength; pos++)
        {
            for (int i = 0; i < dimensions; i++)
            {
                // Cặp chiều (2k, 2k+1) dùng chung một tần số.
                double exponent = 2.0 * (i / 2) / dimensions;
                double angle = pos / Math.Pow(10000, exponent);

                encoding[pos, i] = i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
            }
        }

        return encoding;
    }

    /// <summary>Cộng mã vị trí vào n hàng đầu của chuỗi embedding.</summary>
    public static Mat Add(Mat embeddings, Mat encoding)
    {
        ArgumentNullException.ThrowIfNull(embeddings);
        ArgumentNullException.ThrowIfNull(encoding);

        if (embeddings.Rows > encoding.Rows)
        {
            throw new ArgumentException(
                $"Chuỗi dài {embeddings.Rows} token nhưng bảng vị trí chỉ có {encoding.Rows} dòng.");
        }

        var result = new Mat(embeddings.Rows, embeddings.Cols);
        for (int r = 0; r < embeddings.Rows; r++)
        {
            for (int c = 0; c < embeddings.Cols; c++)
            {
                result[r, c] = embeddings[r, c] + encoding[r, c];
            }
        }

        return result;
    }
}
