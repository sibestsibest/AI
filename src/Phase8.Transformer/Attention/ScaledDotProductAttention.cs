using Ai.Phase8.LinAlg;
using Ai.Phase8.Training;

namespace Ai.Phase8.Attention;

/// <summary>
/// Kết quả của một lượt attention, giữ lại cả các bước trung gian để soi.
/// </summary>
/// <param name="Scores">QKᵀ/√d — điểm thô, chưa chuẩn hoá (n × n).</param>
/// <param name="Weights">softmax theo từng hàng của Scores — mỗi hàng tổng bằng 1 (n × n).</param>
/// <param name="Output">Weights · V — kết quả cuối (n × dᵥ).</param>
public sealed record AttentionResult(Mat Scores, Mat Weights, Mat Output);

/// <summary>
/// SCALED DOT-PRODUCT ATTENTION
///
///     Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
///
/// PHÉP ẨN DỤ THƯ VIỆN — hiểu Q, K, V là gì:
///
///   Bạn đến thư viện tìm sách.
///     QUERY (Q)  = câu hỏi bạn mang theo: "tôi cần gì?"
///     KEY   (K)  = nhãn dán trên gáy mỗi cuốn sách: "tôi nói về gì?"
///     VALUE (V)  = nội dung thật bên trong cuốn sách.
///
///   Bạn so câu hỏi với TỪNG nhãn (Q·Kᵀ) để biết cuốn nào liên quan.
///   Chuẩn hoá thành tỉ lệ phần trăm chú ý (softmax).
///   Rồi lấy nội dung các cuốn sách, trộn lại theo đúng tỉ lệ đó (·V).
///
///   Khác với tra cứu từ điển thông thường (khớp đúng một khoá), attention
///   là tra cứu MỀM: lấy một chút từ mọi nguồn, theo tỉ lệ liên quan.
///
/// TRONG CÂU VĂN:
///   Với "con mèo ngồi trên thảm vì NÓ ấm", khi xử lý token "nó" thì query
///   của nó sẽ khớp mạnh với key của "thảm", nên vector đầu ra của "nó" hút
///   phần lớn thông tin từ "thảm". Đó là cách mô hình giải quyết đại từ —
///   không cần luật ngữ pháp nào.
///
/// VÌ SAO PHẢI CHIA CHO √d?
///   q·k là tổng của d số hạng. Nếu mỗi thành phần có phương sai 1 thì
///   phương sai của tích vô hướng là d, tức độ lệch chuẩn √d. Với d = 64,
///   điểm số văng trong khoảng ±8 hoặc hơn, softmax bão hoà thành gần
///   one-hot: một token chiếm ~100% chú ý, gradient của các token còn lại
///   tắt ngóm và mô hình ngừng học. Chia cho √d kéo phương sai về 1.
/// </summary>
public static class ScaledDotProductAttention
{
    /// <summary>
    /// Pseudocode:
    ///     scores  = (Q · Kᵀ) / √dₖ            # n×n : mỗi token chấm điểm mọi token
    ///     nếu có mask: scores[bị chặn] = −∞   # softmax sẽ biến thành 0
    ///     weights = softmax(mỗi HÀNG scores)  # n×n : mỗi hàng tổng bằng 1
    ///     output  = weights · V               # n×dᵥ: trộn V theo trọng số
    /// </summary>
    /// <param name="q">Query, n × dₖ — mỗi HÀNG là query của một token.</param>
    /// <param name="k">Key, m × dₖ.</param>
    /// <param name="v">Value, m × dᵥ.</param>
    /// <param name="causalMask">
    /// Nếu true, token vị trí i chỉ được nhìn các vị trí ≤ i. Bắt buộc với mô
    /// hình sinh văn bản: khi dự đoán từ thứ i, không được phép nhìn đáp án
    /// ở phía sau.
    /// </param>
    public static AttentionResult Compute(Mat q, Mat k, Mat v, bool causalMask = false)
    {
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(v);

        if (q.Cols != k.Cols)
        {
            throw new ArgumentException($"Q có {q.Cols} chiều nhưng K có {k.Cols} — phải bằng nhau.");
        }

        if (k.Rows != v.Rows)
        {
            throw new ArgumentException($"K có {k.Rows} hàng nhưng V có {v.Rows} — phải bằng nhau.");
        }

        // BƯỚC 1: điểm tương đồng giữa mọi cặp (query, key).
        //         scores[i,j] = "token i quan tâm token j đến mức nào"
        double scale = 1.0 / Math.Sqrt(q.Cols);
        var scores = q.Multiply(k.Transpose());

        for (int i = 0; i < scores.Rows; i++)
        {
            for (int j = 0; j < scores.Cols; j++)
            {
                scores[i, j] *= scale;

                // BƯỚC 2 (tuỳ chọn): che tương lai.
                // Dùng −∞ vì e^(−∞) = 0, nên softmax cho trọng số đúng bằng 0.
                if (causalMask && j > i)
                {
                    scores[i, j] = double.NegativeInfinity;
                }
            }
        }

        // BƯỚC 3: chuẩn hoá từng hàng thành phân bố xác suất.
        var weights = new Mat(scores.Rows, scores.Cols);
        for (int i = 0; i < scores.Rows; i++)
        {
            weights.SetRow(i, Softmax.Apply(scores.Row(i)));
        }

        // BƯỚC 4: trộn các vector V theo trọng số vừa tính.
        var output = weights.Multiply(v);

        return new AttentionResult(scores, weights, output);
    }

    /// <summary>Self-attention: Q, K, V đều lấy từ cùng một chuỗi đầu vào.</summary>
    public static AttentionResult SelfAttention(Mat x, bool causalMask = false) =>
        Compute(x, x, x, causalMask);
}
