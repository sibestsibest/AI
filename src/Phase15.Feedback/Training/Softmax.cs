using Ai.Phase15.LinAlg;

namespace Ai.Phase15.Training;

/// <summary>
/// Softmax — biến một vector số thực bất kỳ (LOGITS) thành PHÂN BỐ XÁC SUẤT.
///
///     softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
///
/// Đầu ra luôn thoả hai tính chất:
///   • mọi phần tử &gt; 0
///   • tổng bằng đúng 1
/// nhờ vậy đọc được như xác suất.
///
/// VÍ DỤ SỐ với logits = [1, 2, 3]:
///     e¹ = 2.7183 , e² = 7.3891 , e³ = 20.0855     tổng = 30.1929
///     p  = [2.7183/30.1929, 7.3891/30.1929, 20.0855/30.1929]
///        = [0.0900, 0.2447, 0.6652]                 tổng = 1.0000
///
/// Để ý: hiệu logits chỉ là 1 và 2, nhưng xác suất chênh nhau 7.4 lần.
/// Hàm e^x khuếch đại khoảng cách — đó là ý nghĩa của chữ "soft-MAX":
/// nó gần như chọn ra phần tử lớn nhất, nhưng mượt và khả vi.
///
/// VÌ SAO KHÔNG CHỈ CHIA CHO TỔNG (zᵢ / Σzⱼ)?
///   • logits có thể âm -> ra "xác suất" âm, vô nghĩa
///   • tổng có thể bằng 0 -> chia cho 0
///   • không khuếch đại được khoảng cách
/// Hàm e^x giải quyết cả ba: luôn dương, không bao giờ triệt tiêu, và tăng
/// theo hàm mũ.
/// </summary>
public static class Softmax
{
    /// <summary>
    /// Áp softmax. Có TRỪ ĐI GIÁ TRỊ LỚN NHẤT trước khi lấy e^x:
    ///
    ///     softmax(z) = softmax(z − c)   với c là hằng số bất kỳ
    ///
    /// vì e^(zᵢ−c)/Σe^(zⱼ−c) = (e^(zᵢ)·e^(−c))/(e^(−c)·Σe^(zⱼ)) = softmax(z)ᵢ
    ///
    /// Chọn c = max(z) để số hạng lớn nhất thành e⁰ = 1. Không làm vậy thì
    /// logit cỡ 1000 sẽ cho e^1000 = Infinity và toàn bộ phép tính thành NaN.
    /// Đây là mẹo bắt buộc trong mọi thư viện thật.
    /// </summary>
    public static Vec Apply(Vec logits)
    {
        ArgumentNullException.ThrowIfNull(logits);
        if (logits.Length == 0) throw new ArgumentException("Logits không được rỗng.", nameof(logits));

        double max = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > max) max = logits[i];
        }

        var result = new Vec(logits.Length);
        double sum = 0;

        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = Math.Exp(logits[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < logits.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    /// <summary>
    /// Softmax có "nhiệt độ" — dùng khi sinh văn bản.
    ///
    ///     softmax(z / T)
    ///
    ///     T &lt; 1  : phân bố nhọn hơn, model bảo thủ, hay chọn từ khả năng cao nhất
    ///     T = 1  : phân bố gốc
    ///     T &gt; 1  : phân bố phẳng hơn, model "sáng tạo"/ngẫu nhiên hơn
    ///     T → 0  : luôn chọn từ có logit lớn nhất (greedy)
    /// </summary>
    public static Vec Apply(Vec logits, double temperature)
    {
        if (temperature <= 0) throw new ArgumentOutOfRangeException(nameof(temperature));

        return Apply(logits.Scale(1.0 / temperature));
    }
}
