namespace Ai.Phase1;

/// <summary>Đạo hàm của Loss theo từng tham số của model.</summary>
public readonly record struct Gradient(double DwLoss, double DbLoss);

/// <summary>Ảnh chụp trạng thái model tại một epoch, dùng để in ra console.</summary>
public readonly record struct TrainingSnapshot(int Epoch, double Weight, double Bias, double Loss);

/// <summary>
/// Model tuyến tính 1 tham số đầu vào:   y_hat = w * x + b
///
/// Đây là mạng neural nhỏ nhất có thể có: 1 neuron, 1 input, không activation.
/// Toàn bộ vòng học được implement thủ công:
///     Forward -> CalculateLoss -> CalculateGradient -> UpdateWeights
/// </summary>
public sealed class LinearRegressionModel
{
    /// <summary>Trọng số (weight) — độ dốc của đường thẳng. Model phải tự học ra ≈ 2.</summary>
    public double W { get; private set; }

    /// <summary>Độ lệch (bias) — điểm cắt trục tung. Model phải tự học ra ≈ 1.</summary>
    public double B { get; private set; }

    /// <summary>
    /// Khởi tạo ngẫu nhiên. Model bắt đầu từ một đường thẳng "bừa bãi",
    /// hoàn toàn không biết gì về dữ liệu.
    /// </summary>
    public LinearRegressionModel(int? seed = null)
    {
        var rng = seed is null ? new Random() : new Random(seed.Value);

        // Random trong khoảng [-1, 1): đủ nhỏ để quá trình học ổn định.
        W = rng.NextDouble() * 2 - 1;
        B = rng.NextDouble() * 2 - 1;
    }

    /// <summary>
    /// Khởi tạo với tham số xác định. Dùng cho unit test (cần kết quả tất định)
    /// và để nạp lại một model đã train xong.
    /// </summary>
    public LinearRegressionModel(double w, double b)
    {
        W = w;
        B = b;
    }

    // ------------------------------------------------------------------
    // BƯỚC 1 — FORWARD: đưa input đi qua model để ra dự đoán
    // ------------------------------------------------------------------

    /// <summary>
    /// Lan truyền xuôi (forward pass): y_hat = w * x + b
    /// Đây là toàn bộ "suy nghĩ" của model — chỉ một phép nhân và một phép cộng.
    /// </summary>
    public double Forward(double x) => W * x + B;

    // ------------------------------------------------------------------
    // BƯỚC 2 — LOSS: đo model sai bao nhiêu
    // ------------------------------------------------------------------

    /// <summary>
    /// Mean Squared Error (MSE):
    ///
    ///     L = (1/n) * Σ (y_hat_i - y_i)^2
    ///
    /// Bình phương sai số để (a) sai âm và sai dương đều bị phạt,
    /// (b) phạt nặng các lỗi lớn, (c) hàm khả vi trơn -> lấy đạo hàm được.
    /// Loss = 0 nghĩa là model dự đoán đúng tuyệt đối.
    /// </summary>
    public double CalculateLoss(IReadOnlyList<Sample> data)
    {
        double sumSquaredError = 0;

        foreach (var sample in data)
        {
            double error = Forward(sample.X) - sample.Y;
            sumSquaredError += error * error;
        }

        return sumSquaredError / data.Count;
    }

    // ------------------------------------------------------------------
    // BƯỚC 3 — GRADIENT: tính hướng dốc của Loss theo w và b
    // ------------------------------------------------------------------

    /// <summary>
    /// Đạo hàm riêng của MSE theo w và b (chain rule — chính là backpropagation
    /// ở dạng đơn giản nhất, cho mạng chỉ có đúng một lớp):
    ///
    ///     L      = (1/n) * Σ e_i^2        với  e_i = (w*x_i + b) - y_i
    ///     dL/de  = (2/n) * e_i
    ///     de/dw  = x_i          de/db = 1
    ///  => dL/dw  = (2/n) * Σ e_i * x_i
    ///     dL/db  = (2/n) * Σ e_i
    ///
    /// Gradient trả lời câu hỏi: "nếu tăng w lên một chút thì Loss tăng hay giảm,
    /// và nhanh cỡ nào?" Dấu của nó cho biết phải đi về hướng ngược lại.
    /// </summary>
    public Gradient CalculateGradient(IReadOnlyList<Sample> data)
    {
        double sumErrorTimesX = 0;
        double sumError = 0;

        foreach (var sample in data)
        {
            double error = Forward(sample.X) - sample.Y;
            sumErrorTimesX += error * sample.X;
            sumError += error;
        }

        int n = data.Count;
        return new Gradient(
            DwLoss: 2.0 / n * sumErrorTimesX,
            DbLoss: 2.0 / n * sumError);
    }

    // ------------------------------------------------------------------
    // BƯỚC 4 — UPDATE: đi ngược hướng gradient một bước nhỏ
    // ------------------------------------------------------------------

    /// <summary>
    /// Gradient Descent:
    ///
    ///     w := w - learningRate * dL/dw
    ///     b := b - learningRate * dL/db
    ///
    /// Dấu trừ vì gradient chỉ hướng Loss TĂNG, mà ta muốn Loss GIẢM.
    /// learningRate là độ dài bước đi: quá nhỏ -> học chậm, quá lớn -> vọt qua
    /// đáy và phân kỳ (Loss bùng nổ thành NaN).
    /// </summary>
    public void UpdateWeights(Gradient gradient, double learningRate)
    {
        W -= learningRate * gradient.DwLoss;
        B -= learningRate * gradient.DbLoss;
    }

    // ------------------------------------------------------------------
    // BƯỚC 5 — TRAIN: lặp lại 4 bước trên nhiều lần
    // ------------------------------------------------------------------

    /// <summary>
    /// Vòng lặp huấn luyện. Mỗi epoch = một lần duyệt toàn bộ dataset
    /// (batch gradient descent: cập nhật một lần trên trung bình cả 5 mẫu).
    /// </summary>
    /// <param name="onEpoch">Callback để quan sát quá trình học.</param>
    public void Train(
        IReadOnlyList<Sample> data,
        int epochs,
        double learningRate,
        Action<TrainingSnapshot>? onEpoch = null)
    {
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            // Forward + Loss: model hiện tại sai bao nhiêu?
            double loss = CalculateLoss(data);

            // Backward: lỗi đó do w và b gây ra theo hướng nào?
            var gradient = CalculateGradient(data);

            // Learn: sửa w và b một chút theo hướng làm Loss giảm.
            UpdateWeights(gradient, learningRate);

            onEpoch?.Invoke(new TrainingSnapshot(epoch, W, B, loss));
        }
    }

    // ------------------------------------------------------------------
    // BƯỚC 6 — PREDICT: dùng model đã học cho dữ liệu mới
    // ------------------------------------------------------------------

    /// <summary>
    /// Suy luận (inference). Về mặt toán học giống hệt <see cref="Forward"/>,
    /// nhưng tách tên riêng vì ý nghĩa khác: forward là một bước bên trong
    /// vòng học, predict là mục đích cuối cùng của model.
    /// </summary>
    public double Predict(double x) => Forward(x);

    public override string ToString() => $"y = {W:F6} * x + {B:F6}";
}
