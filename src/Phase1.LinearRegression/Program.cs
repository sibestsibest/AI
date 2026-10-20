using Ai.Phase1;

// ======================================================================
// MINI AI — PHASE 1: LINEAR REGRESSION HỌC TỪ SỐ 0
//
// Không dùng thư viện AI nào. Toàn bộ là số học thuần C#.
// Model:  y_hat = w * x + b     (1 neuron, 1 input, không activation)
// ======================================================================

// Tham số huấn luyện — có thể truyền qua dòng lệnh: dotnet run -- <epochs> <learningRate>
int epochs = args.Length > 0 ? int.Parse(args[0]) : 1000;
double learningRate = args.Length > 1 ? double.Parse(args[1]) : 0.05;
const int LogEvery = 50;
const int Seed = 42; // cố định seed để chạy lại ra kết quả giống nhau

var data = Dataset.Training;
var model = new LinearRegressionModel(Seed);

// ----------------------------------------------------------------------
// 0. DỮ LIỆU — tất cả những gì model được nhìn thấy
// ----------------------------------------------------------------------
Console.WriteLine("=== PHASE 1: LINEAR REGRESSION ===\n");
Console.WriteLine("Dataset (model KHÔNG biết quy luật sinh ra nó):");
foreach (var s in data)
{
    Console.WriteLine($"  x = {s.X}  ->  y = {s.Y}");
}

// ----------------------------------------------------------------------
// 1. KHỞI TẠO NGẪU NHIÊN — model đoán bừa
// ----------------------------------------------------------------------
Console.WriteLine($"\n--- Khởi tạo ngẫu nhiên (seed = {Seed}) ---");
Console.WriteLine($"  w = {model.W:F6}");
Console.WriteLine($"  b = {model.B:F6}");
Console.WriteLine($"  Loss ban đầu = {model.CalculateLoss(data):F6}");
Console.WriteLine("  Dự đoán lúc chưa học:");
foreach (var s in data)
{
    Console.WriteLine($"    x = {s.X}  ->  y_hat = {model.Forward(s.X),9:F4}   (đúng: {s.Y})");
}

// ----------------------------------------------------------------------
// 2. KIỂM CHỨNG GRADIENT — chứng minh công thức đạo hàm viết tay là đúng
// ----------------------------------------------------------------------
Console.WriteLine("\n--- Gradient check (giải tích vs. đạo hàm số) ---");
var analytic = model.CalculateGradient(data);
var numeric = GradientCheck.Numerical(model.W, model.B, data);
Console.WriteLine($"  dL/dw : giải tích = {analytic.DwLoss,12:F8} | số = {numeric.DwLoss,12:F8} | lệch = {Math.Abs(analytic.DwLoss - numeric.DwLoss):E2}");
Console.WriteLine($"  dL/db : giải tích = {analytic.DbLoss,12:F8} | số = {numeric.DbLoss,12:F8} | lệch = {Math.Abs(analytic.DbLoss - numeric.DbLoss):E2}");

// ----------------------------------------------------------------------
// 3. BÓC TÁCH 3 BƯỚC HỌC ĐẦU TIÊN — nhìn rõ từng phép tính
// ----------------------------------------------------------------------
Console.WriteLine($"\n--- Chi tiết 3 bước gradient descent đầu tiên (lr = {learningRate}) ---");
var trace = new LinearRegressionModel(Seed); // bản sao riêng để minh hoạ, không ảnh hưởng model chính
for (int step = 1; step <= 3; step++)
{
    double lossBefore = trace.CalculateLoss(data);
    var g = trace.CalculateGradient(data);
    double wBefore = trace.W, bBefore = trace.B;

    trace.UpdateWeights(g, learningRate);

    Console.WriteLine($"  Bước {step}:");
    Console.WriteLine($"    Loss              = {lossBefore:F6}");
    Console.WriteLine($"    dL/dw             = {g.DwLoss:F6}   dL/db = {g.DbLoss:F6}");
    Console.WriteLine($"    w: {wBefore:F6} - {learningRate} * ({g.DwLoss:F6}) = {trace.W:F6}");
    Console.WriteLine($"    b: {bBefore:F6} - {learningRate} * ({g.DbLoss:F6}) = {trace.B:F6}");
}

// ----------------------------------------------------------------------
// 4. HUẤN LUYỆN
// ----------------------------------------------------------------------
Console.WriteLine($"\n--- Bắt đầu train: {epochs} epochs, learning rate = {learningRate} ---\n");
Console.WriteLine($"{"Epoch",6} | {"Loss",14} | {"Weight",10} | {"Bias",10} | {"Prediction x=5",14}");
Console.WriteLine(new string('-', 68));

model.Train(data, epochs, learningRate, snapshot =>
{
    bool isMilestone = snapshot.Epoch == 1
                       || snapshot.Epoch % LogEvery == 0
                       || snapshot.Epoch == epochs;

    if (!isMilestone) return;

    // Loss trong snapshot là Loss ĐO TRƯỚC khi cập nhật của epoch đó,
    // còn Weight/Bias là giá trị SAU khi cập nhật.
    double prediction = model.Predict(5);
    Console.WriteLine($"{snapshot.Epoch,6} | {snapshot.Loss,14:F8} | {snapshot.Weight,10:F6} | {snapshot.Bias,10:F6} | {prediction,14:F6}");
});

// ----------------------------------------------------------------------
// 5. KẾT QUẢ — model đã tự học ra quy luật gì?
// ----------------------------------------------------------------------
Console.WriteLine($"\n--- Kết quả sau {epochs} epochs ---");
Console.WriteLine($"  Model học được : {model}");
Console.WriteLine($"  Quy luật thật  : y = 2.000000 * x + 1.000000   (model chưa bao giờ được cho biết)");
Console.WriteLine($"  Sai số w       : {Math.Abs(model.W - 2):E3}");
Console.WriteLine($"  Sai số b       : {Math.Abs(model.B - 1):E3}");
Console.WriteLine($"  Loss cuối       : {model.CalculateLoss(data):E3}");

Console.WriteLine("\n  Dự đoán trên dữ liệu đã học:");
foreach (var s in data)
{
    double yHat = model.Predict(s.X);
    Console.WriteLine($"    x = {s.X}  ->  y_hat = {yHat,10:F6}   (đúng: {s.Y,2})   sai lệch = {Math.Abs(yHat - s.Y):E2}");
}

// ----------------------------------------------------------------------
// 6. TỔNG QUÁT HOÁ — dữ liệu model CHƯA từng thấy
// ----------------------------------------------------------------------
Console.WriteLine("\n  Dự đoán trên dữ liệu MỚI (chưa từng train):");
foreach (double x in new double[] { 6, 7, 10, 100, -3, 0.5 })
{
    Console.WriteLine($"    x = {x,6}  ->  y_hat = {model.Predict(x),12:F6}   (kỳ vọng: {2 * x + 1,10:F6})");
}

Console.WriteLine("\nModel đã tự tìm ra quy luật chỉ từ 5 cặp số, bằng gradient descent.");
