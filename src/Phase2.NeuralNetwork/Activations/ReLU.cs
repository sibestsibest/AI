namespace Ai.Phase2.Activations;

/// <summary>
/// f(z) = max(0, z)  — số âm thành 0, số dương giữ nguyên.
///
/// Đơn giản đến mức khó tin nhưng là activation phổ biến nhất trong deep
/// learning hiện đại, vì với z > 0 đạo hàm bằng đúng 1 — gradient không bị
/// teo dần khi đi qua nhiều layer (vanishing gradient), khác hẳn sigmoid
/// (đạo hàm tối đa chỉ 0.25).
///
/// Đạo hàm:  f'(z) = 1 nếu z > 0, ngược lại 0.
/// Tại z = 0 hàm không khả vi; quy ước lấy 0 — thực tế không ảnh hưởng gì
/// vì xác suất z rơi đúng 0 là ~0.
/// </summary>
public sealed class ReLU : IActivation
{
    public string Name => "ReLU";

    public double Forward(double z) => z > 0 ? z : 0;

    public double Derivative(double z) => z > 0 ? 1 : 0;
}
