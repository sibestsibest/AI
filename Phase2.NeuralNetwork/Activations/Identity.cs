namespace Ai.Phase2.Activations;

/// <summary>
/// f(z) = z — "không làm gì cả".
///
/// Dùng cho layer OUTPUT khi bài toán là hồi quy (dự đoán một số thực bất kỳ),
/// vì sigmoid/tanh sẽ ép kết quả vào khoảng [0,1] hoặc [-1,1].
/// Một mạng chỉ gồm Identity chính là Linear Regression của Phase 1.
/// </summary>
public sealed class Identity : IActivation
{
    public string Name => "Identity";

    public double Forward(double z) => z;

    /// <summary>d/dz (z) = 1</summary>
    public double Derivative(double z) => 1.0;
}
