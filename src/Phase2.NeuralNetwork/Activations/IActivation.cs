namespace Ai.Phase2.Activations;

/// <summary>
/// Hàm kích hoạt (activation): thứ làm cho mạng neural KHÁC một phép cộng tuyến tính.
///
/// Không có activation, xếp bao nhiêu layer cũng vô ích:
///     W₂·(W₁·x + b₁) + b₂  =  (W₂·W₁)·x + (W₂·b₁ + b₂)  =  W'·x + b'
/// tức là vẫn chỉ là MỘT layer tuyến tính. Activation bẻ cong đường thẳng đó,
/// nhờ vậy mạng mới học được quan hệ phi tuyến (Phase 4: XOR).
/// </summary>
public interface IActivation
{
    string Name { get; }

    /// <summary>a = f(z), với z là tổng có trọng số của neuron.</summary>
    double Forward(double z);

    /// <summary>
    /// f'(z) — độ dốc của activation tại z.
    /// Phase 2 chưa dùng tới, nhưng đây là một sự thật giải tích độc lập
    /// và đã có unit test đối chiếu với đạo hàm số. Phase 3 sẽ dùng nó
    /// trong chuỗi chain rule của backpropagation.
    /// </summary>
    double Derivative(double z);
}
