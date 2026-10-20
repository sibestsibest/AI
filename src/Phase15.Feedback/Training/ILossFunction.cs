using Ai.Phase15.LinAlg;

namespace Ai.Phase15.Training;

/// <summary>
/// Hàm mất mát: đo model sai bao nhiêu so với đáp án đúng.
///
/// Tách thành interface vì Phase 6 sẽ cần Cross Entropy cho bài toán
/// dự đoán từ, trong khi Phase 2–4 dùng MSE. Cùng một mạng, đổi loss là đổi
/// cả mục tiêu học.
/// </summary>
public interface ILossFunction
{
    string Name { get; }

    /// <summary>Một số duy nhất: càng nhỏ càng tốt, 0 là hoàn hảo.</summary>
    double Compute(Vec predicted, Vec target);

    /// <summary>
    /// ∂L/∂ŷ — đạo hàm của loss theo TỪNG đầu ra của mạng.
    /// Đây là mắt xích ĐẦU TIÊN của chuỗi backpropagation: nó nói cho layer
    /// cuối biết "đầu ra của anh nên tăng hay giảm". Phase 3 sẽ nối tiếp
    /// chuỗi này ngược về các layer trước.
    /// </summary>
    Vec Gradient(Vec predicted, Vec target);
}
