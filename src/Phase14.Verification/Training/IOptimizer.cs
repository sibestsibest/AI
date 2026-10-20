namespace Ai.Phase14.Training;

/// <summary>
/// Optimizer: quyết định CÁCH dùng gradient để sửa tham số.
///
/// Gradient chỉ nói "hướng nào làm loss tăng". Đi bao xa, có nhớ các bước
/// trước không, mỗi tham số có tốc độ riêng không — đó là việc của optimizer.
/// Phase 2 chỉ có SGD (đúng quy tắc của Phase 1). Momentum/Adam để sau,
/// khi đã thấy rõ SGD yếu ở đâu.
///
/// Optimizer cố tình KHÔNG biết gì về mạng: nó chỉ thấy hai mảng số phẳng.
/// </summary>
public interface IOptimizer
{
    string Name { get; }

    double LearningRate { get; }

    /// <summary>Sửa <paramref name="parameters"/> tại chỗ dựa trên gradient.</summary>
    void Step(double[] parameters, double[] gradients);
}
