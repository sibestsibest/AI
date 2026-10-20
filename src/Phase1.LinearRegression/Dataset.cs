namespace Ai.Phase1;

/// <summary>
/// Một mẫu dữ liệu: đầu vào X và đáp án đúng Y (ground truth).
/// Model KHÔNG biết công thức sinh ra Y, nó chỉ thấy từng cặp (X, Y).
/// </summary>
public readonly record struct Sample(double X, double Y);

public static class Dataset
{
    /// <summary>
    /// Dữ liệu huấn luyện. Con người biết quy luật là y = 2x + 1,
    /// nhưng model chỉ nhận được 5 cặp số này và phải tự suy ra quy luật.
    /// </summary>
    public static readonly Sample[] Training =
    [
        new(1, 3),
        new(2, 5),
        new(3, 7),
        new(4, 9),
        new(5, 11),
    ];
}
