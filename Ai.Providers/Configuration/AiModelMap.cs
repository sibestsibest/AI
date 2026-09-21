using Ai.Providers.Contracts;

namespace Ai.Providers.Configuration;

/// <summary>
/// MODEL THEO LOẠI VIỆC — để không có tên model nào bị ghi cứng trong code.
///
/// Ứng với mục "AI:Models:*" của đề bài:
///
///     Chat / Reasoning / Research / Coding / Vision / Embedding / ImageGeneration
///
/// Mọi trường đều CÓ THỂ BỎ TRỐNG, và bỏ trống nghĩa là "dùng DefaultModel của
/// provider". Đây là điểm quan trọng: một cấu hình chỉ khai đúng một model vẫn
/// phải chạy được mọi loại việc. Bắt khai đủ bảy dòng mới chạy được là cách làm
/// người ta dán bừa cùng một tên vào cả bảy chỗ.
///
/// KHÔNG có giá trị mặc định nào là tên một model thật. Tên model đổi theo
/// thời gian và theo nhà cung cấp; ghi sẵn một cái vào đây là hẹn ngày nó sai.
/// Thiếu cấu hình thì hệ thống báo rõ, chứ không đoán.
/// </summary>
public sealed record AiModelMap
{
    public string? Chat { get; init; }

    public string? Reasoning { get; init; }

    public string? Research { get; init; }

    public string? Coding { get; init; }

    public string? Vision { get; init; }

    public string? Embedding { get; init; }

    public string? ImageGeneration { get; init; }

    /// <summary>Model cho một loại việc, hoặc null để provider dùng mặc định của nó.</summary>
    public string? For(AiTaskType task) => task switch
    {
        AiTaskType.Reasoning => Reasoning ?? Chat,
        AiTaskType.WebResearch => Research ?? Chat,
        AiTaskType.Coding => Coding ?? Chat,
        AiTaskType.DocumentAnalysis => Vision ?? Chat,
        AiTaskType.ImageGeneration => ImageGeneration,

        // Tóm tắt, dịch, gọi công cụ và chat đều là việc sinh chữ thông thường.
        // Tách chúng ra thành từng dòng cấu hình riêng chỉ làm file dài thêm mà
        // không cho ai thêm quyền điều khiển nào có ích.
        _ => Chat,
    };

    /// <summary>Mọi model được khai — dùng để dựng danh sách cho phép của provider.</summary>
    public IReadOnlyList<string> All =>
        [.. new[] { Chat, Reasoning, Research, Coding, Vision, Embedding, ImageGeneration }
            .Where(m => !string.IsNullOrWhiteSpace(m))
            .Select(m => m!)
            .Distinct(StringComparer.OrdinalIgnoreCase)];

    public bool IsEmpty => All.Count == 0;

    public override string ToString() => IsEmpty
        ? "(chưa khai model nào — dùng DefaultModel của provider)"
        : string.Join(", ", All);
}
