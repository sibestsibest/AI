namespace Ai.Providers.Contracts;

/// <summary>
/// MỘT LƯỢC ĐỒ mà câu trả lời phải khớp.
///
/// <see cref="Strict"/> mặc định BẬT. Ở chế độ đó provider bảo đảm kết quả khớp
/// lược đồ ở phía nó, nên phía ta đỡ được phần lớn lỗi phân tích. Nhưng lược đồ
/// nghiêm có ràng buộc riêng (mọi trường phải nằm trong "required",
/// "additionalProperties" phải là false), nên tắt được — dành cho lược đồ có
/// trường tuỳ chọn thật sự.
/// </summary>
public sealed record StructuredFormat(string Name, string JsonSchema)
{
    public bool Strict { get; init; } = true;

    public string? Description { get; init; }
}

/// <summary>
/// MỘT VÒNG model-xin-gọi-công-cụ rồi nhận kết quả — để gửi lại cho lượt sau.
///
/// Giữ CẢ lời xin lẫn kết quả, vì model cần thấy chính xác nó đã xin gì. Gửi
/// mỗi kết quả mà không gửi lời xin thì model không nối được kết quả với lượt
/// gọi nào, và hợp đồng của nhiều nhà cung cấp cũng từ chối thẳng.
/// </summary>
public sealed record AiToolExchange(AiToolCall Call, AiToolResult Result);
