namespace Ai.Providers.Contracts;

/// <summary>
/// MỘT LƯỢT MODEL XIN GỌI CÔNG CỤ — lời đề nghị, không phải lệnh.
///
/// Ba trường, và cả ba đều cần thiết:
///
///   CallId         mã do provider cấp, phải trả lại nguyên văn kèm kết quả
///   Name           tên công cụ model muốn gọi
///   ArgumentsJson  tham số model tự soạn, để NGUYÊN dạng chuỗi JSON
///
/// VÌ SAO ARGUMENTS Ở DẠNG CHUỖI THÔ? Vì đây là dữ liệu KHÔNG TIN CẬY: model
/// soạn ra nó, và nó có thể sai kiểu, thiếu trường, hoặc chứa thứ không ai
/// mong đợi. Chuyển sang đối tượng có kiểu ngay tại đây sẽ làm lỗi phân tích
/// nổ ra trong lớp provider — chỗ không biết gì về công cụ và không sửa được
/// gì. Nơi kiểm tham số là <c>ToolRegistry</c>, ngay trước khi chạy.
///
/// <see cref="CallId"/> có thể rỗng: một số hợp đồng (kiểu OpenAI cũ) không
/// cấp mã riêng cho từng lượt gọi. Lúc đó tên công cụ đóng vai mã, và chỉ gọi
/// được một công cụ mỗi lượt.
/// </summary>
public sealed record AiToolCall(string CallId, string Name, string ArgumentsJson)
{
    public override string ToString() =>
        $"{Name}({(ArgumentsJson.Length > 80 ? ArgumentsJson[..80] + "…" : ArgumentsJson)})";
}

/// <summary>
/// KẾT QUẢ CHẠY MỘT CÔNG CỤ — thứ được gửi NGƯỢC lại cho model.
///
/// <see cref="Output"/> luôn là chuỗi, kể cả khi hỏng. Model cần biết công cụ
/// đã hỏng và hỏng vì sao để nó đổi cách làm; nuốt lỗi rồi gửi chuỗi rỗng sẽ
/// khiến nó gọi lại đúng công cụ đó với đúng tham số đó, mãi mãi.
///
/// <see cref="Succeeded"/> KHÔNG được suy ra từ nội dung: ứng dụng cần đếm
/// riêng số lượt hỏng, và "chuỗi có chứa chữ lỗi" không phải là một phép đo.
/// </summary>
public sealed record AiToolResult(string CallId, string Name, bool Succeeded, string Output)
{
    /// <summary>Công cụ bị TỪ CHỐI chạy (chưa đăng ký, hoặc không đủ quyền) — khác với chạy rồi hỏng.</summary>
    public bool WasRefused { get; init; }

    public TimeSpan Duration { get; init; }

    public override string ToString() =>
        Succeeded ? $"{Name}: OK ({Output.Length} ký tự)" : $"{Name}: HỎNG — {Output}";
}
