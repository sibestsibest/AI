namespace Ai.Phase13.Grounding;

/// <summary>
/// Kho tài liệu nội bộ, khai báo ở ĐÚNG MỘT CHỖ.
///
/// Phase 13 có hai thứ cùng cần danh sách này: <c>SearchTool</c> (đường trả
/// lời trực tiếp của ý định SearchKnowledge, có từ Phase 11) và
/// <see cref="KnowledgeStore"/> (kênh evidence mới của Phase 13). Nếu khai
/// báo hai lần thì chúng sẽ lệch nhau — và lệch theo cách khó thấy nhất:
/// agent trả lời một đằng khi hỏi trực tiếp, dẫn căn cứ một nẻo khi thu thập
/// evidence, mà không có gì báo lỗi.
///
/// Nên: một danh sách, hai chỗ nạp.
/// </summary>
public static class KnowledgeSeed
{
    public static readonly (string Topic, string Content)[] Documents =
    [
        ("Backpropagation",
            "Backpropagation là thuật toán tính gradient bằng chain rule, đi ngược từ loss về từng tham số. " +
            "Nó tốn 1 forward + 1 backward, không phụ thuộc số tham số."),

        ("Gradient descent",
            "Gradient descent cập nhật tham số theo công thức w = w - learningRate * gradient, " +
            "đi ngược hướng dốc lên của hàm loss."),

        ("Softmax",
            "Softmax biến một vector logits thành phân bố xác suất: e^z chia cho tổng e^z. " +
            "Mọi phần tử dương và tổng bằng 1."),

        ("Attention",
            "Attention tính softmax(Q·Kᵀ/√d)·V. Mỗi token dùng query của mình so với key của " +
            "mọi token để quyết định lấy bao nhiêu thông tin từ đâu."),

        ("Embedding",
            "Embedding gán cho mỗi từ một vector học được, nhờ đó khoảng cách giữa các vector " +
            "trở thành thước đo mức giống nhau về ngữ nghĩa."),

        ("Layer normalization",
            "Layer normalization chuẩn hoá từng token về trung bình 0 và phương sai 1, " +
            "rồi nhân gamma cộng beta học được, giữ cho độ lớn không trôi qua các layer."),

        ("Overfitting",
            "Overfitting là khi model học thuộc dữ liệu huấn luyện thay vì học quy luật, " +
            "nên chạy tốt trên tập train nhưng kém trên dữ liệu mới."),

        ("Loss function",
            "Loss function đo mức sai của model bằng một con số. Học chính là đi tìm bộ tham số " +
            "làm con số đó nhỏ nhất."),

        ("Transformer",
            "Transformer xếp chồng các khối gồm self attention và feed forward network, " +
            "kèm residual và layer normalization."),

        ("Neural network",
            "Neural network là nhiều neuron xếp thành layer nối tiếp nhau, mỗi neuron tính " +
            "tổng có trọng số rồi qua một hàm kích hoạt phi tuyến."),

        // --- Hai mục dưới đây là của Phase 13, và chúng có mục đích riêng ---
        //
        // Kho nội bộ CÓ tài liệu về SSRF và prompt injection, mà web cũng có.
        // Nhờ trùng chủ đề, demo và test chứng minh được điều quan trọng nhất
        // của Phase 13: khi hai kênh nói cùng một điều thì kênh ĐÃ KIỂM thắng,
        // và câu trả lời không còn phải cảnh báo "chưa kiểm chứng".
        ("SSRF",
            "SSRF là lỗ hổng khi máy chủ bị lừa gửi request tới địa chỉ nội bộ. " +
            "Cách chặn gồm danh sách trắng tên miền và soi mọi IP sau khi phân giải DNS."),

        ("Prompt injection",
            "Prompt injection là khi nội dung do người ngoài kiểm soát được model đọc như là chỉ thị. " +
            "Cách phòng chính là tách rạch ròi kênh chỉ thị và kênh dữ liệu."),
    ];

    /// <summary>Nạp vào kho kiến thức của Phase 13.</summary>
    public static KnowledgeStore BuildStore(Memory.IClock? clock = null)
    {
        var store = new KnowledgeStore(clock);

        foreach (var (topic, content) in Documents)
        {
            store.Add(topic, content);
        }

        return store;
    }
}
