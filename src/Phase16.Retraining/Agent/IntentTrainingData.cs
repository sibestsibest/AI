namespace Ai.Phase16.Agent;

/// <summary>
/// Ví dụ có nhãn để dạy bộ phân loại ý định.
///
/// Đây là toàn bộ "dữ liệu huấn luyện" của agent. Nó nhỏ đến mức đọc hết
/// được trong một phút — và chính vì thế bạn thấy rõ agent BIẾT gì và
/// KHÔNG BIẾT gì. Một LLM thật học từ hàng nghìn tỉ token, nên không ai
/// trả lời được câu hỏi đó nữa.
/// </summary>
public static class IntentTrainingData
{
    public static readonly (string Text, Intent Label)[] Examples =
    [
        // --- Calculate ---
        ("tính 2 cộng 3", Intent.Calculate),
        ("tính tổng của A và B", Intent.Calculate),
        ("Total bằng bao nhiêu", Intent.Calculate),
        ("kết quả của phép nhân", Intent.Calculate),
        ("cho biết giá trị của C", Intent.Calculate),
        ("calculate the sum", Intent.Calculate),
        ("what is the result", Intent.Calculate),
        ("tính giúp tôi phép chia", Intent.Calculate),
        ("A cộng B bằng mấy", Intent.Calculate),
        ("compute the total value", Intent.Calculate),

        // --- RememberFact ---
        ("hãy nhớ tôi dùng C#", Intent.RememberFact),
        ("ghi nhớ rằng tôi thích Gundam", Intent.RememberFact),
        ("nhớ giúp tôi làm việc với Azure", Intent.RememberFact),
        ("lưu lại thông tin này", Intent.RememberFact),
        ("remember that I use dotnet", Intent.RememberFact),
        ("please remember my preference", Intent.RememberFact),
        ("ghi nhớ sở thích của tôi", Intent.RememberFact),
        ("hãy lưu lại điều sau", Intent.RememberFact),
        ("nhớ rằng tôi ở Hà Nội", Intent.RememberFact),
        ("save this information for later", Intent.RememberFact),

        // --- RecallMemory ---
        ("tôi dùng ngôn ngữ nào", Intent.RecallMemory),
        ("bạn nhớ gì về tôi", Intent.RecallMemory),
        ("tôi thích thứ gì", Intent.RecallMemory),
        ("bạn biết gì về sở thích của tôi", Intent.RecallMemory),
        ("what do you know about me", Intent.RecallMemory),
        ("what did I tell you", Intent.RecallMemory),
        ("tôi làm việc với công nghệ nào", Intent.RecallMemory),
        ("nhắc lại thông tin của tôi", Intent.RecallMemory),
        ("do you recall my preference", Intent.RecallMemory),
        ("tôi đã nói gì trước đó", Intent.RecallMemory),

        // --- SearchKnowledge ---
        ("backpropagation là gì", Intent.SearchKnowledge),
        ("giải thích gradient descent", Intent.SearchKnowledge),
        ("softmax hoạt động thế nào", Intent.SearchKnowledge),
        ("định nghĩa của attention", Intent.SearchKnowledge),
        ("explain what embedding means", Intent.SearchKnowledge),
        ("tìm tài liệu về neural network", Intent.SearchKnowledge),
        ("khái niệm overfitting nghĩa là sao", Intent.SearchKnowledge),
        ("what does layer normalization do", Intent.SearchKnowledge),
        ("cho tôi biết khái niệm loss function", Intent.SearchKnowledge),
        ("tra cứu định nghĩa transformer", Intent.SearchKnowledge),

        // --- QueryDatabase ---
        ("có bao nhiêu đơn hàng", Intent.QueryDatabase),
        ("doanh thu tổng cộng là bao nhiêu", Intent.QueryDatabase),
        ("liệt kê các đơn hàng", Intent.QueryDatabase),
        ("đếm số khách hàng", Intent.QueryDatabase),
        ("how many orders are there", Intent.QueryDatabase),
        ("list all customers in the table", Intent.QueryDatabase),
        ("truy vấn bảng đơn hàng", Intent.QueryDatabase),
        ("tổng doanh thu của bảng orders", Intent.QueryDatabase),
        ("show me the orders table", Intent.QueryDatabase),
        ("đơn hàng lớn nhất là đơn nào", Intent.QueryDatabase),

        // --- SmallTalk ---
        ("xin chào", Intent.SmallTalk),
        ("chào bạn", Intent.SmallTalk),
        ("bạn khoẻ không", Intent.SmallTalk),
        ("hello there", Intent.SmallTalk),
        ("hi how are you", Intent.SmallTalk),
        ("bạn là ai", Intent.SmallTalk),
        ("bạn làm được những gì", Intent.SmallTalk),
        ("what can you do", Intent.SmallTalk),
        ("giới thiệu về bản thân bạn", Intent.SmallTalk),
        ("tạm biệt", Intent.SmallTalk),

        // --- SearchInternet (Phase 12) ---
        //
        // Từ phân biệt với SearchKnowledge là các từ chỉ NGUỒN NGOÀI và chỉ
        // TÍNH MỚI: "mạng", "internet", "online", "web", "mới nhất", "hiện
        // tại", "tin tức", "năm nay". Chúng không xuất hiện ở bất kỳ ví dụ
        // SearchKnowledge nào — nhờ vậy bag-of-words tách được hai ý định dù
        // cả hai đều là "đi tìm thông tin".
        ("tìm trên mạng thông tin về .NET 10", Intent.SearchInternet),
        ("tra trên internet giúp tôi", Intent.SearchInternet),
        ("tin tức mới nhất về trí tuệ nhân tạo", Intent.SearchInternet),
        ("lên mạng xem giá bitcoin hiện tại", Intent.SearchInternet),
        ("search the web for dotnet 10 release notes", Intent.SearchInternet),
        ("look this up online", Intent.SearchInternet),
        ("phiên bản mới nhất hiện nay là bản nào", Intent.SearchInternet),
        ("tìm kiếm trực tuyến về C# 14", Intent.SearchInternet),
        ("what is the latest news online", Intent.SearchInternet),
        ("tra cứu trên web tin mới năm nay", Intent.SearchInternet),
    ];

    public static IEnumerable<string> Texts => Examples.Select(e => e.Text);
}
