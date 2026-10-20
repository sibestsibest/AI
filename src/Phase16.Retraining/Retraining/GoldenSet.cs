using Ai.Phase16.Agent;
using Ai.Phase16.Learning;

namespace Ai.Phase16.Retraining;

/// <summary>
/// BỘ VÀNG — những câu KHÔNG BAO GIỜ được phép hỏng.
///
/// Mỗi ý định một câu, viết tay, diễn đạt khác hẳn các ví dụ trong tập huấn
/// luyện. Đây là danh sách "những việc hệ thống phải làm được", do người chọn,
/// và nó không đổi khi tập dữ liệu lớn lên.
///
/// VÌ SAO CẦN, KHI ĐÃ CÓ TẬP ĐÁNH GIÁ?
///
/// Vì tập đánh giá là một mẫu ngẫu nhiên và nó TRÔI theo dữ liệu: mỗi lần
/// duyệt thêm phản hồi, thành phần của nó đổi, nên "accuracy 92%" của hôm nay
/// và của tháng trước không hẳn nói về cùng một thứ. Bộ vàng thì cố định, nên
/// nó trả lời được câu hỏi mà tập đánh giá không trả lời được: *chức năng cơ
/// bản có còn chạy không*.
///
/// Và quan trọng hơn: accuracy trung bình cho phép đánh đổi. Một model làm
/// hỏng hẳn ý định <c>Calculate</c> nhưng khá lên ở sáu ý định còn lại vẫn có
/// thể có điểm cao hơn. Bộ vàng không cho đánh đổi kiểu đó — sai một câu là
/// KHÔNG triển khai, bất kể điểm tổng đẹp đến đâu.
///
/// LUẬT ĐI KÈM: bộ vàng phải NẰM NGOÀI tập huấn luyện.
///
/// Nếu một câu trong bộ vàng cũng có trong dữ liệu train thì model đã học
/// thuộc chính câu nó sắp bị kiểm, và bộ vàng không còn đo khả năng khái quát
/// hoá nữa — nó chỉ xác nhận model nhớ được bài đã học. <see cref="OverlapWith"/>
/// tồn tại để luật đó kiểm tra được bằng test, chứ không phải chỉ được hứa.
/// </summary>
public static class GoldenSet
{
    /// <summary>
    /// Bảy câu, một câu cho mỗi ý định.
    ///
    /// Cố ý dùng từ ngữ khác với <c>IntentTrainingData</c>: nếu viết na ná các
    /// ví dụ đã train thì bộ vàng chỉ đo trí nhớ, không đo khả năng hiểu.
    /// </summary>
    public static readonly EvaluationExample[] Examples =
    [
        new("cho tôi kết quả phép tính này", Intent.Calculate),
        new("ghi nhớ giúp tôi điều vừa nói", Intent.RememberFact),
        new("trước đó tôi đã kể gì với bạn", Intent.RecallMemory),
        new("giải thích khái niệm này trong tài liệu", Intent.SearchKnowledge),
        new("thống kê số đơn hàng trong bảng", Intent.QueryDatabase),
        new("xin chào, bạn khoẻ chứ", Intent.SmallTalk),
        new("tra cứu trên mạng thông tin mới", Intent.SearchInternet),
    ];

    /// <summary>
    /// Những câu trong bộ vàng trùng với tập dữ liệu huấn luyện.
    ///
    /// Phải luôn rỗng. Không rỗng nghĩa là bộ vàng đã bị "dạy trước", và mọi
    /// kết luận rút ra từ nó đều nhẹ hơn nó tỏ ra.
    /// </summary>
    public static IReadOnlyList<string> OverlapWith(DatasetVersion dataset)
    {
        ArgumentNullException.ThrowIfNull(dataset);

        var datasetKeys = dataset.Examples.Select(e => e.Key).ToHashSet(StringComparer.OrdinalIgnoreCase);

        return [.. Examples.Select(e => e.Key).Where(datasetKeys.Contains)];
    }
}
