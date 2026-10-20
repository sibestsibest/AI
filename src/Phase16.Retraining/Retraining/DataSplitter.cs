using System.Text;
using Ai.Phase16.Agent;
using Ai.Phase16.Learning;

namespace Ai.Phase16.Retraining;

/// <summary>Tập train / tập đánh giá, cùng những gì cần biết về cách chia.</summary>
public sealed record DataSplit
{
    public required IReadOnlyList<DatasetExample> Training { get; init; }

    public required IReadOnlyList<DatasetExample> Validation { get; init; }

    /// <summary>
    /// Ví dụ do người dùng sửa lỗi, được đưa thẳng vào tập train thay vì bốc thăm.
    /// Xem <see cref="DataSplitter.Split"/> để biết vì sao.
    /// </summary>
    public required IReadOnlyList<DatasetExample> CorrectionsRoutedToTraining { get; init; }

    /// <summary>
    /// Nhãn có trong tập dữ liệu nhưng KHÔNG có ví dụ nào trong tập train.
    ///
    /// Đây không phải chuyện nhỏ: model không thể học một nhãn nó chưa từng
    /// thấy, nên nó sẽ sai 100% ở nhãn đó — mà accuracy tổng vẫn có thể đẹp
    /// nếu nhãn đó hiếm. Cửa triển khai (<see cref="DeploymentGate"/>) coi đây
    /// là lý do từ chối.
    /// </summary>
    public required IReadOnlyList<Intent> LabelsMissingFromTraining { get; init; }

    public IReadOnlyList<(string Text, Intent Label)> TrainingPairs =>
        [.. Training.Select(e => (e.Input, e.Label))];

    public override string ToString() =>
        $"{Training.Count} train / {Validation.Count} đánh giá";
}

/// <summary>
/// CHIA DỮ LIỆU TẤT ĐỊNH — không xáo trộn ngẫu nhiên.
///
/// Mỗi ví dụ được gán về tập train hay tập đánh giá bằng BĂM CHÍNH DANH TÍNH
/// của nó, chứ không bằng vị trí trong danh sách hay một bộ sinh ngẫu nhiên.
/// Hai tính chất đi kèm, và cả hai đều cần:
///
///   1. CÙNG DỮ LIỆU -> CÙNG CÁCH CHIA, ở mọi lần chạy và mọi máy. Không có nó
///      thì hai lượt train không so được với nhau: chênh lệch accuracy có thể
///      chỉ là do lần này tập đánh giá dễ hơn.
///
///   2. THÊM VÍ DỤ KHÔNG LÀM XÊ DỊCH VÍ DỤ CŨ. Đây mới là tính chất quan
///      trọng. Xáo ngẫu nhiên lại toàn bộ mỗi lần tập dữ liệu lớn lên sẽ khiến
///      một ví dụ hôm qua nằm ở tập đánh giá hôm nay rơi vào tập train — model
///      mới đã HỌC chính câu nó sắp bị kiểm, nên điểm cao lên mà chẳng giỏi
///      thêm. Băm theo danh tính thì ví dụ nào ở đâu là cố định vĩnh viễn.
///
/// VÌ SAO TỰ VIẾT FNV-1a THAY VÌ DÙNG <c>string.GetHashCode()</c>?
///
/// Vì .NET ngẫu nhiên hoá GetHashCode theo từng tiến trình: cùng một chuỗi cho
/// hai giá trị khác nhau ở hai lần chạy. Dùng nó thì cách chia đổi mỗi lần
/// khởi động lại chương trình — hỏng cả hai tính chất trên, mà không có gì báo.
///
/// MỘT NGOẠI LỆ, VÀ NÓ ĐƯỢC PHÁT HIỆN RA BẰNG SỐ ĐO
///
/// Ví dụ đến từ lời sửa của người dùng (<see cref="CandidateProvenance.UserCorrection"/>)
/// KHÔNG bốc thăm — chúng luôn vào tập train.
///
/// Lần đầu chạy không có ngoại lệ này, chuyện xảy ra đúng như sau: người dùng
/// sửa câu "overfitting là gì" (agent đoán <c>Calculate</c>, lẽ ra là
/// <c>SearchKnowledge</c>), Phase 15 duyệt ví dụ đó vào tập dữ liệu, rồi băm
/// xếp nó vào TẬP ĐÁNH GIÁ — bucket 70, dưới ngưỡng 200. Model mới không hề
/// học nó, nên nó vẫn đoán <c>Calculate</c> y như trước. Từ "overfitting" chỉ
/// xuất hiện trong đúng ví dụ ấy, nên không có đường nào khác để học.
///
/// Và vì cách chia là TẤT ĐỊNH, nó sẽ nằm ở tập đánh giá MÃI MÃI: người dùng
/// sửa bao nhiêu lần cũng không đổi được gì.
///
/// Đó là lý do có ngoại lệ. Cái phải trả giá được nói thẳng: ví dụ sửa lỗi
/// KHÔNG còn được đo trên dữ liệu chưa thấy, nên chúng không chứng minh được
/// model đã khái quát hoá. Bù lại bằng hai chốt độc lập: BỘ VÀNG cố định và
/// không bao giờ nằm trong dữ liệu train (<see cref="GoldenSet"/>), và điều
/// kiện KHÔNG LÀM HỎNG thứ model cũ đang làm đúng.
///
/// Lựa chọn còn lại — giữ nguyên bốc thăm cho mọi ví dụ — nghe "trung thực"
/// hơn về mặt đo lường, nhưng nó tạo ra một hệ thống bỏ ngoài tai một phần năm
/// số lời sửa của người dùng. Vòng phản hồi mà không sửa được lỗi thì cả Phase
/// 15 chỉ còn là một cách thu thập dữ liệu rất công phu.
/// </summary>
public static class DataSplitter
{
    /// <summary>Số ngăn để rải ví dụ vào. 1000 ngăn cho độ phân giải tới 0.1%.</summary>
    private const int Buckets = 1000;

    public static DataSplit Split(DatasetVersion dataset, double validationShare)
    {
        ArgumentNullException.ThrowIfNull(dataset);

        if (validationShare is < 0 or > 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(validationShare), validationShare,
                "phần dữ liệu đánh giá phải nằm trong [0, 0.5]");
        }

        int threshold = (int)Math.Round(validationShare * Buckets);

        var training = new List<DatasetExample>();
        var validation = new List<DatasetExample>();
        var corrections = new List<DatasetExample>();

        foreach (var example in dataset.Examples)
        {
            bool isCorrection = example.Provenance == CandidateProvenance.UserCorrection;

            if (!isCorrection && Bucket(example.Key) < threshold)
            {
                validation.Add(example);
                continue;
            }

            training.Add(example);

            if (isCorrection && Bucket(example.Key) < threshold)
            {
                corrections.Add(example);
            }
        }

        var trainedLabels = training.Select(e => e.Label).ToHashSet();
        var missing = dataset.Examples
            .Select(e => e.Label)
            .Distinct()
            .Where(label => !trainedLabels.Contains(label))
            .OrderBy(label => (int)label)
            .ToList();

        return new DataSplit
        {
            Training = training,
            Validation = validation,
            CorrectionsRoutedToTraining = corrections,
            LabelsMissingFromTraining = missing,
        };
    }

    /// <summary>Ngăn của một ví dụ: 0..999, tất định theo danh tính chuẩn hoá của nó.</summary>
    public static int Bucket(string key)
    {
        ArgumentNullException.ThrowIfNull(key);

        return (int)(Fnv1a(key) % Buckets);
    }

    /// <summary>
    /// FNV-1a 64 bit — nhỏ, tất định, và đủ đều cho việc rải ví dụ vào ngăn.
    ///
    /// Không cần chống va chạm kiểu mật mã ở đây: kẻ tấn công có ép được một
    /// câu rơi vào tập đánh giá thì cũng chẳng được lợi gì. Thứ cần là TẤT
    /// ĐỊNH, và đó chính là thứ GetHashCode không cho.
    /// </summary>
    public static ulong Fnv1a(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        const ulong offsetBasis = 14695981039346656037;
        const ulong prime = 1099511628211;

        ulong hash = offsetBasis;

        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            hash ^= b;
            hash *= prime;
        }

        return hash;
    }
}
