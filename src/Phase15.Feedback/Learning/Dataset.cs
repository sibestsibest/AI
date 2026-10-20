using System.Security.Cryptography;
using System.Text;
using Ai.Phase15.Agent;
using Ai.Phase15.Memory;

namespace Ai.Phase15.Learning;

/// <summary>
/// MỘT VÍ DỤ trong tập dữ liệu đã duyệt.
///
/// Giữ luôn <see cref="Provenance"/> và <see cref="FeedbackId"/> ngay trong
/// từng ví dụ, chứ không chỉ ghi ở mức phiên bản tập dữ liệu. Vì sao? Vì câu
/// hỏi thực tế hay phải trả lời là "ví dụ NÀY từ đâu ra" — chẳng hạn khi một
/// model mới train ra tệ đi và cần biết ví dụ nào vừa được thêm, ai thêm, dựa
/// trên phản hồi nào. Truy nguồn ở mức tập thì không trả lời được câu đó.
/// </summary>
public sealed record DatasetExample
{
    public required string Input { get; init; }

    public required Intent Label { get; init; }

    public required CandidateProvenance Provenance { get; init; }

    public string? FeedbackId { get; init; }

    public required DateTimeOffset AddedAt { get; init; }

    /// <summary>
    /// DANH TÍNH CHUẨN HOÁ của ví dụ — dùng để so trùng và để tính vân tay.
    ///
    /// Chuẩn hoá bằng <see cref="Keywords.Canonical"/> nên "tính 2 cộng 3" và
    /// "Tính 2 Cộng 3!!" là MỘT ví dụ. Nếu so bằng chuỗi thô thì chỉ cần thêm
    /// một dấu chấm là lọt qua cửa chống trùng — và tập dữ liệu dần đầy những
    /// bản sao gần giống nhau, khiến một câu duy nhất được lặp lại hàng chục
    /// lần và kéo lệch cả model.
    ///
    /// HỆ QUẢ ĐÁNG BIẾT: "tính 2 cộng 3" và "tính 2 cộng 4" cũng là MỘT ví dụ,
    /// vì <see cref="Keywords.Extract"/> bỏ token dài một ký tự nên cả hai cùng
    /// còn lại {tính, cộng}.
    ///
    /// Nghe như mất thông tin, nhưng đó đúng là điều cần: khoá chuẩn hoá dùng
    /// CHÍNH bộ tách từ mà <c>IntentClassifier.Encode</c> dùng, nên hai câu có
    /// cùng khoá sẽ encode thành cùng một vector bag-of-words. Model KHÔNG phân
    /// biệt được chúng — nên với tập dữ liệu, chúng là một ví dụ lặp lại, và
    /// giữ cả hai chỉ làm ví dụ đó nặng gấp đôi mà không dạy thêm điều gì.
    ///
    /// Đổi kiến trúc (thêm n-gram, dùng embedding của Phase 5) thì khoá này
    /// phải đổi theo. Ràng buộc đó là thật, và nó nằm ở đây chứ không rải rác.
    /// </summary>
    public string Key => $"{Label}|{Keywords.Canonical(Input)}";

    public override string ToString() => $"\"{Input}\" -> {Label}";
}

/// <summary>
/// MỘT PHIÊN BẢN TẬP DỮ LIỆU — bất biến, có cha, có vân tay.
///
/// Tập dữ liệu ở đây là APPEND-ONLY theo phiên bản: mỗi lần duyệt phản hồi
/// sinh ra một phiên bản MỚI, phiên bản cũ không bị sửa. Đổi lấy tốn bộ nhớ,
/// nhưng nhận lại hai thứ mà Phase 16 bắt buộc phải có:
///
///   • so sánh được hai model bằng cách chỉ ra chúng train từ tập nào
///   • quay lui được: model cũ vẫn dựng lại được từ đúng dữ liệu của nó
///
/// VÂN TAY — và một chỗ dễ làm sai:
///
/// Vân tay phải xác định được TRỌNG SỐ, không chỉ xác định NỘI DUNG. Hai tập
/// có cùng bộ ví dụ nhưng khác THỨ TỰ sẽ train ra hai model khác nhau (gradient
/// được cộng dồn theo thứ tự, và số thực dấu phẩy động không có tính kết hợp).
/// Nếu vân tay chỉ băm nội dung mà bỏ qua thứ tự, ta sẽ có hai model khác nhau
/// cùng khai một vân tay — và toàn bộ lời hứa "tái lập được" thành vô nghĩa.
///
/// Cách xử lý: <see cref="Create"/> SẮP XẾP ví dụ về một thứ tự chuẩn trước
/// khi băm. Nhờ vậy cùng nội dung thì luôn cùng thứ tự, nên cùng vân tay và
/// cùng trọng số — bất kể ví dụ được thêm vào theo trình tự nào.
/// </summary>
public sealed record DatasetVersion
{
    public required int Version { get; init; }

    public required IReadOnlyList<DatasetExample> Examples { get; init; }

    public int? ParentVersion { get; init; }

    public required DateTimeOffset CreatedAt { get; init; }

    /// <summary>Băm nội dung + thứ tự. 12 ký tự hex là đủ để nhận diện bằng mắt.</summary>
    public required string Fingerprint { get; init; }

    /// <summary>Ghi chú của người tạo phiên bản này.</summary>
    public string Note { get; init; } = "";

    /// <summary>Các ví dụ được thêm so với phiên bản cha.</summary>
    public IReadOnlyList<DatasetExample> Added { get; init; } = [];

    /// <summary>Ứng viên bị loại khi tạo phiên bản này, kèm lý do — phần audit.</summary>
    public IReadOnlyList<string> RejectedReasons { get; init; } = [];

    public string Label => $"v{Version}";

    public int Count => Examples.Count;

    /// <summary>Số ví dụ theo từng nhãn — chỗ hay lộ ra tập dữ liệu bị lệch.</summary>
    public IReadOnlyDictionary<Intent, int> LabelCounts =>
        Examples.GroupBy(e => e.Label).ToDictionary(g => g.Key, g => g.Count());

    public IReadOnlyList<(string Text, Intent Label)> AsTrainingPairs() =>
        [.. Examples.Select(e => (e.Input, e.Label))];

    public override string ToString() => $"{Label} ({Count} ví dụ, vân tay {Fingerprint})";

    /// <summary>
    /// Dựng một phiên bản: sắp xếp về thứ tự chuẩn rồi tính vân tay.
    ///
    /// Sắp theo (nhãn, danh tính chuẩn hoá) bằng so sánh Ordinal — KHÔNG dùng
    /// so sánh theo văn hoá. So sánh theo văn hoá cho thứ tự khác nhau giữa các
    /// máy có locale khác nhau, và khi đó cùng một tập dữ liệu sẽ train ra hai
    /// model khác nhau ở hai máy khác nhau. Đó là loại lỗi tái lập mà không ai
    /// tìm ra được.
    /// </summary>
    public static DatasetVersion Create(int version, IEnumerable<DatasetExample> examples,
        DateTimeOffset createdAt, int? parentVersion = null, string note = "",
        IEnumerable<DatasetExample>? added = null, IEnumerable<string>? rejectedReasons = null)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var ordered = examples
            .OrderBy(e => (int)e.Label)
            .ThenBy(e => e.Key, StringComparer.Ordinal)
            .ToList();

        return new DatasetVersion
        {
            Version = version,
            Examples = ordered,
            ParentVersion = parentVersion,
            CreatedAt = createdAt,
            Fingerprint = ComputeFingerprint(ordered),
            Note = note,
            Added = [.. added ?? []],
            RejectedReasons = [.. rejectedReasons ?? []],
        };
    }

    /// <summary>
    /// Vân tay = 12 ký tự đầu của SHA-256 trên danh sách danh tính đã sắp thứ tự.
    ///
    /// Vì sao SHA-256 chứ không phải <c>string.GetHashCode()</c>? Vì
    /// <c>GetHashCode</c> của .NET được ngẫu nhiên hoá theo từng tiến trình:
    /// cùng một chuỗi cho hai giá trị khác nhau ở hai lần chạy. Dùng nó thì
    /// vân tay đổi mỗi lần khởi động lại chương trình, và không so sánh được
    /// hai lần train với nhau — đúng thứ mà vân tay tồn tại để làm.
    /// </summary>
    public static string ComputeFingerprint(IReadOnlyList<DatasetExample> orderedExamples)
    {
        ArgumentNullException.ThrowIfNull(orderedExamples);

        var builder = new StringBuilder();
        foreach (var example in orderedExamples)
        {
            builder.Append(example.Key).Append('\n');
        }

        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
        return Convert.ToHexString(hash)[..12].ToLowerInvariant();
    }
}
