using Ai.Phase16.Agent;
using Ai.Phase16.Embeddings;
using Ai.Phase16.Internet;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Learning;

/// <summary>Kết luận về MỘT ứng viên.</summary>
public sealed record CandidateVerdict(TrainingCandidate Candidate, bool Approved, string Reason)
{
    public override string ToString() =>
        $"{(Approved ? "DUYỆT" : "LOẠI")} [{Candidate.Id}] \"{Candidate.Input}\" -> {Candidate.Label}: {Reason}";
}

/// <summary>Kết quả kiểm cả lô.</summary>
public sealed record ValidationOutcome
{
    public IReadOnlyList<CandidateVerdict> Verdicts { get; init; } = [];

    /// <summary>Cảnh báo — không chặn, nhưng phải đọc trước khi train.</summary>
    public IReadOnlyList<string> Warnings { get; init; } = [];

    public IReadOnlyList<TrainingCandidate> Approved =>
        [.. Verdicts.Where(v => v.Approved).Select(v => v.Candidate with
        {
            Status = CandidateStatus.Approved,
            Decision = v.Reason,
        })];

    public IReadOnlyList<TrainingCandidate> Rejected =>
        [.. Verdicts.Where(v => !v.Approved).Select(v => v.Candidate with
        {
            Status = CandidateStatus.Rejected,
            Decision = v.Reason,
        })];

    public IReadOnlyList<string> RejectionReasons =>
        [.. Verdicts.Where(v => !v.Approved).Select(v => $"{v.Candidate.Id}: {v.Reason}")];

    public override string ToString() =>
        $"duyệt {Approved.Count}/{Verdicts.Count}, {Warnings.Count} cảnh báo";
}

/// <summary>
/// KIỂM CHẤT LƯỢNG DỮ LIỆU — cửa duy nhất vào tập dữ liệu đã duyệt.
///
/// SÁU CỬA, theo thứ tự, và thứ tự này có ý:
///
///   1. Nguồn nhãn không được phép        -> LOẠI (luật cứng, không ngoại lệ)
///   2. Nhãn không tồn tại                -> LOẠI
///   3. Văn bản rỗng / không có từ khoá   -> LOẠI
///   4. Có dấu hiệu cố ra lệnh            -> LOẠI
///   5. Trùng ví dụ đã có                 -> LOẠI (không phải lỗi, chỉ là không có gì mới)
///   6. XUNG ĐỘT NHÃN với ví dụ đã có     -> LOẠI (phải có người quyết)
///
/// Luật cứng đứng đầu vì nó là luật DUY NHẤT không có ngoại lệ nào: một ứng
/// viên mà nhãn do model tự đoán hoặc do web cung cấp thì không cần kiểm gì
/// thêm. Kiểm trước những cửa rẻ hơn rồi mới xét nguồn cũng cho kết quả giống
/// nhau, nhưng đọc code lại thấy luật cứng nằm lẫn giữa các quy tắc chất lượng
/// — và luật cứng không cùng loại với quy tắc chất lượng.
///
/// CỬA 6 LÀ CỬA ĐÁNG NÓI NHẤT.
///
/// Xung đột nhãn là khi cùng một câu đã có trong tập với nhãn KHÁC. Nó không
/// phải lỗi gõ nhầm: đó là hai người (hoặc cùng một người ở hai thời điểm)
/// không đồng ý với nhau về việc câu đó nghĩa là gì. Nhận cả hai vào thì tập
/// dữ liệu tự mâu thuẫn, và model học được đúng một điều: câu đó là 50/50.
/// Loss không bao giờ xuống, mà cũng không có gì báo là dữ liệu sai — chỉ thấy
/// accuracy trần thấp một cách bí ẩn.
///
/// Nó cũng là đường tấn công thẳng nhất: gửi liên tiếp nhiều phản hồi gán nhãn
/// ngược nhau cho những câu phổ biến là đủ làm hỏng tập dữ liệu, mà không câu
/// nào trong đó "trông" độc hại. Nên cửa này TỪ CHỐI và giao lại cho người
/// quyết, thay vì tự chọn nhãn mới hay nhãn cũ.
///
/// HẠN MỨC MỖI LẦN DUYỆT (<see cref="MaxAddedShare"/>) là lớp phòng vệ thứ hai
/// cho cùng nguy cơ đó: một lô duyệt không được phép làm phình tập dữ liệu quá
/// một tỉ lệ nhất định. Phản hồi thật đến rải rác theo thời gian; một trăm ví
/// dụ mới trong một lô là dấu hiệu của tự động hoá, không phải của người dùng.
/// Phần vượt hạn mức KHÔNG bị xoá — nó được để lại cho lần duyệt sau.
/// </summary>
public sealed class CandidateValidator
{
    /// <summary>Độ dài tối đa của một câu đầu vào. Dài hơn thì gần như chắc chắn không phải câu hỏi.</summary>
    public int MaxInputLength { get; init; } = 200;

    /// <summary>
    /// Một lô duyệt được phép làm tập dữ liệu phình thêm tối đa bao nhiêu phần.
    /// 0.25 = tối đa một phần tư kích thước hiện tại.
    /// </summary>
    public double MaxAddedShare { get; init; } = 0.25;

    /// <summary>
    /// Kiểm một lô ứng viên trên nền tập dữ liệu hiện tại.
    ///
    /// Kiểm TUẦN TỰ và cộng dồn: ví dụ vừa được duyệt trong lô này lập tức trở
    /// thành nền để xét những ví dụ sau nó. Nếu kiểm song song, độc lập từng
    /// ứng viên với tập cũ, thì hai ứng viên xung đột nhãn với NHAU sẽ cùng
    /// được duyệt — vì không ứng viên nào xung đột với tập cũ. Tập dữ liệu vẫn
    /// tự mâu thuẫn, chỉ khác là bây giờ cửa 6 báo "không có xung đột".
    /// </summary>
    public ValidationOutcome Validate(IEnumerable<TrainingCandidate> candidates,
        DatasetVersion current, Vocabulary? vocabulary = null)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(current);

        var verdicts = new List<CandidateVerdict>();
        var warnings = new List<string>();

        // Danh tính đã có -> để bắt trùng; và bản đồ câu -> nhãn để bắt xung đột.
        var existingKeys = current.Examples.Select(e => e.Key).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var labelByInput = new Dictionary<string, Intent>(StringComparer.OrdinalIgnoreCase);

        foreach (var example in current.Examples)
        {
            labelByInput[Keywords.Canonical(example.Input)] = example.Label;
        }

        int quota = Quota(current.Count);
        int approved = 0;

        foreach (var candidate in candidates)
        {
            var verdict = Check(candidate, existingKeys, labelByInput, vocabulary, warnings);

            // Hạn mức xét SAU cùng: chỉ những ứng viên vốn hợp lệ mới tính vào
            // hạn mức. Ứng viên bị loại vì chất lượng không "chiếm chỗ" của lô sau.
            if (verdict.Approved && approved >= quota)
            {
                verdict = new CandidateVerdict(candidate, false,
                    $"vượt hạn mức {quota} ví dụ mới mỗi lần duyệt (tập hiện có {current.Count}) — " +
                    "để lại cho lần duyệt sau");
            }

            if (verdict.Approved)
            {
                approved++;

                string canonical = Keywords.Canonical(candidate.Input);
                existingKeys.Add($"{candidate.Label}|{canonical}");
                labelByInput[canonical] = candidate.Label;
            }

            verdicts.Add(verdict);
        }

        return new ValidationOutcome { Verdicts = verdicts, Warnings = warnings };
    }

    /// <summary>
    /// Hạn mức: tỉ lệ trên kích thước hiện tại, nhưng không bao giờ nhỏ hơn 1.
    ///
    /// Sàn bằng 1 là cần thiết cho tập rỗng hoặc rất nhỏ: 25% của 0 là 0, và
    /// khi đó không ví dụ nào vào được — tập dữ liệu không bao giờ khởi động
    /// nổi.
    /// </summary>
    private int Quota(int currentCount) => Math.Max(1, (int)Math.Floor(currentCount * MaxAddedShare));

    private CandidateVerdict Check(TrainingCandidate candidate, HashSet<string> existingKeys,
        Dictionary<string, Intent> labelByInput, Vocabulary? vocabulary, List<string> warnings)
    {
        // --- Cửa 1: LUẬT CỨNG ---
        if (!candidate.MayBecomeTrainingData)
        {
            return new CandidateVerdict(candidate, false,
                $"nguồn nhãn {candidate.Provenance} không được thành dữ liệu huấn luyện");
        }

        // --- Cửa 2 ---
        if (!Enum.IsDefined(candidate.Label))
        {
            return new CandidateVerdict(candidate, false, $"nhãn '{candidate.Label}' không tồn tại");
        }

        // --- Cửa 3 ---
        if (string.IsNullOrWhiteSpace(candidate.Input))
        {
            return new CandidateVerdict(candidate, false, "câu đầu vào rỗng");
        }

        if (candidate.Input.Length > MaxInputLength)
        {
            return new CandidateVerdict(candidate, false,
                $"câu đầu vào dài {candidate.Input.Length} ký tự, vượt mức {MaxInputLength}");
        }

        var words = Keywords.Extract(candidate.Input);

        if (words.Count == 0)
        {
            return new CandidateVerdict(candidate, false,
                "câu không còn từ khoá nào sau khi bỏ từ dừng — bag-of-words sẽ ra vector rỗng");
        }

        // --- Cửa 4 ---
        UntrustedText.Sanitize(candidate.Input, out int neutralized);

        if (neutralized > 0)
        {
            return new CandidateVerdict(candidate, false,
                $"câu có {neutralized} mẫu cố ra lệnh cho agent — không nhận vào dữ liệu huấn luyện");
        }

        // --- Cửa 5 ---
        if (existingKeys.Contains(candidate.Key))
        {
            return new CandidateVerdict(candidate, false, "đã có đúng ví dụ này trong tập (cùng câu, cùng nhãn)");
        }

        // --- Cửa 6 ---
        string canonical = Keywords.Canonical(candidate.Input);

        if (labelByInput.TryGetValue(canonical, out var existingLabel) && existingLabel != candidate.Label)
        {
            return new CandidateVerdict(candidate, false,
                $"xung đột nhãn: tập đang gán câu này là {existingLabel}, ứng viên nói là {candidate.Label} — " +
                "phải có người quyết, hệ thống không tự chọn");
        }

        // --- Nhận, kèm cảnh báo nếu có ---
        //
        // Từ mới KHÔNG phải lý do loại: Phase 16 dựng lại từ điển từ chính tập
        // dữ liệu, nên từ mới sẽ học được. Nhưng nó làm SỐ THAM SỐ của model
        // đổi, tức là model mới không còn cùng kiến trúc với model cũ — người
        // duyệt cần biết điều đó trước khi so sánh hai model với nhau.
        if (vocabulary is not null)
        {
            var newWords = words.Where(w => !vocabulary.Contains(w)).ToList();

            if (newWords.Count > 0)
            {
                warnings.Add($"{candidate.Id}: thêm {newWords.Count} từ mới vào từ điển " +
                             $"({string.Join(", ", newWords.Take(5))}) — số tham số của model sẽ đổi");
            }
        }

        return new CandidateVerdict(candidate, true, $"qua cả 6 cửa kiểm, nguồn nhãn {candidate.Provenance}");
    }
}
