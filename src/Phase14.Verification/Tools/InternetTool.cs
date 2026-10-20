using System.Text;
using Ai.Phase14.Internet;

namespace Ai.Phase14.Tools;

/// <summary>
/// Công cụ bất đồng bộ.
///
/// Ba công cụ của Phase 11 (calculator, search, database) đều chạy trong bộ
/// nhớ nên trả kết quả tức thì — <see cref="ITool.Execute"/> đồng bộ là đúng
/// với chúng. Internet thì khác hẳn: nó chờ mạng, nên phải bất đồng bộ và
/// phải nhận <see cref="CancellationToken"/>.
///
/// Không sửa <see cref="ITool"/> thành async cho cả bốn: bắt calculator trả
/// về Task chỉ để trông cho đều là biến một phép cộng thành một máy trạng
/// thái, và làm sai lệch điều mà chữ ký hàm đang nói về bản chất công việc.
/// </summary>
public interface IAsyncTool
{
    string Name { get; }

    string Description { get; }

    Task<ToolResult> ExecuteAsync(string input, CancellationToken cancellationToken = default);
}

/// <summary>
/// Công cụ tra Internet — cửa duy nhất để nội dung web đi vào agent.
///
/// Nó trả về TÓM TẮT CÁC NGUỒN, không trả về "câu trả lời". Khác biệt này là
/// cố ý: ở Phase 12 agent chưa có quyền kết luận từ nội dung web. Nó chỉ được
/// nói "đây là những gì tôi tìm thấy, từ những nguồn này". Việc rút ra kết
/// luận là của Phase 12 (evidence) và Phase 14 (kiểm chứng).
///
/// Mọi nội dung đi qua đây đều mang nhãn không tin cậy, và phần đầu output
/// nói rõ điều đó — để không ai (kể cả phase sau) nhầm nó với sự thật đã kiểm.
/// </summary>
public sealed class InternetSearchTool(ISearchService searchService) : IAsyncTool
{
    private readonly ISearchService _search = searchService ?? throw new ArgumentNullException(nameof(searchService));

    /// <summary>
    /// Dịch vụ tra cứu bên dưới.
    ///
    /// Công khai vì PHASE 13 cần nó: <c>EvidenceCollector</c> dùng Internet
    /// như MỘT KÊNH căn cứ ngang hàng với knowledge và memory, nên nó nói
    /// chuyện trực tiếp với <see cref="ISearchService"/> thay vì đi qua lớp
    /// công cụ này (lớp công cụ trả về văn bản đã ghép sẵn, còn collector cần
    /// dữ liệu có cấu trúc).
    /// </summary>
    public ISearchService Service => _search;

    public string Name => "internet";

    public string Description => "tra cứu thông tin trên Internet (nội dung trả về là dữ liệu chưa kiểm chứng)";

    /// <summary>Bao nhiêu nguồn được nêu trong câu trả lời.</summary>
    public int MaxSourcesInAnswer { get; init; } = 3;

    /// <summary>Báo cáo của lượt tra gần nhất — Phase 12 sẽ lấy từ đây để dựng evidence.</summary>
    public InternetSearchReport? LastReport { get; private set; }

    public async Task<ToolResult> ExecuteAsync(string input, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);

        InternetSearchReport report;

        try
        {
            report = await _search.SearchAsync(input, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;   // huỷ là huỷ, không biến thành "tra thất bại"
        }
        catch (OperationCanceledException)
        {
            return ToolResult.Fail("tra cứu Internet hết thời gian");
        }

        LastReport = report;

        if (!report.HasSources)
        {
            string why = report.Warnings.Count > 0
                ? string.Join("; ", report.Warnings)
                : "không tìm thấy nguồn nào";

            return ToolResult.Fail($"không tra được trên Internet ({why})");
        }

        return new ToolResult(true, Summarize(report), Explain(report));
    }

    /// <summary>
    /// Ghép các nguồn thành một đoạn đọc được — KHÔNG diễn giải, không kết luận.
    ///
    /// Đoạn trích của từng nguồn được giữ nguyên (bản đã làm cùn) và luôn đi
    /// kèm URL. Bỏ URL đi là bỏ luôn khả năng truy nguồn, và một khẳng định
    /// không truy được nguồn thì không khác gì bịa.
    /// </summary>
    private string Summarize(InternetSearchReport report)
    {
        var builder = new StringBuilder();

        builder.Append("Tìm thấy ").Append(report.Sources.Count)
            .Append(" nguồn (").Append(report.DistinctHostCount).Append(" tên miền khác nhau). ")
            .AppendLine("Nội dung dưới đây là DỮ LIỆU LẤY TỪ INTERNET, CHƯA KIỂM CHỨNG:");

        foreach (var source in report.Sources.Take(MaxSourcesInAnswer))
        {
            var document = report.Documents.FirstOrDefault(d => d.Url == source.Url);
            var content = document?.Text ?? source.Result.Snippet;

            builder.AppendLine()
                .Append("  • ").Append(source.Result.Title.Preview(80)).AppendLine()
                .Append("    ").Append(content.Preview(280)).AppendLine()
                .Append("    nguồn: ").Append(source.Url);

            if (document is not null)
            {
                builder.Append(" (đã tải lúc ").Append(document.RetrievedAt.ToString("HH:mm:ss")).Append(')');
            }

            builder.AppendLine();
        }

        if (report.Warnings.Count > 0)
        {
            builder.AppendLine().Append("  Lưu ý: ").Append(string.Join("; ", report.Warnings));
        }

        return builder.ToString().TrimEnd();
    }

    /// <summary>Dấu vết để giải trình: đã gửi truy vấn nào, xếp hạng ra sao, vì sao.</summary>
    private static string Explain(InternetSearchReport report)
    {
        var builder = new StringBuilder();

        builder.Append("truy vấn: ")
            .AppendLine(string.Join(" | ", report.Queries.Select(q => $"\"{q.Text}\"")));

        if (report.FromCache) builder.AppendLine("lấy từ cache");
        if (report.DuplicatesRemoved > 0) builder.Append("đã bỏ ").Append(report.DuplicatesRemoved).AppendLine(" kết quả trùng");

        foreach (var source in report.Sources)
        {
            builder.Append("  ").Append(source.Score.ToString("F3")).Append("  ")
                .Append(source.Result.Host).Append("  — ").AppendLine(source.Explanation);
        }

        int injections = report.Documents.Count(d => d.Text.LooksLikeInjection)
            + report.Sources.Count(s => s.Result.Snippet.LooksLikeInjection);

        if (injections > 0)
        {
            builder.Append("CẢNH BÁO: ").Append(injections)
                .AppendLine(" nguồn có nội dung giống chỉ thị — đã vô hiệu hoá và trừ điểm");
        }

        return builder.ToString().TrimEnd();
    }
}
