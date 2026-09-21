using System.Globalization;
using Ai.Phase16.Agent;
using Ai.Phase16.Feedback;
using Ai.Phase16.Verification;
using Ai.Providers.Routing;

namespace Ai.Chat;

/// <summary>
/// MÀN HÌNH CHAT — vòng lặp đọc–trả lời, và toàn bộ phần in ấn.
///
/// Không giữ trạng thái nào của riêng nó ngoài hai công tắc hiển thị. Mọi thứ
/// khác nằm trong <see cref="ChatSession"/>.
///
/// MẶC ĐỊNH IN GỌN, và đó là lựa chọn có chủ ý. Agent này có rất nhiều thứ
/// đáng in ra — ý định, độ tự tin, từng mẩu căn cứ, các bước quyết định, kết
/// quả kiểm tra cuối. In hết mọi lượt thì câu trả lời bị chôn giữa siêu dữ
/// liệu, và người dùng thôi không đọc nữa. Nên phần đó nằm sau lệnh
/// <c>/chitiet</c>: bật khi cần soi, tắt khi chỉ muốn hỏi.
/// </summary>
public sealed class ChatConsole(ChatSession session)
{
    private const int WrapWidth = 88;

    private bool _showDetail;

    public async Task RunAsync()
    {
        using var input = OpenInput();

        PrintBanner();

        while (true)
        {
            Write("\nBạn> ", ConsoleColor.Cyan);

            string? line = await input.ReadLineAsync().ConfigureAwait(false);

            // Ctrl+C / Ctrl+Z, hoặc đầu vào bị chuyển hướng và đã hết — thoát
            // thay vì lặp vô hạn trên null.
            if (line is null) break;

            line = line.Trim();
            if (line.Length == 0) continue;

            if (line.StartsWith('/'))
            {
                if (await HandleCommandAsync(line).ConfigureAwait(false)) break;
                continue;
            }

            await AskAsync(line).ConfigureAwait(false);
        }

        WriteLine("\nTạm biệt.", ConsoleColor.DarkGray);
    }

    /// <summary>
    /// Mở luồng đầu vào ĐỌC ĐÚNG TIẾNG VIỆT — và đây là bản sửa cho một lỗi thật.
    ///
    /// Trên Windows, <see cref="Console.ReadLine"/> giải mã đầu vào theo codepage
    /// của console (mặc định là một bảng mã 8-bit), nên câu hỏi tiếng Việt gõ
    /// vào hoặc đưa qua pipe bị băm thành ký tự rác. Hậu quả không hiện ra như
    /// một lỗi mã hoá: bộ tách từ không nhận ra từ nào, bộ phân loại trả về độ
    /// tự tin 0%, và agent lịch sự nói "tôi không chắc bạn muốn gì" cho MỌI câu
    /// hỏi — trông y như model hỏng.
    ///
    /// Hai việc, vì hai đường vào khác nhau:
    ///   • đặt <see cref="Console.InputEncoding"/> = UTF-8 cho người GÕ TRỰC TIẾP
    ///   • đọc bằng <see cref="StreamReader"/> UTF-8 khi đầu vào bị CHUYỂN HƯỚNG
    ///     (chạy test bằng pipe) — lúc đó codepage console không còn tác dụng
    ///
    /// Bọc try/catch vì đặt InputEncoding ném lỗi khi tiến trình không gắn với
    /// console nào; khi đó nhánh StreamReader mới là nhánh có tác dụng.
    /// </summary>
    private static TextReader OpenInput()
    {
        try
        {
            Console.InputEncoding = System.Text.Encoding.UTF8;
        }
        catch (Exception ex) when (ex is IOException or PlatformNotSupportedException)
        {
            // Không có console để đặt codepage — đầu vào chuyển hướng, bỏ qua.
        }

        return Console.IsInputRedirected
            ? new StreamReader(Console.OpenStandardInput(), new System.Text.UTF8Encoding(false))
            : Console.In;
    }

    // ------------------------------------------------------------------
    // Hỏi – đáp
    // ------------------------------------------------------------------

    private async Task AskAsync(string question)
    {
        var started = DateTimeOffset.UtcNow;

        ChatTurn turn;

        try
        {
            turn = await session.AskAsync(question).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // Một câu hỏi làm hỏng một lượt thì không được làm sập cả phiên:
            // người dùng mất luôn bộ nhớ hội thoại vì một lỗi họ không gây ra.
            WriteLine($"\nLỗi khi xử lý câu hỏi: {ex.Message}", ConsoleColor.Red);
            return;
        }

        var elapsed = DateTimeOffset.UtcNow - started;

        Console.WriteLine();
        Write("AI  ", ConsoleColor.Green);

        var grounded = turn.Response.Grounded;

        if (grounded is not null)
        {
            WriteLine(Badge(grounded), BadgeColor(grounded.Verdict));
        }
        else
        {
            WriteLine($"[{turn.Response.Intent} · {turn.Response.Confidence:P0}" +
                      (turn.Response.ToolUsed is null ? "" : $" · {turn.Response.ToolUsed}") + "]",
                ConsoleColor.DarkGray);
        }

        foreach (string line in Wrap(turn.Response.Answer))
        {
            Console.WriteLine($"        {line}");
        }

        if (turn.Response.UsedUnverifiedInternetData)
        {
            WriteLine("        (có dùng nội dung web CHƯA KIỂM CHỨNG)", ConsoleColor.Yellow);
        }

        if (turn.UsedFallback)
        {
            WriteLine($"        (bộ phân loại ý định không nhận ra câu này — " +
                      $"{turn.RoutedAs!.Confidence:P0} tự tin, đã đi thẳng vào đường ống kiểm chứng)",
                ConsoleColor.DarkGray);
        }

        if (_showDetail) PrintDetail(turn, elapsed);
    }

    private static string Badge(GroundedAnswer answer) =>
        string.Format(CultureInfo.InvariantCulture, "[{0} · tự tin {1:P0} · dải {2}]",
            answer.Verdict, answer.Confidence, answer.Band);

    private static ConsoleColor BadgeColor(AnswerVerdict verdict) => verdict switch
    {
        AnswerVerdict.Answered => ConsoleColor.DarkGreen,
        AnswerVerdict.Uncertain => ConsoleColor.Yellow,
        AnswerVerdict.Conflicted => ConsoleColor.Magenta,
        _ => ConsoleColor.DarkGray,
    };

    // ------------------------------------------------------------------
    // Chi tiết
    // ------------------------------------------------------------------

    private void PrintDetail(ChatTurn turn, TimeSpan elapsed)
    {
        var response = turn.Response;

        WriteLine($"\n        ── chi tiết ─────────────────────────────────────────", ConsoleColor.DarkGray);
        Detail("id", turn.Record.Id);
        Detail("ý định", $"{response.Intent} ({response.Confidence:P1})" +
                         (response.Confidence < IntentPrediction.MinimumConfidence ? "  ← dưới ngưỡng tự tin" : ""));
        Detail("từ nhận ra", $"{session.Classifier.KnownWordCount(turn.Question)} từ có trong từ điển " +
                             $"({session.Classifier.Vocabulary.Count} từ)");

        if (turn.UsedFallback)
        {
            Detail("đường đi", "đường lui — bỏ qua định tuyến, vào thẳng đường ống kiểm chứng");
        }

        if (response.ToolUsed is not null) Detail("công cụ", response.ToolUsed);

        if (response.Grounded is { } grounded)
        {
            Detail("kết luận", $"{grounded.Verdict}, tự tin {grounded.Confidence:F3}, dải {grounded.Band}");
            Detail("khẳng định", $"{grounded.Facts.Count} sự thật, {grounded.Inferences.Count} suy ra" +
                                 (grounded.AllClaimsSupported ? ", tất cả đều có căn cứ đỡ" : ", CÓ khẳng định không được đỡ"));

            if (grounded.CitedEvidence.Count > 0)
            {
                Detail("căn cứ", "");

                foreach (var evidence in grounded.CitedEvidence)
                {
                    Console.WriteLine($"                 [{evidence.Kind}/{evidence.Trust}] {evidence.Source.Cite()}");
                    Console.WriteLine($"                   khớp {evidence.Relevance:F2} · tin cậy {evidence.Reliability:F2}");
                }
            }

            if (grounded.Contradictions.Count > 0)
            {
                Detail("mâu thuẫn", $"{grounded.Contradictions.Count}, nặng nhất {grounded.Contradictions[0].Severity:F2}");
            }

            if (grounded.Validation is { } validation)
            {
                Detail("kiểm tra cuối", validation.ToString());
            }

            foreach (string warning in grounded.Warnings)
            {
                WriteLine($"        cảnh báo     : {warning}", ConsoleColor.Yellow);
            }
        }

        // Chẩn đoán khâu diễn đạt (ticket "UI / debug"): provider nào, model
        // nào, bao lâu, và nếu văn bản LLM bị loại thì vì sao. Chỉ hiện khi
        // người dùng đã bật LLM — không bật thì dòng này chỉ là tiếng ồn.
        if (session.LlmEnabled && session.LastCompose is { } compose)
        {
            Detail("diễn đạt", compose.ToString());

            if (compose.UsedFallbackProvider)
            {
                Detail("", $"đã chuyển provider: {string.Join(" → ", compose.Attempts.Select(a => a.ToString()))}");
            }
        }

        if (response.Decisions.Count > 0)
        {
            Detail("các bước", "");

            foreach (var step in response.Decisions)
            {
                Console.WriteLine($"                 {step.Question} -> {step.Decision}");
            }
        }

        Detail("thời gian", $"{elapsed.TotalMilliseconds:F0} ms");
    }

    private static void Detail(string label, string value) =>
        Console.WriteLine($"        {label,-13}: {value}");

    // ------------------------------------------------------------------
    // Lệnh
    // ------------------------------------------------------------------

    /// <summary>Trả về true nếu người dùng muốn thoát.</summary>
    private async Task<bool> HandleCommandAsync(string input)
    {
        string[] parts = input.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        string command = parts[0].ToLowerInvariant();
        string[] args = parts[1..];

        switch (command)
        {
            case "/thoat" or "/quit" or "/exit":
                return true;

            case "/giupdo" or "/help" or "/?":
                PrintHelp();
                return false;

            case "/chitiet":
                _showDetail = !_showDetail;
                WriteLine(_showDetail ? "Đã BẬT chi tiết." : "Đã TẮT chi tiết.", ConsoleColor.DarkGray);

                // Bật lên thì in luôn chi tiết của lượt vừa rồi — người dùng gõ
                // lệnh này chính vì họ đang thắc mắc về câu trả lời đó.
                if (_showDetail && session.LastTurn is { } last) PrintDetail(last, TimeSpan.Zero);
                return false;

            case "/vidu":
                PrintExamples();
                return false;

            case "/trangthai":
                PrintStatus();
                return false;

            case "/gopy":
                Rate(args);
                return false;

            case "/hoc":
                PrintLearning();
                return false;

            case "/trainlai":
                await RetrainAsync().ConfigureAwait(false);
                return false;

            case "/loi":
                PrintErrorCases();
                return false;

            case "/hieuchuan":
                PrintCalibration();
                return false;

            case "/llm":
                await ToggleLlmAsync(args).ConfigureAwait(false);
                return false;

            case "/provider":
                PrintProviders();
                return false;

            case "/che" or "/mode":
                SetMode(args);
                return false;

            case "/hoi" or "/ask":
                await AskModelAsync(args).ConfigureAwait(false);
                return false;

            case "/moi":
                session.ResetConversation();
                WriteLine("Đã bắt đầu hội thoại mới (model và tập dữ liệu giữ nguyên).", ConsoleColor.DarkGray);
                return false;

            default:
                WriteLine($"Không có lệnh '{command}'. Gõ /giupdo để xem danh sách.", ConsoleColor.Red);
                return false;
        }
    }

    private void Rate(string[] args)
    {
        if (args.Length == 0)
        {
            WriteLine("Cách dùng: /gopy dung   |   /gopy sai <ÝĐịnh>   |   /gopy thieu   |   /gopy vodung",
                ConsoleColor.Red);
            WriteLine($"Ý định: {string.Join(", ", Enum.GetNames<Intent>())}", ConsoleColor.DarkGray);
            return;
        }

        var (rating, expected) = ParseRating(args);

        if (rating is null)
        {
            WriteLine($"Không hiểu '{args[0]}'. Dùng: dung | sai | thieu | vodung", ConsoleColor.Red);
            return;
        }

        if (rating == FeedbackRating.Incorrect && args.Length > 1 && expected is null)
        {
            WriteLine($"Không có ý định '{args[1]}'. Chọn một trong: {string.Join(", ", Enum.GetNames<Intent>())}",
                ConsoleColor.Red);
            return;
        }

        var result = session.Rate(rating.Value, expected);

        if (!result.Accepted)
        {
            WriteLine($"Không nhận phản hồi: {result.Reason}", ConsoleColor.Red);
            return;
        }

        WriteLine($"Đã ghi phản hồi {result.Feedback!.Id} ({rating})" +
                  (expected is null ? "" : $", lẽ ra là {expected}"), ConsoleColor.DarkGray);

        if (rating != FeedbackRating.Correct && expected is null)
        {
            WriteLine("  Mẹo: phản hồi có kèm ý định đúng (/gopy sai SearchInternet) mới sinh ra được", ConsoleColor.DarkGray);
            WriteLine("  dữ liệu huấn luyện — vì nhãn phải do NGƯỜI gán, không phải do model tự đoán.", ConsoleColor.DarkGray);
        }
    }

    private static (FeedbackRating? Rating, Intent? Expected) ParseRating(string[] args)
    {
        Intent? expected = args.Length > 1 && Enum.TryParse<Intent>(args[1], ignoreCase: true, out var parsed)
            ? parsed
            : null;

        FeedbackRating? rating = args[0].ToLowerInvariant() switch
        {
            "dung" or "đúng" or "ok" or "y" => FeedbackRating.Correct,
            "sai" => FeedbackRating.Incorrect,
            "thieu" or "thiếu" => FeedbackRating.Incomplete,
            "vodung" or "vôdụng" or "vo-dung" => FeedbackRating.Unhelpful,
            _ => null,
        };

        return (rating, expected);
    }

    private void PrintLearning()
    {
        var report = session.Learn();

        WriteLine($"\nPhase 15 — {report}", ConsoleColor.DarkGray);

        foreach (var errorCase in report.ErrorCases)
        {
            Console.WriteLine($"  ca lỗi {errorCase}  (train được: {(errorCase.IsTrainable ? "CÓ" : "KHÔNG")})");
        }

        foreach (string skipped in report.Skipped)
        {
            Console.WriteLine($"  bỏ qua {skipped}");
        }

        foreach (var candidate in report.Rejected)
        {
            Console.WriteLine($"  LOẠI  {candidate.Input} -> {candidate.Label}: {candidate.Decision}");
        }

        foreach (var example in report.NewVersion?.Added ?? [])
        {
            WriteLine($"  THÊM  {example}  (nguồn nhãn {example.Provenance})", ConsoleColor.Green);
        }

        foreach (string warning in report.Warnings)
        {
            WriteLine($"  cảnh báo: {warning}", ConsoleColor.Yellow);
        }

        if (report.ProducedNewVersion)
        {
            WriteLine($"  Tập dữ liệu giờ là {report.NewVersion}", ConsoleColor.DarkGray);
            WriteLine("  Trọng số CHƯA đổi — gõ /trainlai để huấn luyện lại.", ConsoleColor.DarkGray);
        }
    }

    private async Task RetrainAsync()
    {
        WriteLine("\nĐang huấn luyện lại (chia dữ liệu -> train -> đánh giá -> so sánh -> cửa)…",
            ConsoleColor.DarkGray);

        var report = await session.RetrainAsync().ConfigureAwait(false);

        Console.WriteLine($"  ứng viên {report.Candidate.Label}: {report.Validation}");
        Console.WriteLine($"  bộ vàng: {report.Golden.Correct}/{report.Golden.Total}");
        Console.WriteLine($"  chia: {report.Split}");

        if (report.Comparison is { } comparison)
        {
            Console.WriteLine($"  so với model đang chạy: {comparison}");
        }

        Console.WriteLine();

        foreach (var check in report.Decision.Checks)
        {
            WriteLine($"     {check}", check.Passed ? ConsoleColor.DarkGreen : ConsoleColor.Red);
        }

        if (report.Deployed)
        {
            WriteLine($"\n  ĐÃ TRIỂN KHAI {report.Candidate.Label} — agent đang dùng model mới.", ConsoleColor.Green);
            WriteLine("  Bộ nhớ hội thoại bắt đầu lại (agent được dựng lại quanh model mới).", ConsoleColor.DarkGray);
        }
        else
        {
            WriteLine($"\n  TỪ CHỐI {report.Candidate.Label} — agent giữ nguyên model cũ.", ConsoleColor.Red);
            WriteLine("  Model bị từ chối vẫn nằm trong sổ kèm trọng số và lý do.", ConsoleColor.DarkGray);
        }
    }

    /// <summary>
    /// Bật/tắt LLM ngoài cho khâu DIỄN ĐẠT.
    ///
    /// Nói rõ ngay tại đây LLM được phép làm gì: nó viết lại căn cứ cho gọn.
    /// Nó không chọn căn cứ, không quyết định có trả lời hay không, không gắn
    /// nguồn. Người dùng cần biết ranh giới đó, vì nó là lý do câu trả lời vẫn
    /// truy được nguồn sau khi bật.
    /// </summary>
    private async Task ToggleLlmAsync(string[] args)
    {
        string mode = args.Length > 0 ? args[0].ToLowerInvariant() : (session.LlmEnabled ? "off" : "on");

        if (mode is "off" or "tat" or "tắt")
        {
            session.DisableLlm();
            WriteLine("Đã TẮT LLM ngoài — quay về bộ sinh theo khuôn của Phase 14.", ConsoleColor.DarkGray);
            return;
        }

        WriteLine("\nĐang kiểm tra provider…", ConsoleColor.DarkGray);

        string? problem = await session.EnableLlmAsync().ConfigureAwait(false);

        if (problem is not null)
        {
            WriteLine($"Không bật được: {problem}", ConsoleColor.Red);
            WriteLine("Gõ /provider để xem cấu hình đang đọc được gì.", ConsoleColor.DarkGray);
            return;
        }

        WriteLine("Đã BẬT LLM ngoài cho khâu DIỄN ĐẠT.", ConsoleColor.Green);
        WriteLine("  LLM chỉ viết lại các căn cứ cho gọn. Nó KHÔNG chọn căn cứ, KHÔNG quyết định", ConsoleColor.DarkGray);
        WriteLine("  có trả lời hay không, KHÔNG gắn nguồn. Văn bản nó viết phải qua được", ConsoleColor.DarkGray);
        WriteLine("  AnswerValidator của Phase 14, nếu không thì câu trả lời khuôn mẫu cũ được dùng.", ConsoleColor.DarkGray);
    }

    /// <summary>Đổi chế độ cho đường hỏi thẳng model (<c>/hoi</c>).</summary>
    private void SetMode(string[] args)
    {
        if (args.Length == 0)
        {
            WriteLine($"\nChế độ hiện tại: {session.Mode}", ConsoleColor.DarkGray);
            Console.WriteLine("  auto      — tự phân loại theo luật (mặc định)");
            Console.WriteLine("  fast      — model nhỏ, KHÔNG công cụ, ít token");
            Console.WriteLine("  reasoning — model mạnh, nhiều token hơn");
            Console.WriteLine("  research  — bật công cụ tra web");
            Console.WriteLine("  coding    — cấu hình thiên về mã");
            Console.WriteLine("\nCách dùng: /che <tên>");
            return;
        }

        var mode = Modes.Parse(args[0]);
        session.Mode = mode;

        var policy = Modes.Policy(mode);

        WriteLine($"Chế độ: {mode}", ConsoleColor.Green);
        WriteLine($"  công cụ {(policy.AllowTools ? "BẬT" : "tắt")}, " +
                  $"tra web {(policy.AllowWebSearch ? "BẬT" : "tắt")}, " +
                  $"tối đa {policy.MaxTokens} token ra", ConsoleColor.DarkGray);
    }

    /// <summary>
    /// HỎI THẲNG MODEL qua đường điều phối có công cụ — KHÁC đường mặc định.
    ///
    /// Đường mặc định (gõ thẳng câu hỏi) vẫn là đường CÓ KIỂM CHỨNG của Phase
    /// 13–14: thu thập căn cứ, dò mâu thuẫn, truy nguồn từng câu. Lệnh này đi
    /// đường khác: để model gọi công cụ của MiniAI rồi tự trả lời.
    ///
    /// Tách thành một LỆNH RIÊNG chứ không đổi hành vi mặc định, vì hai đường
    /// cho hai loại bảo đảm khác nhau — và người dùng có quyền biết mình đang
    /// nhận bảo đảm nào.
    /// </summary>
    private async Task AskModelAsync(string[] args)
    {
        if (args.Length == 0)
        {
            WriteLine("Cách dùng: /hoi <câu hỏi>", ConsoleColor.Red);
            WriteLine("  Đường này để MODEL gọi công cụ và tự trả lời — khác đường kiểm chứng mặc định.",
                ConsoleColor.DarkGray);
            return;
        }

        if (session.Ai.Providers.Count == 0)
        {
            WriteLine("Chưa cấu hình provider nào — gõ /provider để xem.", ConsoleColor.Red);
            return;
        }

        string question = string.Join(' ', args);

        WriteLine($"\nĐang hỏi model (chế độ {session.Mode})…", ConsoleColor.DarkGray);

        var result = await session.AskModelAsync(question).ConfigureAwait(false);

        if (!result.Success)
        {
            WriteLine($"Không trả lời được: {result.Error?.Message}", ConsoleColor.Red);
            return;
        }

        Console.WriteLine();
        WriteLine(result.Message, ConsoleColor.White);

        var d = result.Diagnostics;

        WriteLine($"\n  [{result.Provider}/{result.Model}] {d.Latency.TotalMilliseconds:F0} ms, {result.Usage}",
            ConsoleColor.DarkGray);

        if (result.ToolCalls.Count > 0)
        {
            WriteLine("  công cụ: " + string.Join(", ", result.ToolCalls.Select(t =>
                $"{t.Name}{(t.Refused ? " (TỪ CHỐI)" : t.Succeeded ? "" : " (hỏng)")}")), ConsoleColor.DarkGray);
        }

        if (d.RetryCount > 0)
        {
            WriteLine($"  đã viết lại {d.RetryCount} lần: {string.Join("; ", d.ValidationProblems)}",
                ConsoleColor.DarkYellow);
        }

        if (d.HitToolLimit)
        {
            WriteLine("  CHẠM TRẦN số lượt công cụ — câu trả lời có thể chưa đầy đủ.", ConsoleColor.DarkYellow);
        }
    }

    private void PrintProviders()
    {
        WriteLine($"\nLớp provider — nguồn cấu hình: {session.Ai.ConfigSource}", ConsoleColor.DarkGray);
        Console.WriteLine($"  chiến lược      : {session.Ai.Options.Strategy}");
        Console.WriteLine($"  LLM diễn đạt    : {(session.LlmEnabled ? "ĐANG BẬT" : "đang tắt")}");

        if (session.Ai.Options.Providers.Count == 0)
        {
            WriteLine("  (chưa cấu hình provider nào)", ConsoleColor.DarkGray);
        }

        foreach (string line in session.Ai.Describe())
        {
            Console.WriteLine($"  {line}");
        }

        foreach (string problem in session.Ai.Problems)
        {
            WriteLine($"  VẤN ĐỀ: {problem}", ConsoleColor.Yellow);
        }

        var health = session.Ai.Health.All;

        if (health.Count > 0)
        {
            Console.WriteLine("  sức khoẻ:");

            foreach (var entry in health)
            {
                Console.WriteLine($"     {entry}");
            }
        }

        var usage = session.Ai.Usage.Summarize();

        if (usage.Count > 0)
        {
            Console.WriteLine("  đã dùng:");

            foreach (var summary in usage)
            {
                Console.WriteLine($"     {summary}");
            }
        }

        var (used, rejected) = session.LlmComposeCounts;

        if (used + rejected > 0)
        {
            Console.WriteLine($"  diễn đạt: dùng văn bản LLM {used} lần, bị loại {rejected} lần");
        }

        if (session.LastCompose is { } last)
        {
            Console.WriteLine($"  lần gần nhất: {last}");
        }
    }

    private void PrintErrorCases()
    {
        var cases = session.TopErrorCases();

        if (cases.Count == 0)
        {
            int pending = session.Status().FeedbackCount;

            WriteLine(pending > 0
                    ? $"\nCó {pending} phản hồi chưa được xử lý — gõ /hoc để biến chúng thành ca lỗi."
                    : "\nChưa có ca lỗi nào. Chấm sai một câu trả lời (/gopy sai …) rồi gõ /hoc.",
                ConsoleColor.DarkGray);
            return;
        }

        WriteLine("\nCa lỗi, nặng nhất trước:", ConsoleColor.DarkGray);

        foreach (var errorCase in cases)
        {
            Console.WriteLine($"  {errorCase}");
            Console.WriteLine($"     đoán {errorCase.PredictedIntent}, lẽ ra {errorCase.ExpectedIntent?.ToString() ?? "(người dùng không nói)"}" +
                              $" · train được: {(errorCase.IsTrainable ? "CÓ" : "KHÔNG")}");
        }
    }

    private void PrintCalibration()
    {
        var report = session.Calibration();

        WriteLine($"\nHiệu chuẩn — {report}", ConsoleColor.DarkGray);

        if (report.TotalRated == 0)
        {
            WriteLine("  Chưa chấm lượt nào đi qua đường ống kiểm chứng.", ConsoleColor.DarkGray);
            WriteLine("  (lượt trả lời bằng công cụ không tự nhận độ tự tin nên không hiệu chuẩn được)",
                ConsoleColor.DarkGray);
            return;
        }

        foreach (var bucket in report.Buckets)
        {
            WriteLine($"  {bucket}", bucket.IsOverconfident ? ConsoleColor.Yellow : ConsoleColor.Gray);
        }
    }

    private void PrintStatus()
    {
        var status = session.Status();

        WriteLine("\nTrạng thái phiên:", ConsoleColor.DarkGray);
        Console.WriteLine($"  model đang chạy : {status.ModelLabel}, {status.ParameterCount} tham số");
        Console.WriteLine($"  tập dữ liệu     : v{status.DatasetVersion} (vân tay {status.DatasetFingerprint})");
        Console.WriteLine($"  đã hỏi          : {status.TurnCount} lượt");
        Console.WriteLine($"  ký ức dài hạn   : {status.MemoryCount}");
        Console.WriteLine($"  kho kiến thức   : {status.KnowledgeCount} tài liệu");
        Console.WriteLine($"  phản hồi / lỗi  : {status.FeedbackCount} / {status.ErrorCaseCount}");
        Console.WriteLine($"  Internet        : ngoại tuyến (StaticSearchProvider — không có lời gọi mạng nào)");

        if (session.Registry.History.Count > 0)
        {
            Console.WriteLine("  lịch sử model   :");

            foreach (var entry in session.Registry.History)
            {
                Console.WriteLine($"     {entry}");
            }
        }
    }

    // ------------------------------------------------------------------
    // In ấn tĩnh
    // ------------------------------------------------------------------

    private static void PrintBanner()
    {
        WriteLine("\n================================================================", ConsoleColor.DarkGray);
        WriteLine("  MINI AI — màn hình chat", ConsoleColor.Green);
        WriteLine("  Lõi học máy là C# thuần, không ML.NET — và MẶC ĐỊNH chạy ngoại tuyến.", ConsoleColor.DarkGray);
        WriteLine("  LLM ngoài (Ollama / OpenAI) là tuỳ chọn, phải bật tường minh:", ConsoleColor.DarkGray);
        WriteLine("  /llm để diễn đạt câu trả lời, /hoi để model gọi công cụ.", ConsoleColor.DarkGray);
        WriteLine("  Gõ /giupdo để xem lệnh, /thoat để ra.", ConsoleColor.DarkGray);
        WriteLine("================================================================", ConsoleColor.DarkGray);
    }

    private static void PrintHelp()
    {
        WriteLine("\nLệnh:", ConsoleColor.DarkGray);
        Console.WriteLine("  /chitiet            bật/tắt phần chi tiết (ý định, căn cứ, các bước quyết định)");
        Console.WriteLine("  /vidu               vài câu hỏi mẫu cho từng ý định");
        Console.WriteLine("  /trangthai          model, tập dữ liệu, ký ức, lịch sử triển khai");
        Console.WriteLine();
        Console.WriteLine("  /gopy dung          chấm câu trả lời vừa rồi là ĐÚNG");
        Console.WriteLine("  /gopy sai <ÝĐịnh>   chấm là SAI, kèm ý định lẽ ra phải hiểu");
        Console.WriteLine("  /gopy thieu         đúng nhưng thiếu");
        Console.WriteLine("  /gopy vodung        đúng mà không dùng được");
        Console.WriteLine("  /loi                ca lỗi đã thu, nặng nhất trước");
        Console.WriteLine("  /hieuchuan          độ tự tin agent tự nhận có khớp thực tế không");
        Console.WriteLine();
        Console.WriteLine("  /hoc                Phase 15: phản hồi -> tập dữ liệu mới (KHÔNG đổi trọng số)");
        Console.WriteLine("  /trainlai           Phase 16: train lại -> đánh giá -> chỉ triển khai nếu đạt");
        Console.WriteLine();
        Console.WriteLine("  /llm on | off       dùng LLM ngoài cho khâu DIỄN ĐẠT (mặc định tắt)");
        Console.WriteLine("  /provider           cấu hình provider, sức khoẻ, token đã dùng, lần diễn đạt gần nhất");
        Console.WriteLine();
        Console.WriteLine("  /hoi <câu hỏi>      hỏi THẲNG model, cho nó gọi công cụ của MiniAI");
        Console.WriteLine("                      (khác đường mặc định: đường mặc định có KIỂM CHỨNG căn cứ)");
        Console.WriteLine("  /che <tên>          chế độ cho /hoi: auto | fast | reasoning | research | coding");
        Console.WriteLine();
        Console.WriteLine("  /moi                bắt đầu hội thoại mới (giữ model và dữ liệu)");
        Console.WriteLine("  /thoat              ra khỏi màn hình");

        WriteLine("\nVòng đầy đủ để thử: hỏi một câu -> /gopy sai <ÝĐịnh> -> /hoc -> /trainlai -> hỏi lại.",
            ConsoleColor.DarkGray);
    }

    private static void PrintExamples()
    {
        WriteLine("\nCâu hỏi mẫu:", ConsoleColor.DarkGray);
        Console.WriteLine("  Calculate       : 2 + 3 * 4        |  tính tổng của A và B");
        Console.WriteLine("  RememberFact    : hãy nhớ tôi dùng C#");
        Console.WriteLine("  RecallMemory    : tôi dùng ngôn ngữ nào");
        Console.WriteLine("  SearchKnowledge : SSRF là gì       |  softmax hoạt động thế nào");
        Console.WriteLine("  QueryDatabase   : có bao nhiêu đơn hàng");
        Console.WriteLine("  SearchInternet  : tìm trên mạng thông tin về SSRF");
        Console.WriteLine("  SmallTalk       : chào bạn");
        WriteLine("  Thử cả câu agent KHÔNG biết: cách nấu phở bò Hà Nội", ConsoleColor.DarkGray);
    }

    // ------------------------------------------------------------------
    // Tiện ích
    // ------------------------------------------------------------------

    /// <summary>
    /// Ngắt dòng theo từ, giữ nguyên các dòng có sẵn.
    ///
    /// Giữ dòng có sẵn là điều bắt buộc: câu trả lời của Phase 14 là văn bản có
    /// gạch đầu dòng và dòng trích dẫn nguồn — ngắt lại từ đầu sẽ trộn hết
    /// chúng thành một đoạn văn không đọc được.
    /// </summary>
    private static IEnumerable<string> Wrap(string text)
    {
        foreach (string paragraph in text.Split('\n'))
        {
            string line = paragraph.TrimEnd();

            if (line.Length <= WrapWidth)
            {
                yield return line;
                continue;
            }

            var current = new System.Text.StringBuilder();

            foreach (string word in line.Split(' '))
            {
                if (current.Length > 0 && current.Length + 1 + word.Length > WrapWidth)
                {
                    yield return current.ToString();
                    current.Clear();
                }

                if (current.Length > 0) current.Append(' ');
                current.Append(word);
            }

            if (current.Length > 0) yield return current.ToString();
        }
    }

    private static void Write(string text, ConsoleColor color)
    {
        var previous = Console.ForegroundColor;
        Console.ForegroundColor = color;
        Console.Write(text);
        Console.ForegroundColor = previous;
    }

    private static void WriteLine(string text, ConsoleColor color)
    {
        var previous = Console.ForegroundColor;
        Console.ForegroundColor = color;
        Console.WriteLine(text);
        Console.ForegroundColor = previous;
    }
}
