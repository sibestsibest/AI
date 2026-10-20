using Ai.Phase16.Agent;
using Ai.Phase16.Feedback;
using Ai.Phase16.Learning;
using Ai.Phase16.Memory;
using Ai.Phase16.Retraining;
using Ai.Phase16.Verification;

namespace Ai.Phase16.Demos;

/// <summary>Demo riêng cho Phase 16 — Controlled Retraining.</summary>
public static class RetrainingDemo
{
    /// <summary>
    /// Câu mà model m1 thực sự hiểu sai: nó đoán <see cref="Intent.Calculate"/>,
    /// lẽ ra phải là <see cref="Intent.SearchKnowledge"/>. Sau khi bỏ từ dừng
    /// thì câu chỉ còn một từ khoá — "overfitting" — và từ đó xuất hiện trong
    /// cả tập dữ liệu đúng một lần.
    /// </summary>
    private const string MisreadQuestion = "overfitting là gì";

    public static void Run()
    {
        ShowPipeline();

        var clock = Clock();
        var stack = LearningStackFactory.Build(clock);
        var registry = new ModelRegistry(clock);
        var pipeline = new RetrainingPipeline(registry, clock: clock);

        var first = ShowFirstModel(pipeline, stack.Datasets.Current);
        ShowReproducibility(first, clock);
        var second = ShowFeedbackBecomesBehaviour(pipeline, stack, first, clock);
        ShowGateRejection(pipeline, registry, stack.Datasets.Current, second);
        ShowRollback(registry);
    }

    private static ManualClock Clock() => new(new DateTimeOffset(2026, 1, 1, 9, 0, 0, TimeSpan.Zero));

    private static void ShowPipeline()
    {
        Console.WriteLine("\n\n=== DEMO 26: ĐƯỜNG ỐNG HUẤN LUYỆN LẠI CÓ KIỂM SOÁT ===\n");

        Console.WriteLine("     Tập dữ liệu (Phase 15: có phiên bản + vân tay)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Chia tất định          DataSplitter (băm danh tính, không xáo ngẫu nhiên)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Train                  ModelTrainer (seed + từ điển có thứ tự)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Đánh giá               Evaluator (tập đánh giá + BỘ VÀNG)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     So sánh                ModelComparison (từng ví dụ, không chỉ hiệu số)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Cửa triển khai         DeploymentGate (6 điều kiện, mỗi cái một lý do)");
        Console.WriteLine("        ↓");
        Console.WriteLine("     Sổ đăng ký             ModelRegistry (triển khai / từ chối / quay lui)");

        Console.WriteLine("\n  Không có nhánh nào đặt được model đang chạy mà không đi qua cửa đánh giá.");
    }

    private static RetrainingReport ShowFirstModel(RetrainingPipeline pipeline, DatasetVersion dataset)
    {
        Console.WriteLine("\n\n=== DEMO 27: ĐO TRÊN DỮ LIỆU CHƯA THẤY ===\n");

        var report = pipeline.Run(dataset, new TrainingConfig(), "model đầu tiên");

        Console.WriteLine($"  Tập dữ liệu: {dataset}");
        Console.WriteLine($"  Chia:        {report.Split}");
        Console.WriteLine($"  Model:       {report.Candidate.Label}, {report.Candidate.ParameterCount} tham số\n");

        var onTraining = Evaluator.Evaluate(report.Candidate.Restore(), report.Split.Training);

        Console.WriteLine($"  Accuracy trên TẬP TRAIN     : {onTraining.Accuracy:P1}   <- con số Phase 11 công bố");
        Console.WriteLine($"  Accuracy trên TẬP ĐÁNH GIÁ  : {report.Validation.Accuracy:P1}   <- con số nói đúng về khái quát hoá");
        Console.WriteLine($"  Macro F1                    : {report.Validation.MacroF1:F3}");
        Console.WriteLine($"  Bộ vàng                     : {report.Golden.Correct}/{report.Golden.Total}");

        Console.WriteLine("\n  Cùng một model, cùng một bộ dữ liệu. Khác nhau chỉ ở chỗ ĐO Ở ĐÂU.");
        Console.WriteLine("  \"100% accuracy\" của Phase 11 là accuracy trên chính những câu đã học.");

        Console.WriteLine($"\n  Trong {report.Validation.Total} câu đánh giá, {report.Validation.WithoutTrainedWords.Count} câu " +
                          $"không có MỘT TỪ NÀO từng xuất hiện khi train,");
        Console.WriteLine($"  và {report.Validation.WithoutTrainedWordsWrong} trong số đó bị sai. Bỏ chúng ra thì accuracy là " +
                          $"{report.Validation.AccuracyOnTrainedVocabulary:P1}.");
        Console.WriteLine("  Đó là phần do THIẾU DỮ LIỆU, không phải do model: với 70 ví dụ, giữ lại 20%");
        Console.WriteLine("  để đánh giá là lấy đi ví dụ DUY NHẤT chứa những từ ấy.");

        Console.WriteLine("\n  Từng ý định:");
        foreach (var label in report.Validation.PerLabel)
        {
            Console.WriteLine($"     {label}");
        }

        return report;
    }

    private static void ShowReproducibility(RetrainingReport first, ManualClock clock)
    {
        Console.WriteLine("\n\n=== DEMO 28: TÁI LẬP ===\n");

        var again = new RetrainingPipeline(new ModelRegistry(clock), clock: clock)
            .Run(LearningStackFactory.Build(clock).Datasets.Current, new TrainingConfig());

        var other = new RetrainingPipeline(new ModelRegistry(clock), clock: clock)
            .Run(LearningStackFactory.Build(clock).Datasets.Current, new TrainingConfig { Seed = 1234 });

        Console.WriteLine($"  Lần 1, seed 42   : {first.Candidate.Fingerprint}");
        Console.WriteLine($"  Lần 2, seed 42   : {again.Candidate.Fingerprint}   <- giống hệt, đến từng chữ số");
        Console.WriteLine($"  Lần 3, seed 1234 : {other.Candidate.Fingerprint}   <- khác");

        Console.WriteLine("\n  Ba nguồn bất định đã bị bịt để có được điều này:");
        Console.WriteLine("    1. trọng số khởi tạo -> seed nằm TRONG cấu hình và được lưu cùng model");
        Console.WriteLine("    2. thứ tự từ điển    -> sắp Ordinal, không duyệt HashSet");
        Console.WriteLine("    3. thứ tự ví dụ      -> tập dữ liệu Phase 15 đã sắp về thứ tự chuẩn");
        Console.WriteLine("\n  Nguồn thứ ba nghe như vặt vãnh, nhưng cộng số thực dấu phẩy động KHÔNG có");
        Console.WriteLine("  tính kết hợp: lệch 1e-16 mỗi bước, qua 3000 epoch là hai model khác nhau.");
    }

    private static RetrainingReport ShowFeedbackBecomesBehaviour(RetrainingPipeline pipeline,
        LearningStack stack, RetrainingReport first, IClock clock)
    {
        Console.WriteLine("\n\n=== DEMO 29: PHẢN HỒI TRỞ THÀNH HÀNH VI ===\n");

        var m1 = first.Candidate.Restore();
        var before = m1.Classify(MisreadQuestion);

        Console.WriteLine($"  \"{MisreadQuestion}\"");
        Console.WriteLine($"     m1 đoán: {before.Intent} ({before.Confidence:P1})  <- SAI, lẽ ra là SearchKnowledge\n");

        // Phase 15: người dùng sửa -> ứng viên -> kiểm -> phiên bản dữ liệu mới.
        var record = stack.Journal.Record(
            new GroundedAnswer
            {
                Question = MisreadQuestion,
                Text = "…",
                Verdict = AnswerVerdict.Answered,
                Confidence = 0.8,
                AnsweredAt = clock.Now,
                Summary = new ReasoningSummary(),
            },
            before.Intent,
            before.Confidence);

        stack.Feedback.Submit(record.Id, FeedbackRating.Incorrect, expectedIntent: Intent.SearchKnowledge);

        var learned = stack.Pipeline.Run(m1.Vocabulary, "sửa một ca hiểu sai ý định");

        Console.WriteLine($"  Phase 15: {learned}");

        var report = pipeline.Run(learned.NewVersion!, new TrainingConfig(), "sau phản hồi người dùng");
        var after = report.Candidate.Restore().Classify(MisreadQuestion);

        Console.WriteLine($"  Phase 16: {report}\n");
        Console.WriteLine($"     m2 đoán: {after.Intent} ({after.Confidence:P1})  <- ĐÚNG\n");

        Console.WriteLine($"  So với model đang chạy: {report.Comparison}");
        Console.WriteLine($"  Ví dụ sửa lỗi được ĐƯA THẲNG vào tập train: {report.Split.CorrectionsRoutedToTraining.Count}");

        Console.WriteLine("\n  Chi tiết đó là bản sửa cho một lỗi thật. Băm xếp câu này vào tập ĐÁNH GIÁ");
        Console.WriteLine("  (bucket 70, dưới ngưỡng 200), nên model mới không hề học nó và vẫn đoán sai y");
        Console.WriteLine("  như cũ. Mà cách chia là TẤT ĐỊNH, nên nó sẽ nằm đó mãi: người dùng sửa bao");
        Console.WriteLine("  nhiêu lần cũng không đổi được gì.");
        Console.WriteLine("\n  Giá phải trả được nói thẳng: ví dụ sửa lỗi không còn được đo trên dữ liệu");
        Console.WriteLine("  chưa thấy. Bù lại bằng hai chốt độc lập — bộ vàng, và điều kiện không làm");
        Console.WriteLine("  hỏng thứ model cũ đang làm đúng.");

        return report;
    }

    private static void ShowGateRejection(RetrainingPipeline pipeline, ModelRegistry registry,
        DatasetVersion dataset, RetrainingReport second)
    {
        Console.WriteLine("\n\n=== DEMO 30: MODEL TỆ HƠN KHÔNG THAY ĐƯỢC MODEL ĐANG CHẠY ===\n");

        var bad = pipeline.Run(dataset, new TrainingConfig { Epochs = 1 }, "mới train 1 epoch");

        Console.WriteLine($"  Ứng viên {bad.Candidate.Label}: {bad.Validation}");
        Console.WriteLine($"  Bộ vàng: {bad.Golden.Correct}/{bad.Golden.Total}\n");

        foreach (var check in bad.Decision.Checks)
        {
            Console.WriteLine($"     {check}");
        }

        Console.WriteLine($"\n  Kết luận: {(bad.Decision.Approved ? "TRIỂN KHAI" : "TỪ CHỐI")}");
        Console.WriteLine($"  Model đang chạy sau lượt này: {registry.Active!.Label} " +
                          $"(vẫn là {second.Candidate.Label}, không đổi)");

        Console.WriteLine($"\n  {bad.Candidate.Label} KHÔNG bị xoá — nó vẫn trong sổ kèm trọng số và lý do bị");
        Console.WriteLine("  từ chối. Model bị từ chối là dữ liệu quý: nó cho biết cấu hình nào đã thử và");
        Console.WriteLine("  hỏng ra sao. Xoá đi thì lần sau có người thử lại đúng cấu hình đó.");
    }

    private static void ShowRollback(ModelRegistry registry)
    {
        Console.WriteLine("\n\n=== DEMO 31: QUAY LUI ===\n");

        var active = registry.Active!;
        var beforeRollback = registry.RestoreActive().Classify(MisreadQuestion);

        Console.WriteLine($"  Đang chạy {active.Label}: \"{MisreadQuestion}\" -> {beforeRollback.Intent}");

        var previous = registry.Rollback("giả định m2 gây lỗi trên thực tế");

        var afterRollback = registry.RestoreActive().Classify(MisreadQuestion);

        Console.WriteLine($"  Quay lui -> {previous!.Label}: \"{MisreadQuestion}\" -> {afterRollback.Intent}");

        Console.WriteLine("\n  Hành vi trở lại THẬT, không chỉ đổi một con số phiên bản — vì phiên bản model");
        Console.WriteLine("  lưu cả trọng số. Quay lui bằng cách train lại từ dữ liệu cũ thì phải tin rằng");
        Console.WriteLine("  máy đó, hôm đó, còn cho ra đúng dãy số hôm nay.");

        Console.WriteLine("\n  Lịch sử triển khai:");
        foreach (var entry in registry.History)
        {
            Console.WriteLine($"     {entry}");
        }

        Console.WriteLine("\n  Quay lui chỉ về được model ĐÃ TỪNG CHẠY. Nhảy vào một model chưa bao giờ vượt");
        Console.WriteLine("  cửa đánh giá không phải quay lui — đó là triển khai một model chưa được duyệt,");
        Console.WriteLine("  bằng một cái tên nghe an toàn hơn.");
    }
}
