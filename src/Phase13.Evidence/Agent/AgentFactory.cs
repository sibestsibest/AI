using Ai.Phase13.Grounding;
using Ai.Phase13.Internet;
using Ai.Phase13.Memory;
using Ai.Phase13.Reasoning;
using Ai.Phase13.Tools;

namespace Ai.Phase13.Agent;

/// <summary>Dựng sẵn một agent đã train xong, đã nạp dữ liệu — dùng cho demo và test.</summary>
public static class AgentFactory
{
    public static IntentClassifier TrainClassifier(int seed = 42, int epochs = 3000)
    {
        var classifier = new IntentClassifier(IntentTrainingData.Texts, new Random(seed));
        classifier.Train(IntentTrainingData.Examples, epochs);
        return classifier;
    }

    public static AiController Build(int seed = 42, IClock? clock = null, int epochs = 3000,
        InternetSearchTool? internet = null, bool withEvidence = true) =>
        Build(TrainClassifier(seed, epochs), clock, internet, withEvidence);

    /// <summary>
    /// Dựng agent quanh một classifier ĐÃ train sẵn.
    ///
    /// Huấn luyện tốn vài giây; bộ nhớ, công cụ và kho tài liệu thì dựng
    /// tức thì. Tách ra để tái dùng được classifier khi cần nhiều agent
    /// (ví dụ mỗi unit test một agent sạch).
    ///
    /// <paramref name="withEvidence"/> tắt được cả tầng Phase 13 để kiểm chứng
    /// rằng hành vi của Phase 11–12 vẫn còn nguyên khi không có nó.
    /// </summary>
    public static AiController Build(IntentClassifier classifier, IClock? clock = null,
        InternetSearchTool? internet = null, bool withEvidence = true)
    {
        ArgumentNullException.ThrowIfNull(classifier);

        clock ??= new SystemClock();
        var memory = new MemoryStore(clock, shortTermCapacity: 6);

        var reasoning = new ReasoningEngine();
        reasoning.LearnAll("A = 10", "B = 20", "C = A + B");

        // Kho tài liệu và kho kiến thức nạp từ CÙNG MỘT danh sách
        // (<see cref="KnowledgeSeed"/>), nên chúng không thể lệch nhau.
        var search = new SearchTool();
        foreach (var (topic, content) in KnowledgeSeed.Documents)
        {
            search.AddDocument(topic, content);
        }

        var database = new DatabaseTool();
        database.Insert(new Order(1, "Alice", 1_200_000));
        database.Insert(new Order(2, "Bob", 450_000));
        database.Insert(new Order(3, "Chi", 2_800_000));
        database.Insert(new Order(4, "Dung", 760_000));

        if (!withEvidence)
        {
            return new AiController(classifier, memory, new CalculatorTool(reasoning), search, database, internet);
        }

        // --- PHASE 13 ---
        var knowledge = KnowledgeSeed.BuildStore(clock);
        var context = new ConversationContext(clock);

        var collector = new EvidenceCollector(
            knowledge,
            memory,
            context,
            internet?.Service,
            clock: clock);

        return new AiController(classifier, memory, new CalculatorTool(reasoning), search, database,
            internet, knowledge, context, collector);
    }
}
