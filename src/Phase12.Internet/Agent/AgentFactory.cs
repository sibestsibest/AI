using Ai.Phase12.Internet;
using Ai.Phase12.Memory;
using Ai.Phase12.Reasoning;
using Ai.Phase12.Tools;

namespace Ai.Phase12.Agent;

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
        InternetSearchTool? internet = null) =>
        Build(TrainClassifier(seed, epochs), clock, internet);

    /// <summary>
    /// Dựng agent quanh một classifier ĐÃ train sẵn.
    ///
    /// Huấn luyện tốn vài giây; bộ nhớ, công cụ và kho tài liệu thì dựng
    /// tức thì. Tách ra để tái dùng được classifier khi cần nhiều agent
    /// (ví dụ mỗi unit test một agent sạch).
    /// </summary>
    public static AiController Build(IntentClassifier classifier, IClock? clock = null,
        InternetSearchTool? internet = null)
    {
        ArgumentNullException.ThrowIfNull(classifier);

        clock ??= new SystemClock();
        var memory = new MemoryStore(clock, shortTermCapacity: 6);

        var reasoning = new ReasoningEngine();
        reasoning.LearnAll("A = 10", "B = 20", "C = A + B");

        var search = new SearchTool();
        search.AddDocument("Backpropagation",
            "Backpropagation là thuật toán tính gradient bằng chain rule, đi ngược từ loss về từng tham số. " +
            "Nó tốn 1 forward + 1 backward, không phụ thuộc số tham số.");
        search.AddDocument("Gradient descent",
            "Gradient descent cập nhật tham số theo công thức w = w - learningRate * gradient, " +
            "đi ngược hướng dốc lên của hàm loss.");
        search.AddDocument("Softmax",
            "Softmax biến một vector logits thành phân bố xác suất: e^z chia cho tổng e^z. " +
            "Mọi phần tử dương và tổng bằng 1.");
        search.AddDocument("Attention",
            "Attention tính softmax(Q·Kᵀ/√d)·V. Mỗi token dùng query của mình so với key của " +
            "mọi token để quyết định lấy bao nhiêu thông tin từ đâu.");
        search.AddDocument("Embedding",
            "Embedding gán cho mỗi từ một vector học được, nhờ đó khoảng cách giữa các vector " +
            "trở thành thước đo mức giống nhau về ngữ nghĩa.");
        search.AddDocument("Layer normalization",
            "Layer normalization chuẩn hoá từng token về trung bình 0 và phương sai 1, " +
            "rồi nhân gamma cộng beta học được, giữ cho độ lớn không trôi qua các layer.");
        search.AddDocument("Overfitting",
            "Overfitting là khi model học thuộc dữ liệu huấn luyện thay vì học quy luật, " +
            "nên chạy tốt trên tập train nhưng kém trên dữ liệu mới.");
        search.AddDocument("Loss function",
            "Loss function đo mức sai của model bằng một con số. Học chính là đi tìm bộ tham số " +
            "làm con số đó nhỏ nhất.");
        search.AddDocument("Transformer",
            "Transformer xếp chồng các khối gồm self attention và feed forward network, " +
            "kèm residual và layer normalization.");
        search.AddDocument("Neural network",
            "Neural network là nhiều neuron xếp thành layer nối tiếp nhau, mỗi neuron tính " +
            "tổng có trọng số rồi qua một hàm kích hoạt phi tuyến.");

        var database = new DatabaseTool();
        database.Insert(new Order(1, "Alice", 1_200_000));
        database.Insert(new Order(2, "Bob", 450_000));
        database.Insert(new Order(3, "Chi", 2_800_000));
        database.Insert(new Order(4, "Dung", 760_000));

        return new AiController(classifier, memory, new CalculatorTool(reasoning), search, database, internet);
    }
}
