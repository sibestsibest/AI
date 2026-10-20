namespace Ai.Phase8.Language;

/// <summary>Ngữ liệu văn bản tí hon cho mô hình ngôn ngữ.</summary>
public static class TextCorpus
{
    /// <summary>
    /// Đúng 5 câu của đề bài. Với ngữ liệu này, phân bố THẬT của từ tiếp theo
    /// sau "I like" là đều nhau: cats 1/3, dogs 1/3, programming 1/3.
    /// Model học đúng phải cho ra xấp xỉ 0.333 mỗi từ.
    /// </summary>
    public static readonly string[] Basic =
    [
        "I like cats",
        "I like dogs",
        "I like programming",
        "I love cats",
        "I love programming",
    ];

    /// <summary>
    /// Cùng các câu nhưng có TẦN SUẤT khác nhau. Phân bố thật sau "I like":
    ///     cats 3/6 = 0.500 , dogs 2/6 = 0.333 , programming 1/6 = 0.167
    /// Dùng để chứng minh model học được tần suất chứ không chỉ học tập hợp.
    /// </summary>
    public static readonly string[] Weighted =
    [
        "I like cats",
        "I like cats",
        "I like cats",
        "I like dogs",
        "I like dogs",
        "I like programming",
        "I love cats",
        "I love programming",
        "I love programming",
    ];
}
