using Ai.Phase6.Embeddings;

namespace Ai.Phase6.Language;

/// <summary>
/// Tokenizer: cắt chuỗi thành các đơn vị rời rạc (token) rồi quy về chỉ số.
///
///     "I like cats"  ->  ["I", "like", "cats"]  ->  [0, 1, 3]
///
/// Ở đây tách theo khoảng trắng (word-level tokenizer) vì ngữ liệu tí hon.
/// Mô hình thật dùng sub-word (BPE): "programming" có thể thành
/// ["program", "ming"], nhờ đó từ điển nhỏ mà vẫn biểu diễn được từ lạ.
/// Nguyên lý thì y hệt: chuỗi -> danh sách số nguyên.
///
/// Hai token đặc biệt:
///   &lt;s&gt;    đánh dấu đầu câu — cần để có ngữ cảnh cho từ đầu tiên
///   &lt;/s&gt;  đánh dấu cuối câu — để model học được KHI NÀO nên dừng
/// </summary>
public sealed class Tokenizer
{
    public const string StartToken = "<s>";
    public const string EndToken = "</s>";

    public Vocabulary Vocabulary { get; }

    public int StartId { get; }

    public int EndId { get; }

    public Tokenizer(IEnumerable<string> sentences)
    {
        ArgumentNullException.ThrowIfNull(sentences);

        Vocabulary = new Vocabulary();

        // Thêm token đặc biệt trước để chúng luôn có chỉ số 0 và 1.
        StartId = Vocabulary.Add(StartToken);
        EndId = Vocabulary.Add(EndToken);

        foreach (string sentence in sentences)
        {
            foreach (string word in Split(sentence))
            {
                Vocabulary.Add(word);
            }
        }
    }

    public static string[] Split(string sentence)
    {
        ArgumentNullException.ThrowIfNull(sentence);

        return sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    /// <summary>Chuỗi -> danh sách chỉ số token.</summary>
    public int[] Encode(string sentence) =>
        Split(sentence).Select(Vocabulary.IndexOf).ToArray();

    /// <summary>Danh sách chỉ số -> chuỗi.</summary>
    public string Decode(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        return string.Join(' ', tokenIds.Select(Vocabulary.WordAt));
    }

    /// <summary>
    /// Bọc câu bằng token đầu/cuối, và thêm đủ &lt;s&gt; ở đầu để token đầu tiên
    /// cũng có ngữ cảnh đầy đủ:
    ///
    ///     contextSize = 2, "I like cats"
    ///     -> [&lt;s&gt;, &lt;s&gt;, I, like, cats, &lt;/s&gt;]
    /// </summary>
    public int[] EncodeWithBoundaries(string sentence, int contextSize)
    {
        if (contextSize < 1) throw new ArgumentOutOfRangeException(nameof(contextSize));

        var tokens = new List<int>();
        for (int i = 0; i < contextSize; i++) tokens.Add(StartId);
        tokens.AddRange(Encode(sentence));
        tokens.Add(EndId);

        return [.. tokens];
    }
}
