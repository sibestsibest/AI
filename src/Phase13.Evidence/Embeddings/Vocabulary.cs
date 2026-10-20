namespace Ai.Phase13.Embeddings;

/// <summary>
/// Từ điển: ánh xạ hai chiều giữa TỪ (chuỗi) và CHỈ SỐ (số nguyên).
///
/// Máy tính không tính toán được trên chuỗi. Bước đầu tiên của mọi mô hình
/// ngôn ngữ là quy mỗi từ về một số nguyên duy nhất:
///
///     "cat" -> 0 , "dog" -> 1 , "car" -> 2 , ...
///
/// Lưu ý: những con số này KHÔNG mang ý nghĩa. "cat"=0 và "dog"=1 gần nhau
/// chỉ là ngẫu nhiên do thứ tự thêm vào. Chính vì chỉ số vô nghĩa nên ta cần
/// embedding — xem <see cref="EmbeddingTable"/>.
/// </summary>
public sealed class Vocabulary
{
    private readonly Dictionary<string, int> _wordToIndex = new(StringComparer.Ordinal);
    private readonly List<string> _indexToWord = [];

    public int Count => _indexToWord.Count;

    public IReadOnlyList<string> Words => _indexToWord;

    /// <summary>Thêm từ nếu chưa có, trả về chỉ số của nó.</summary>
    public int Add(string word)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(word);

        if (_wordToIndex.TryGetValue(word, out int existing))
        {
            return existing;
        }

        int index = _indexToWord.Count;
        _wordToIndex[word] = index;
        _indexToWord.Add(word);
        return index;
    }

    public int IndexOf(string word)
    {
        ArgumentNullException.ThrowIfNull(word);

        return _wordToIndex.TryGetValue(word, out int index)
            ? index
            : throw new KeyNotFoundException($"Từ '{word}' không có trong từ điển.");
    }

    public bool Contains(string word) => _wordToIndex.ContainsKey(word);

    public string WordAt(int index) => _indexToWord[index];

    public static Vocabulary FromWords(IEnumerable<string> words)
    {
        ArgumentNullException.ThrowIfNull(words);

        var vocabulary = new Vocabulary();
        foreach (string word in words)
        {
            vocabulary.Add(word);
        }

        return vocabulary;
    }
}
