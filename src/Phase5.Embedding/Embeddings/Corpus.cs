namespace Ai.Phase5.Embeddings;

/// <summary>
/// Kho ngữ liệu tí hon. Mỗi dòng là một nhóm từ cùng xuất hiện.
///
/// Thiết kế có chủ đích:
///   cat và dog  -> cùng đi với pet, animal, fur, tail   => phải GIỐNG nhau
///   apple       -> đi với sweet, fruit, food, eat       => nhóm riêng
///   car         -> đi với fast, vehicle, wheel, engine  => nhóm riêng
///
/// Từ chức năng (the, is, a, and) đã lược bỏ: chúng đi với mọi từ nên chỉ
/// làm nhiễu. Mô hình thật xử lý bằng cách hạ trọng số từ có tần suất cao.
/// </summary>
public static class Corpus
{
    public static readonly string[][] Sentences =
    [
        ["cat", "pet", "animal"],
        ["dog", "pet", "animal"],
        ["cat", "fur", "tail"],
        ["dog", "fur", "tail"],
        ["cat", "animal", "tail"],
        ["dog", "animal", "fur"],

        ["apple", "sweet", "fruit"],
        ["apple", "food", "eat"],
        ["apple", "fruit", "eat"],

        ["car", "fast", "vehicle"],
        ["car", "wheel", "engine"],
        ["car", "vehicle", "engine"],
    ];

    public static Vocabulary BuildVocabulary() =>
        Vocabulary.FromWords(Sentences.SelectMany(s => s));

    /// <summary>
    /// Sinh dữ liệu huấn luyện dưới dạng cặp CÓ HƯỚNG (trung tâm -> ngữ cảnh):
    ///   - CẶP DƯƠNG (label 1): mọi cặp từ có thứ tự trong cùng một dòng.
    ///   - CẶP ÂM   (label 0): từ ngữ cảnh lấy ngẫu nhiên, miễn là cặp đó
    ///                          không phải cặp dương.
    ///
    /// Không có cặp âm thì model học được lời giải tầm thường: đẩy MỌI vector
    /// về cùng một hướng, khi đó mọi tích vô hướng đều lớn và mọi từ đều
    /// "giống" nhau. Cặp âm là lực đẩy cân bằng lại lực hút.
    /// </summary>
    public static WordPair[] BuildPairs(Vocabulary vocabulary, Random rng, int negativesPerPositive = 4)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        ArgumentNullException.ThrowIfNull(rng);

        var positives = new HashSet<(int Center, int Context)>();
        foreach (var sentence in Sentences)
        {
            foreach (string center in sentence)
            {
                foreach (string context in sentence)
                {
                    if (center == context) continue;
                    positives.Add((vocabulary.IndexOf(center), vocabulary.IndexOf(context)));
                }
            }
        }

        var pairs = positives
            .Select(p => new WordPair(p.Center, p.Context, Label: 1))
            .ToList();

        int targetNegatives = positives.Count * negativesPerPositive;
        int attempts = 0;

        while (pairs.Count < positives.Count + targetNegatives && attempts < targetNegatives * 100)
        {
            attempts++;

            int center = rng.Next(vocabulary.Count);
            int context = rng.Next(vocabulary.Count);
            if (center == context) continue;
            if (positives.Contains((center, context))) continue;

            pairs.Add(new WordPair(center, context, Label: 0));
        }

        return [.. pairs];
    }
}
