using Ai.Phase5.Demos;
using Ai.Phase5.Embeddings;
using Ai.Phase5.LinAlg;
using Ai.Phase5.Training;

namespace Ai.Phase5.Tests;

public class VocabularyTests
{
    [Fact]
    public void Add_KhongTaoBanTrung()
    {
        var vocabulary = new Vocabulary();

        int first = vocabulary.Add("cat");
        int again = vocabulary.Add("cat");
        int second = vocabulary.Add("dog");

        Assert.Equal(0, first);
        Assert.Equal(0, again);
        Assert.Equal(1, second);
        Assert.Equal(2, vocabulary.Count);
    }

    [Fact]
    public void AnhXaHaiChieu_KhopNhau()
    {
        var vocabulary = Vocabulary.FromWords(["cat", "dog", "car", "apple"]);

        for (int i = 0; i < vocabulary.Count; i++)
        {
            Assert.Equal(i, vocabulary.IndexOf(vocabulary.WordAt(i)));
        }
    }

    [Fact]
    public void TuKhongCoTrongTuDien_ThiNemLoi()
    {
        var vocabulary = Vocabulary.FromWords(["cat"]);

        Assert.False(vocabulary.Contains("dragon"));
        Assert.Throws<KeyNotFoundException>(() => vocabulary.IndexOf("dragon"));
    }
}

public class SimilarityTests
{
    [Theory]
    [InlineData(1, 0, 1, 0, 1.0)]        // cùng hướng
    [InlineData(1, 0, 0, 1, 0.0)]        // vuông góc
    [InlineData(1, 0, -1, 0, -1.0)]      // ngược hướng
    [InlineData(3, 4, 6, 8, 1.0)]        // cùng hướng, khác độ dài: 50/(5·10)
    [InlineData(1, 0, 1, 1, 0.7071067811865475)]   // 45°
    public void Cosine_TrungVoiKetQuaTinhTay(double a0, double a1, double b0, double b1, double expected)
    {
        Assert.Equal(expected, Similarity.Cosine(Vec.Of(a0, a1), Vec.Of(b0, b1)), precision: 10);
    }

    [Fact]
    public void Cosine_DoiXung()
    {
        var a = Vec.Of(1.3, -0.7, 2.1);
        var b = Vec.Of(-0.4, 1.9, 0.2);

        Assert.Equal(Similarity.Cosine(a, b), Similarity.Cosine(b, a), precision: 12);
    }

    [Fact]
    public void Cosine_VoiChinhNo_Bang1()
    {
        var a = Vec.Of(1.3, -0.7, 2.1);

        Assert.Equal(1.0, Similarity.Cosine(a, a), precision: 10);
    }

    [Fact]
    public void Cosine_LuonTrongKhoangTruMotDenMot()
    {
        var rng = new Random(1);

        for (int trial = 0; trial < 200; trial++)
        {
            var a = new Vec(5);
            var b = new Vec(5);
            for (int i = 0; i < 5; i++)
            {
                a[i] = rng.NextDouble() * 10 - 5;
                b[i] = rng.NextDouble() * 10 - 5;
            }

            Assert.InRange(Similarity.Cosine(a, b), -1.0000001, 1.0000001);
        }
    }

    [Fact]
    public void Cosine_VectorKhong_TraVeKhong()
    {
        // Vector 0 không có hướng -> không so sánh được, quy ước 0.
        Assert.Equal(0.0, Similarity.Cosine(new Vec(3), Vec.Of(1, 2, 3)));
    }

    [Fact]
    public void Cosine_KhacKichThuoc_ThiNemLoi()
    {
        Assert.Throws<ArgumentException>(() => Similarity.Cosine(Vec.Of(1), Vec.Of(1, 2)));
    }

    /// <summary>Điểm cốt lõi: one-hot không diễn đạt được độ giống nhau.</summary>
    [Fact]
    public void OneHot_MoiCapTuKhacNhauDeuCoSimilarityBangKhong()
    {
        const int size = 16;

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                double cosine = Similarity.Cosine(Similarity.OneHot(i, size), Similarity.OneHot(j, size));
                Assert.Equal(i == j ? 1.0 : 0.0, cosine, precision: 12);
            }
        }
    }

    [Fact]
    public void EuclideanDistance_TinhDung()
    {
        // ‖[0,0] − [3,4]‖ = 5
        Assert.Equal(5.0, Similarity.EuclideanDistance(Vec.Of(0, 0), Vec.Of(3, 4)), precision: 10);
        Assert.Equal(0.0, Similarity.EuclideanDistance(Vec.Of(1, 2), Vec.Of(1, 2)), precision: 12);
    }
}

public class EmbeddingTableTests
{
    private static EmbeddingTable Build(int dimensions = 4) =>
        new(Vocabulary.FromWords(["cat", "dog", "car"]), dimensions, new Random(1));

    [Fact]
    public void Lookup_TraVeVectorDungKichThuoc()
    {
        var table = Build(dimensions: 4);

        Assert.Equal(4, table.Lookup("cat").Length);
        Assert.Equal(3, table.Count);
        Assert.Equal(4, table.Dimensions);
    }

    [Fact]
    public void Lookup_TheoTenVaTheoChiSo_ChoCungVector()
    {
        var table = Build();

        Assert.Same(table.Lookup("dog"), table.Lookup(table.Vocabulary.IndexOf("dog")));
    }

    [Fact]
    public void MoiTuCoVectorRieng()
    {
        var table = Build();

        Assert.NotSame(table.Lookup("cat"), table.Lookup("dog"));
        Assert.NotEqual(table.Lookup("cat").ToArray(), table.Lookup("dog").ToArray());
    }

    [Fact]
    public void GetSetParameters_KhepKin()
    {
        var table = Build();
        var original = table.GetParameters();

        Assert.Equal(table.Count * table.Dimensions, original.Length);

        table.SetParameters(new double[original.Length]);
        Assert.Equal(new double[original.Length], table.GetParameters());

        table.SetParameters(original);
        Assert.Equal(original, table.GetParameters());
    }

    [Fact]
    public void SetParameters_SaiSoLuong_ThiNemLoi()
    {
        Assert.Throws<ArgumentException>(() => Build().SetParameters([1.0]));
    }

    [Fact]
    public void AccumulateGradient_CongDon_VaZeroGradientsXoaSach()
    {
        var table = Build(dimensions: 2);
        int cat = table.Vocabulary.IndexOf("cat");

        table.ZeroGradients();
        table.AccumulateGradient(cat, Vec.Of(1, 2));
        table.AccumulateGradient(cat, Vec.Of(10, 20));

        Assert.Equal([11.0, 22.0], table.GradientOf(cat).ToArray());

        table.ZeroGradients();
        Assert.Equal([0.0, 0.0], table.GradientOf(cat).ToArray());
    }
}

public class SkipGramModelTests
{
    /// <summary>
    /// Kiểm tra gradient bằng tay:
    ///   center[a] = [1,0] , context[b] = [1,0] , label = 1
    ///   score = 1 ; p = sigmoid(1) = 0.7310585786
    ///   L = −log(p) = 0.3132616875
    ///   δ = p − 1 = −0.2689414214
    ///   ∂L/∂center[a] = δ·context[b] = [−0.2689414214, 0]
    ///   với lr = 1  ->  center[a] = [1,0] − [−0.2689,0] = [1.2689414214, 0]
    /// </summary>
    [Fact]
    public void MotBuocHoc_KhopVoiTinhTay()
    {
        var vocabulary = Vocabulary.FromWords(["a", "b"]);
        var model = new SkipGramModel(vocabulary, dimensions: 2, new Random(1),
            new StochasticGradientDescent(1.0));

        model.Center.SetParameters([1, 0, 0, 0]);    // center[a]=[1,0], center[b]=[0,0]
        model.Context.SetParameters([0, 0, 1, 0]);   // context[a]=[0,0], context[b]=[1,0]

        double loss = model.RunEpoch([new WordPair(Center: 0, Context: 1, Label: 1)]);

        const double expected = 1.2689414213699951;   // 1 + 0.2689414213699951

        Assert.Equal(0.3132616875182228, loss, precision: 10);
        Assert.Equal(expected, model.Center.Lookup("a")[0], precision: 10);
        Assert.Equal(0.0, model.Center.Lookup("a")[1], precision: 12);
        Assert.Equal(expected, model.Context.Lookup("b")[0], precision: 10);
        Assert.Equal(0.0, model.Context.Lookup("b")[1], precision: 12);

        // center[b] và context[a] đều là vector 0 nên gradient của chúng bằng 0 -> không đổi.
        Assert.Equal([0.0, 0.0], model.Center.Lookup("b").ToArray());
        Assert.Equal([0.0, 0.0], model.Context.Lookup("a").ToArray());
    }

    [Fact]
    public void Probability_LaSigmoidCuaTichVoHuong()
    {
        var model = new SkipGramModel(Vocabulary.FromWords(["a", "b"]), 2, new Random(1),
            new StochasticGradientDescent(0.1));

        model.Center.SetParameters([2, 0, 0, 0]);
        model.Context.SetParameters([0, 0, 3, 0]);

        // sigmoid(2·3) = sigmoid(6)
        Assert.Equal(1.0 / (1 + Math.Exp(-6)), model.Probability(0, 1), precision: 12);
    }

    [Fact]
    public void KhongCoCap_ThiNemLoi()
    {
        var model = new SkipGramModel(Vocabulary.FromWords(["a"]), 2, new Random(1),
            new StochasticGradientDescent(0.1));

        Assert.Throws<ArgumentException>(() => model.RunEpoch([]));
    }

    [Fact]
    public void LossGiamDeuQuaTungEpoch()
    {
        var vocabulary = Corpus.BuildVocabulary();
        var rng = new Random(42);
        var model = new SkipGramModel(vocabulary, 8, rng, new StochasticGradientDescent(0.5));
        var pairs = Corpus.BuildPairs(vocabulary, rng);
        var losses = new List<double>();

        model.Train(pairs, 500, (_, loss) => losses.Add(loss));

        Assert.True(losses[^1] < losses[0] / 4, $"Loss phải giảm mạnh: {losses[0]} -> {losses[^1]}");
    }
}

public class CorpusTests
{
    [Fact]
    public void CatVaDog_KhongBaoGioLaCapDuong()
    {
        // Đây chính là lý do phải dùng HAI bảng embedding: cat và dog không
        // hề xuất hiện cùng nhau, chúng chỉ chia sẻ NGỮ CẢNH.
        var vocabulary = Corpus.BuildVocabulary();
        var pairs = Corpus.BuildPairs(vocabulary, new Random(1));

        int cat = vocabulary.IndexOf("cat");
        int dog = vocabulary.IndexOf("dog");

        Assert.DoesNotContain(pairs, p =>
            p.Label == 1 && ((p.Center == cat && p.Context == dog) || (p.Center == dog && p.Context == cat)));
    }

    [Fact]
    public void CatVaDog_ChiaSeCungNguCanh()
    {
        var vocabulary = Corpus.BuildVocabulary();
        var pairs = Corpus.BuildPairs(vocabulary, new Random(1));

        var catContexts = pairs
            .Where(p => p.Label == 1 && p.Center == vocabulary.IndexOf("cat"))
            .Select(p => vocabulary.WordAt(p.Context))
            .ToHashSet();

        var dogContexts = pairs
            .Where(p => p.Label == 1 && p.Center == vocabulary.IndexOf("dog"))
            .Select(p => vocabulary.WordAt(p.Context))
            .ToHashSet();

        Assert.Equal(catContexts, dogContexts);
    }

    [Fact]
    public void SinhDuCapDuongVaCapAm()
    {
        var vocabulary = Corpus.BuildVocabulary();
        var pairs = Corpus.BuildPairs(vocabulary, new Random(1), negativesPerPositive: 4);

        int positives = pairs.Count(p => p.Label == 1);
        int negatives = pairs.Count(p => p.Label == 0);

        Assert.True(positives > 0);
        Assert.Equal(positives * 4, negatives);
    }
}

public class LearnedEmbeddingTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(42)]
    [InlineData(7)]
    public void CatVaDog_TroThanhGiongNhau(int seed)
    {
        var table = EmbeddingDemo.TrainEmbeddings(seed);

        double catDog = Similarity.Cosine(table.Lookup("cat"), table.Lookup("dog"));
        double catCar = Similarity.Cosine(table.Lookup("cat"), table.Lookup("car"));
        double catApple = Similarity.Cosine(table.Lookup("cat"), table.Lookup("apple"));

        Assert.True(catDog > 0.8, $"cos(cat,dog) = {catDog} — phải cao");
        Assert.True(catDog > catCar + 0.5, $"cos(cat,dog)={catDog} phải cao hơn hẳn cos(cat,car)={catCar}");
        Assert.True(catDog > catApple + 0.5, $"cos(cat,dog)={catDog} phải cao hơn hẳn cos(cat,apple)={catApple}");
    }

    [Fact]
    public void TuGanNhatCuaCat_LaDog()
    {
        var table = EmbeddingDemo.TrainEmbeddings();

        Assert.Equal("dog", Similarity.NearestNeighbours(table, "cat", take: 1)[0].Word);
        Assert.Equal("cat", Similarity.NearestNeighbours(table, "dog", take: 1)[0].Word);
    }

    [Fact]
    public void CarGanCacTuVeXeCo()
    {
        var table = EmbeddingDemo.TrainEmbeddings();
        var neighbours = Similarity.NearestNeighbours(table, "car", take: 3).Select(n => n.Word).ToArray();

        Assert.Contains("vehicle", neighbours);
        Assert.DoesNotContain("cat", neighbours);
        Assert.DoesNotContain("dog", neighbours);
    }

    [Fact]
    public void AppleGanCacTuVeDoAn()
    {
        var table = EmbeddingDemo.TrainEmbeddings();
        var neighbours = Similarity.NearestNeighbours(table, "apple", take: 3).Select(n => n.Word).ToArray();

        Assert.True(
            neighbours.Intersect(["eat", "fruit", "sweet", "food"]).Any(),
            $"apple phải gần nhóm đồ ăn, nhưng gần: {string.Join(", ", neighbours)}");
    }

    [Fact]
    public void EmbeddingHocDuoc_TotHonHanOneHot()
    {
        var table = EmbeddingDemo.TrainEmbeddings();
        int size = table.Count;
        var vocabulary = table.Vocabulary;

        double embeddingCatDog = Similarity.Cosine(table.Lookup("cat"), table.Lookup("dog"));
        double oneHotCatDog = Similarity.Cosine(
            Similarity.OneHot(vocabulary.IndexOf("cat"), size),
            Similarity.OneHot(vocabulary.IndexOf("dog"), size));

        Assert.Equal(0.0, oneHotCatDog, precision: 12);
        Assert.True(embeddingCatDog > 0.8);
    }
}
