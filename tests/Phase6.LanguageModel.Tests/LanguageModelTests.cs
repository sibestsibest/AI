using Ai.Phase6.Demos;
using Ai.Phase6.Language;
using Ai.Phase6.LinAlg;
using Ai.Phase6.Training;

namespace Ai.Phase6.Tests;

public class SoftmaxTests
{
    /// <summary>
    /// logits [1,2,3]: e¹=2.71828, e²=7.38906, e³=20.08554, tổng=30.19287
    /// -> [0.09003057, 0.24472847, 0.66524096]
    /// </summary>
    [Fact]
    public void Apply_TrungVoiKetQuaTinhTay()
    {
        var p = Softmax.Apply(Vec.Of(1, 2, 3));

        Assert.Equal(0.09003057317038046, p[0], precision: 10);
        Assert.Equal(0.24472847105479764, p[1], precision: 10);
        Assert.Equal(0.6652409557748219, p[2], precision: 10);
    }

    [Fact]
    public void TongLuonBangMot()
    {
        var rng = new Random(1);

        for (int trial = 0; trial < 50; trial++)
        {
            var logits = new Vec(7);
            for (int i = 0; i < logits.Length; i++) logits[i] = rng.NextDouble() * 20 - 10;

            Assert.Equal(1.0, Softmax.Apply(logits).Sum(), precision: 12);
        }
    }

    [Fact]
    public void MoiXacSuatDeuDuong()
    {
        var p = Softmax.Apply(Vec.Of(-100, 0, 100));

        Assert.All(Enumerable.Range(0, p.Length), i => Assert.True(p[i] >= 0));
    }

    [Fact]
    public void LogitsBangNhau_ChoPhanBoDeu()
    {
        var p = Softmax.Apply(Vec.Of(0, 0, 0));

        Assert.All(Enumerable.Range(0, 3), i => Assert.Equal(1.0 / 3, p[i], precision: 12));
    }

    /// <summary>softmax(z) = softmax(z + c) — cộng hằng số không đổi kết quả.</summary>
    [Fact]
    public void BatBienVoiPhepCongHangSo()
    {
        var a = Softmax.Apply(Vec.Of(1, 2, 3));
        var b = Softmax.Apply(Vec.Of(1001, 1002, 1003));

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(a[i], b[i], precision: 12);
        }
    }

    /// <summary>Không trừ max trước khi lấy e^x thì e^1000 = Infinity và ra NaN.</summary>
    [Fact]
    public void KhongTranSoVoiLogitsRatLon()
    {
        var p = Softmax.Apply(Vec.Of(1000, 1001, 1002));

        Assert.All(Enumerable.Range(0, 3), i => Assert.True(double.IsFinite(p[i])));
        Assert.Equal(1.0, p.Sum(), precision: 12);
    }

    [Fact]
    public void GiuNguyenThuTu()
    {
        var p = Softmax.Apply(Vec.Of(0.5, 3.2, -1.0, 2.0));

        Assert.True(p[1] > p[3]);
        Assert.True(p[3] > p[0]);
        Assert.True(p[0] > p[2]);
    }

    [Fact]
    public void NhietDoThap_LamPhanBoNhonHon()
    {
        var logits = Vec.Of(1, 2, 3);

        var cold = Softmax.Apply(logits, 0.1);
        var normal = Softmax.Apply(logits, 1.0);
        var hot = Softmax.Apply(logits, 10.0);

        Assert.True(cold[2] > normal[2]);   // T thấp -> dồn vào từ mạnh nhất
        Assert.True(normal[2] > hot[2]);    // T cao  -> phẳng dần về đều
        Assert.True(hot[2] - hot[0] < 0.1);
    }

    [Fact]
    public void NhietDoKhongDuong_ThiNemLoi()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Softmax.Apply(Vec.Of(1, 2), 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Softmax.Apply(Vec.Of(1, 2), -1));
    }
}

public class SoftmaxCrossEntropyTests
{
    private static readonly SoftmaxCrossEntropy Loss = new();

    /// <summary>p₃ = 0.66524096 -> L = −log(0.66524096) = 0.40760596</summary>
    [Fact]
    public void Compute_TrungVoiKetQuaTinhTay()
    {
        Assert.Equal(0.40760596444438024, Loss.Compute(Vec.Of(1, 2, 3), Vec.Of(0, 0, 1)), precision: 10);
    }

    [Theory]
    [InlineData(0, 2.407605964444380)]    // −log(0.09003057)
    [InlineData(1, 1.407605964444380)]    // −log(0.24472847)
    [InlineData(2, 0.407605964444380)]    // −log(0.66524096)
    public void LossCangLonKhiDapAnDungCangItXacSuat(int targetIndex, double expected)
    {
        var target = new Vec(3);
        target[targetIndex] = 1;

        Assert.Equal(expected, Loss.Compute(Vec.Of(1, 2, 3), target), precision: 10);
    }

    [Fact]
    public void PhanBoDeu_ChoLossBangLogV()
    {
        // Đoán bừa trên V phương án -> L = −log(1/V) = log(V), perplexity = V
        for (int v = 2; v <= 8; v++)
        {
            var target = new Vec(v);
            target[0] = 1;

            double loss = Loss.Compute(new Vec(v), target);

            Assert.Equal(Math.Log(v), loss, precision: 10);
            Assert.Equal(v, SoftmaxCrossEntropy.Perplexity(loss), precision: 8);
        }
    }

    /// <summary>Điểm đẹp nhất: ∂L/∂logits rút gọn thành đúng p − y.</summary>
    [Fact]
    public void Gradient_BangXacSuatTruTarget()
    {
        var logits = Vec.Of(1, 2, 3);
        var target = Vec.Of(0, 0, 1);

        var gradient = Loss.Gradient(logits, target);
        var probabilities = Softmax.Apply(logits);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(probabilities[i] - target[i], gradient[i], precision: 12);
        }

        // Đáp án đúng có gradient ÂM (bị đẩy lên), các đáp án sai gradient DƯƠNG.
        Assert.True(gradient[2] < 0);
        Assert.True(gradient[0] > 0 && gradient[1] > 0);
    }

    [Fact]
    public void Gradient_TongBangKhong()
    {
        // Σ(pᵢ − yᵢ) = 1 − 1 = 0 : xác suất bị lấy từ chỗ này bù sang chỗ kia.
        Assert.Equal(0.0, Loss.Gradient(Vec.Of(0.3, -1.2, 2.5), Vec.Of(0, 1, 0)).Sum(), precision: 12);
    }

    [Fact]
    public void Gradient_TrungVoiDaoHamSo()
    {
        var logits = Vec.Of(0.7, -1.3, 2.2, 0.1);
        var target = Vec.Of(0, 0, 1, 0);
        const double h = 1e-6;

        var analytic = Loss.Gradient(logits, target);

        for (int i = 0; i < logits.Length; i++)
        {
            var plus = logits.ToArray();
            var minus = logits.ToArray();
            plus[i] += h;
            minus[i] -= h;

            double numeric = (Loss.Compute(new Vec(plus), target) - Loss.Compute(new Vec(minus), target)) / (2 * h);

            Assert.True(Math.Abs(numeric - analytic[i]) < 1e-6, $"#{i}: {analytic[i]} vs {numeric}");
        }
    }

    [Fact]
    public void Perplexity_LaEmuLoss()
    {
        Assert.Equal(1.0, SoftmaxCrossEntropy.Perplexity(0), precision: 12);
        Assert.Equal(Math.E, SoftmaxCrossEntropy.Perplexity(1), precision: 12);
    }
}

public class TokenizerTests
{
    private static readonly Tokenizer Sut = new(TextCorpus.Basic);

    [Fact]
    public void TokenDacBiet_CoChiSo0Va1()
    {
        Assert.Equal(0, Sut.StartId);
        Assert.Equal(1, Sut.EndId);
        Assert.Equal(Tokenizer.StartToken, Sut.Vocabulary.WordAt(0));
        Assert.Equal(Tokenizer.EndToken, Sut.Vocabulary.WordAt(1));
    }

    [Fact]
    public void TuDien_ChuaDuMoiTuVaKhongTrungLap()
    {
        // <s>, </s>, I, like, dogs, cats, programming, love = 8
        Assert.Equal(8, Sut.Vocabulary.Count);
        Assert.Equal(Sut.Vocabulary.Count, Sut.Vocabulary.Words.Distinct().Count());
    }

    [Fact]
    public void EncodeDecode_KhepKin()
    {
        foreach (string sentence in TextCorpus.Basic)
        {
            Assert.Equal(sentence, Sut.Decode(Sut.Encode(sentence)));
        }
    }

    [Fact]
    public void EncodeWithBoundaries_ThemDuTokenBien()
    {
        var tokens = Sut.EncodeWithBoundaries("I like cats", contextSize: 2);

        Assert.Equal("<s> <s> I like cats </s>", Sut.Decode(tokens));
        Assert.Equal(6, tokens.Length);
    }

    [Fact]
    public void Split_BoQuaKhoangTrangThua()
    {
        Assert.Equal(["I", "like", "cats"], Tokenizer.Split("  I   like  cats  "));
    }

    [Fact]
    public void TuLa_ThiNemLoi()
    {
        Assert.Throws<KeyNotFoundException>(() => Sut.Encode("dragon"));
    }
}

public class LanguageModelTests
{
    [Fact]
    public void BuildSamples_SinhDungSoMau()
    {
        var model = LanguageModelDemo.Build(TextCorpus.Basic);
        var samples = model.BuildSamples(TextCorpus.Basic);

        // Mỗi câu 3 từ -> [<s>,<s>,I,like,cats,</s>] -> 4 mẫu. 5 câu -> 20.
        Assert.Equal(20, samples.Length);
        Assert.All(samples, s => Assert.Equal(model.ContextSize, s.Context.Length));
    }

    [Fact]
    public void Predict_TraVePhanBoXacSuatHopLe()
    {
        var model = LanguageModelDemo.Build(TextCorpus.Basic);
        var p = model.Predict("I like");

        Assert.Equal(model.VocabularySize, p.Length);
        Assert.Equal(1.0, p.Sum(), precision: 12);
        Assert.All(Enumerable.Range(0, p.Length), i => Assert.InRange(p[i], 0.0, 1.0));
    }

    [Fact]
    public void NguCanhNganHonCuaSo_DuocDemBangTokenDau()
    {
        var model = LanguageModelDemo.Build(TextCorpus.Basic);

        // "I" chỉ có 1 token nhưng cửa sổ là 2 -> phải tự đệm <s>, không nổ.
        Assert.Equal(1.0, model.Predict("I").Sum(), precision: 12);
    }

    [Fact]
    public void LossGiamManhKhiTrain()
    {
        var model = LanguageModelDemo.Build(TextCorpus.Basic);
        var samples = model.BuildSamples(TextCorpus.Basic);

        double before = model.AverageLoss(samples);
        model.Train(samples, 1000);
        double after = model.AverageLoss(samples);

        Assert.True(before > 1.5, $"Model chưa train phải lúng túng: loss = {before}");
        Assert.True(after < 0.5, $"Sau train loss phải thấp: {after}");
    }

    /// <summary>
    /// Với 5 câu gốc, phân bố THẬT sau "I like" là đều 1/3 mỗi từ.
    /// Model học đúng phải tái tạo lại đúng con số đó.
    /// </summary>
    [Fact]
    public void HocDungPhanBoDeu()
    {
        var model = LanguageModelDemo.Train(TextCorpus.Basic);
        var vocabulary = model.Tokenizer.Vocabulary;
        var p = model.Predict("I like");

        foreach (string word in new[] { "cats", "dogs", "programming" })
        {
            Assert.Equal(1.0 / 3, p[vocabulary.IndexOf(word)], precision: 2);
        }

        // "I love" chỉ đi với cats và programming.
        var q = model.Predict("I love");
        Assert.Equal(0.5, q[vocabulary.IndexOf("cats")], precision: 2);
        Assert.Equal(0.5, q[vocabulary.IndexOf("programming")], precision: 2);
        Assert.True(q[vocabulary.IndexOf("dogs")] < 0.01);
    }

    /// <summary>Model học TẦN SUẤT, không chỉ học tập hợp các từ có thể.</summary>
    [Fact]
    public void HocDungTanSuatKhiNguLieuKhongDeu()
    {
        var model = LanguageModelDemo.Train(TextCorpus.Weighted);
        var vocabulary = model.Tokenizer.Vocabulary;
        var p = model.Predict("I like");

        Assert.Equal(3.0 / 6, p[vocabulary.IndexOf("cats")], precision: 2);
        Assert.Equal(2.0 / 6, p[vocabulary.IndexOf("dogs")], precision: 2);
        Assert.Equal(1.0 / 6, p[vocabulary.IndexOf("programming")], precision: 2);
    }

    [Fact]
    public void HocDuocKhiNaoKetThucCau()
    {
        var model = LanguageModelDemo.Train(TextCorpus.Basic);
        var p = model.Predict("like cats");

        Assert.True(p[model.Tokenizer.EndId] > 0.95,
            $"Sau 'like cats' phải là hết câu, nhưng p = {p[model.Tokenizer.EndId]}");
    }

    [Fact]
    public void Generate_SinhRaCauCoTrongNguLieu()
    {
        var model = LanguageModelDemo.Train(TextCorpus.Basic);
        var rng = new Random(7);

        for (int trial = 0; trial < 20; trial++)
        {
            string sentence = "I " + model.Generate("I", maxTokens: 6, temperature: 1.0, rng);
            Assert.Contains(sentence, TextCorpus.Basic);
        }
    }

    [Fact]
    public void GenerateGreedy_LuonChoCungKetQua()
    {
        var model = LanguageModelDemo.Train(TextCorpus.Basic);

        string first = model.Generate("I like");
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(first, model.Generate("I like"));
        }
    }

    /// <summary>Gradient phải chảy tới tận bảng embedding, không dừng ở layer đầu.</summary>
    [Fact]
    public void EmbeddingDuocCapNhatTrongLucTrain()
    {
        var model = LanguageModelDemo.Build(TextCorpus.Basic);
        var before = model.Embeddings.GetParameters();

        model.Train(model.BuildSamples(TextCorpus.Basic), 10);

        var after = model.Embeddings.GetParameters();
        Assert.NotEqual(before, after);
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Max() > 1e-6);
    }

    /// <summary>
    /// Embedding là PHỤ PHẨM của việc dự đoán từ: 'cats' và 'dogs' đứng ở
    /// cùng vị trí ngữ pháp nên vector của chúng phải xích lại gần nhau.
    /// </summary>
    [Fact]
    public void EmbeddingHocDuocQuanHeNguPhap()
    {
        var model = LanguageModelDemo.Train(TextCorpus.Basic);
        var table = model.Embeddings;

        double catsDogs = Embeddings.Similarity.Cosine(table.Lookup("cats"), table.Lookup("dogs"));
        double catsStart = Embeddings.Similarity.Cosine(table.Lookup("cats"), table.Lookup("<s>"));

        Assert.True(catsDogs > 0.5, $"cos(cats,dogs) = {catsDogs}");
        Assert.True(catsDogs > catsStart, $"{catsDogs} phải lớn hơn {catsStart}");
    }

    [Fact]
    public void KhongCoMau_ThiNemLoi()
    {
        var model = LanguageModelDemo.Build(TextCorpus.Basic);

        Assert.Throws<ArgumentException>(() => model.RunEpoch([]));
    }

    [Fact]
    public void SaiSoTokenNguCanh_ThiNemLoi()
    {
        var model = LanguageModelDemo.Build(TextCorpus.Basic);

        Assert.Throws<ArgumentException>(() => model.Logits([1, 2, 3]));
    }
}
