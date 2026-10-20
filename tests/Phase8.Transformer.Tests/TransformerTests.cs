using Ai.Phase8.Activations;
using Ai.Phase8.Demos;
using Ai.Phase8.Language;
using Ai.Phase8.LinAlg;
using Ai.Phase8.Training;
using Ai.Phase8.Transformer;

namespace Ai.Phase8.Tests;

public class PositionalEncodingTests
{
    [Fact]
    public void ViTriKhong_LaSinVaCosCuaGocKhong()
    {
        // pos = 0 -> mọi góc = 0 -> sin(0) = 0, cos(0) = 1, xen kẽ.
        var encoding = PositionalEncoding.Build(4, 6);

        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(i % 2 == 0 ? 0.0 : 1.0, encoding[0, i], precision: 12);
        }
    }

    [Fact]
    public void LuonNamTrongKhoangTruMotDenMot()
    {
        var encoding = PositionalEncoding.Build(50, 16);

        for (int r = 0; r < encoding.Rows; r++)
        {
            for (int c = 0; c < encoding.Cols; c++)
            {
                Assert.InRange(encoding[r, c], -1.0, 1.0);
            }
        }
    }

    [Fact]
    public void MoiViTriCoMotDauVanTayRieng()
    {
        var encoding = PositionalEncoding.Build(20, 16);

        for (int a = 0; a < encoding.Rows; a++)
        {
            for (int b = a + 1; b < encoding.Rows; b++)
            {
                Assert.NotEqual(encoding.Row(a).ToArray(), encoding.Row(b).ToArray());
            }
        }
    }

    [Fact]
    public void ChieuDauDoiNhanhHonChieuCuoi()
    {
        var encoding = PositionalEncoding.Build(10, 16);

        double firstDimensionChange = 0;
        double lastDimensionChange = 0;

        for (int pos = 1; pos < encoding.Rows; pos++)
        {
            firstDimensionChange += Math.Abs(encoding[pos, 0] - encoding[pos - 1, 0]);
            lastDimensionChange += Math.Abs(encoding[pos, 14] - encoding[pos - 1, 14]);
        }

        Assert.True(firstDimensionChange > lastDimensionChange * 10);
    }

    [Fact]
    public void Add_CongVaoTungHang()
    {
        var embeddings = Mat.FromRows([1, 1, 1, 1], [2, 2, 2, 2]);
        var encoding = PositionalEncoding.Build(4, 4);

        var result = PositionalEncoding.Add(embeddings, encoding);

        Assert.Equal(1 + encoding[0, 0], result[0, 0], precision: 12);
        Assert.Equal(2 + encoding[1, 3], result[1, 3], precision: 12);
    }

    [Fact]
    public void ChuoiDaiHonBangViTri_ThiNemLoi()
    {
        Assert.Throws<ArgumentException>(() =>
            PositionalEncoding.Add(new Mat(10, 4), PositionalEncoding.Build(4, 4)));
    }
}

public class LayerNormTests
{
    [Fact]
    public void MoiHangCoTrungBinhKhongVaPhuongSaiMot()
    {
        var norm = new LayerNorm("t", 4);
        var y = norm.Forward(Mat.FromRows([1, 2, 3, 4], [100, 200, 300, 400], [-5, 0, 5, 10]));

        for (int i = 0; i < y.Rows; i++)
        {
            var row = y.Row(i);
            double mean = row.Sum() / row.Length;
            double variance = Enumerable.Range(0, row.Length).Sum(j => (row[j] - mean) * (row[j] - mean)) / row.Length;

            Assert.Equal(0.0, mean, precision: 10);
            Assert.Equal(1.0, variance, precision: 4);
        }
    }

    /// <summary>
    /// [1,2,3,4] và [100,200,300,400] chỉ khác nhau một hệ số nhân, nên sau
    /// chuẩn hoá phải gần như trùng khít. Chỉ "gần như" chứ không tuyệt đối:
    /// ε = 1e-5 cộng vào phương sai làm lệch tương đối ~4e-6 ở hàng có
    /// phương sai nhỏ (1.25) và ~4e-10 ở hàng có phương sai lớn (12500).
    /// Đây là cái giá phải trả để không bao giờ chia cho 0.
    /// </summary>
    [Fact]
    public void BatBienVoiThangDo()
    {
        var norm = new LayerNorm("t", 4);
        var y = norm.Forward(Mat.FromRows([1, 2, 3, 4], [100, 200, 300, 400]));

        for (int j = 0; j < 4; j++)
        {
            Assert.Equal(y[0, j], y[1, j], precision: 4);
        }
    }

    [Fact]
    public void GammaVaBetaDuocApDung()
    {
        var norm = new LayerNorm("t", 2);
        norm.Gamma.Value[0, 0] = 10;
        norm.Beta.Value[0, 1] = 5;

        var withoutScaling = new LayerNorm("u", 2).Forward(Mat.FromRows([1, 3]));
        var withScaling = norm.Forward(Mat.FromRows([1, 3]));

        Assert.Equal(withoutScaling[0, 0] * 10, withScaling[0, 0], precision: 10);
        Assert.Equal(withoutScaling[0, 1] + 5, withScaling[0, 1], precision: 10);
    }

    [Fact]
    public void Backward_TrungVoiDaoHamSo()
    {
        var rng = new Random(1);
        var norm = new LayerNorm("t", 5);
        norm.Gamma.Value[0, 2] = 1.7;    // γ khác 1 để test cả nhánh γ
        norm.Beta.Value[0, 3] = -0.4;

        var input = ModuleGradientCheck.RandomMat(4, 5, rng);

        double parameterError = ModuleGradientCheck.MaxParameterError(
            norm.Forward, norm.Backward,
            () => { foreach (var p in norm.Parameters()) p.ZeroGrad(); },
            [.. norm.Parameters()], input, rng);

        double inputError = ModuleGradientCheck.MaxInputError(
            norm.Forward, norm.Backward,
            () => { foreach (var p in norm.Parameters()) p.ZeroGrad(); },
            input, rng);

        Assert.True(parameterError < 1e-6, $"gradient tham số lệch {parameterError:E3}");
        Assert.True(inputError < 1e-6, $"gradient đầu vào lệch {inputError:E3}");
    }

    [Fact]
    public void BackwardTruocForward_ThiNemLoi()
    {
        Assert.Throws<InvalidOperationException>(() => new LayerNorm("t", 3).Backward(new Mat(2, 3)));
    }
}

public class DenseTests
{
    [Fact]
    public void Forward_TinhDungXNhanWCongB()
    {
        var dense = new Dense("t", 2, 3, new Identity(), new Random(1));
        dense.W.Value[0, 0] = 1; dense.W.Value[0, 1] = 2; dense.W.Value[0, 2] = 3;
        dense.W.Value[1, 0] = 4; dense.W.Value[1, 1] = 5; dense.W.Value[1, 2] = 6;
        dense.B.Value[0, 0] = 10;

        var y = dense.Forward(Mat.FromRows([1, 1]));

        Assert.Equal(15.0, y[0, 0], precision: 10);   // 1·1 + 1·4 + 10
        Assert.Equal(7.0, y[0, 1], precision: 10);    // 1·2 + 1·5
        Assert.Equal(9.0, y[0, 2], precision: 10);    // 1·3 + 1·6
    }

    [Theory]
    [InlineData("relu")]
    [InlineData("tanh")]
    [InlineData("identity")]
    public void Backward_TrungVoiDaoHamSo(string activationName)
    {
        IActivation activation = activationName switch
        {
            "relu" => new ReLU(),
            "tanh" => new Tanh(),
            _ => new Identity(),
        };

        var rng = new Random(2);
        var dense = new Dense("t", 4, 3, activation, rng);
        var input = ModuleGradientCheck.RandomMat(5, 4, rng);

        double parameterError = ModuleGradientCheck.MaxParameterError(
            dense.Forward, dense.Backward,
            () => { foreach (var p in dense.Parameters()) p.ZeroGrad(); },
            [.. dense.Parameters()], input, rng);

        Assert.True(parameterError < 1e-6, $"{activationName}: lệch {parameterError:E3}");
    }
}

public class SelfAttentionLayerTests
{
    [Fact]
    public void CausalMask_ChanTamGiacTren()
    {
        var layer = new SelfAttentionLayer("t", 4, 4, new Random(1), causalMask: true);
        layer.Forward(ModuleGradientCheck.RandomMat(4, 4, new Random(2)));

        var weights = layer.LastAttentionWeights!;
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = i + 1; j < weights.Cols; j++)
            {
                Assert.Equal(0.0, weights[i, j], precision: 12);
            }

            Assert.Equal(1.0, weights.Row(i).Sum(), precision: 10);
        }
    }

    /// <summary>
    /// Backward của attention là chỗ dễ sai nhất: Jacobian softmax, và X đi
    /// vào theo BA nhánh Q/K/V nên gradient phải cộng cả ba.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Backward_TrungVoiDaoHamSo(bool causalMask)
    {
        var rng = new Random(3);
        var layer = new SelfAttentionLayer("t", 5, 4, rng, causalMask);
        var input = ModuleGradientCheck.RandomMat(4, 5, rng);

        double parameterError = ModuleGradientCheck.MaxParameterError(
            layer.Forward, layer.Backward,
            () => { foreach (var p in layer.Parameters()) p.ZeroGrad(); },
            [.. layer.Parameters()], input, rng);

        double inputError = ModuleGradientCheck.MaxInputError(
            layer.Forward, layer.Backward,
            () => { foreach (var p in layer.Parameters()) p.ZeroGrad(); },
            input, rng);

        Assert.True(parameterError < 1e-6, $"mask={causalMask}: gradient tham số lệch {parameterError:E3}");
        Assert.True(inputError < 1e-6, $"mask={causalMask}: gradient đầu vào lệch {inputError:E3}");
    }

    [Fact]
    public void ParameterCount_LaBonMaTran()
    {
        var layer = new SelfAttentionLayer("t", 8, 4, new Random(1), false);

        Assert.Equal(4, layer.Parameters().Count());
        Assert.Equal(8 * 4 * 4, layer.Parameters().Sum(p => p.Count));
    }
}

public class TransformerBlockTests
{
    [Fact]
    public void GiuNguyenKichThuoc()
    {
        var block = new TransformerBlock("t", 6, 4, 12, new Random(1));

        var y = block.Forward(ModuleGradientCheck.RandomMat(3, 6, new Random(2)));

        Assert.Equal(3, y.Rows);
        Assert.Equal(6, y.Cols);
    }

    [Fact]
    public void Backward_TrungVoiDaoHamSo()
    {
        var rng = new Random(4);
        var block = new TransformerBlock("t", 5, 4, 8, rng);
        var input = ModuleGradientCheck.RandomMat(3, 5, rng);

        double parameterError = ModuleGradientCheck.MaxParameterError(
            block.Forward, block.Backward,
            () => { foreach (var p in block.Parameters()) p.ZeroGrad(); },
            [.. block.Parameters()], input, rng);

        double inputError = ModuleGradientCheck.MaxInputError(
            block.Forward, block.Backward,
            () => { foreach (var p in block.Parameters()) p.ZeroGrad(); },
            input, rng);

        Assert.True(parameterError < 1e-6, $"gradient tham số lệch {parameterError:E3}");
        Assert.True(inputError < 1e-6, $"gradient đầu vào lệch {inputError:E3}");
    }
}

public class MiniTransformerTests
{
    private static MiniTransformer BuildSmall(string[] corpus, int seed = 42) =>
        new(
            new Tokenizer(corpus),
            modelDimensions: 8,
            headDimensions: 8,
            feedForwardSize: 8,
            blockCount: 1,
            maxSequenceLength: 8,
            new Random(seed),
            new StochasticGradientDescent(0.1));

    [Fact]
    public void GradientCheck_ToanKienTrucDat()
    {
        var model = BuildSmall(TextCorpus.Basic);
        var samples = model.BuildSamples(TextCorpus.Basic).Take(3).ToArray();

        var check = TransformerGradientCheck.Check(model, samples);

        Assert.True(check.Passed,
            $"lệch {check.MaxAbsoluteError:E3} tại #{check.WorstIndex} " +
            $"(backprop {check.Analytic[check.WorstIndex]:E6}, số {check.Numerical[check.WorstIndex]:E6})");
    }

    [Fact]
    public void GradientCheck_VoiHaiKhoiChongLenNhau()
    {
        var model = new MiniTransformer(
            new Tokenizer(TextCorpus.Basic), 6, 6, 6, blockCount: 2, maxSequenceLength: 8,
            new Random(1), new StochasticGradientDescent(0.1));

        var check = TransformerGradientCheck.Check(model, model.BuildSamples(TextCorpus.Basic).Take(2).ToArray());

        Assert.True(check.Passed, $"lệch {check.MaxAbsoluteError:E3}");
    }

    [Fact]
    public void BuildSamples_SinhMoiTienToCuaCau()
    {
        var model = BuildSmall(TextCorpus.Basic);
        var samples = model.BuildSamples(["I like cats"]);

        // [<s>, I, like, cats, </s>] -> 4 mẫu, ngữ cảnh dài dần 1,2,3,4.
        Assert.Equal(4, samples.Length);
        Assert.Equal([1, 2, 3, 4], samples.Select(s => s.Context.Length).ToArray());
        Assert.Equal(model.Tokenizer.EndId, samples[^1].Target);
    }

    [Fact]
    public void Predict_LaPhanBoXacSuatHopLe()
    {
        var model = BuildSmall(TextCorpus.Basic);
        var p = model.Predict("I like");

        Assert.Equal(model.VocabularySize, p.Length);
        Assert.Equal(1.0, p.Sum(), precision: 12);
    }

    [Fact]
    public void GetSetParameters_KhepKin()
    {
        var model = BuildSmall(TextCorpus.Basic);
        var original = model.GetAllParameters();
        var before = model.Predict("I like").ToArray();

        model.SetAllParameters(new double[original.Length]);
        model.SetAllParameters(original);

        Assert.Equal(original, model.GetAllParameters());
        Assert.Equal(before, model.Predict("I like").ToArray());
    }

    [Fact]
    public void ParameterCount_KhopVoiMangPhang()
    {
        var model = BuildSmall(TextCorpus.Basic);

        Assert.Equal(model.ParameterCount, model.GetAllParameters().Length);
        Assert.Equal(model.ParameterCount, model.GetAllGradients().Length);
    }

    [Fact]
    public void LossGiamKhiTrain()
    {
        var model = BuildSmall(TextCorpus.Basic);
        var samples = model.BuildSamples(TextCorpus.Basic);

        double before = model.AverageLoss(samples);
        model.Train(samples, 300);
        double after = model.AverageLoss(samples);

        Assert.True(before > 1.5, $"loss ban đầu = {before}");
        Assert.True(after < before / 2, $"{before} -> {after}");
    }

    [Fact]
    public void ChuoiDaiQuaGioiHan_ThiNemLoi()
    {
        var model = BuildSmall(TextCorpus.Basic);

        Assert.Throws<ArgumentException>(() => model.Forward(new int[20]));
        Assert.Throws<ArgumentException>(() => model.Forward([]));
    }

    [Fact]
    public void BackwardTruocForward_ThiNemLoi()
    {
        var model = BuildSmall(TextCorpus.Basic);

        Assert.Throws<InvalidOperationException>(() => model.Backward(new Vec(model.VocabularySize)));
    }
}

public class TrainedTransformerTests
{
    [Fact]
    public void HocDungPhanBoTanSuat()
    {
        var model = TransformerDemo.Train(TextCorpus.Weighted, epochs: 1200);
        var v = model.Tokenizer.Vocabulary;
        var p = model.Predict("I like");

        Assert.Equal(0.5, p[v.IndexOf("cats")], precision: 1);
        Assert.Equal(1.0 / 3, p[v.IndexOf("dogs")], precision: 1);
        Assert.Equal(1.0 / 6, p[v.IndexOf("programming")], precision: 1);
    }

    [Fact]
    public void SinhRaCauCoTrongNguLieu()
    {
        var model = TransformerDemo.Train(TextCorpus.Basic, epochs: 1200);
        var rng = new Random(7);

        for (int trial = 0; trial < 10; trial++)
        {
            string sentence = "I " + model.Generate("I", maxTokens: 6, temperature: 1.0, rng);
            Assert.Contains(sentence, TextCorpus.Basic);
        }
    }

    [Fact]
    public void AttentionCuaModelDaTrain_TonTrongCausalMask()
    {
        var model = TransformerDemo.Train(TextCorpus.Basic, epochs: 200);
        model.Predict("I like");

        var weights = model.LastAttentionWeights!;
        Assert.Equal(1.0, weights[0, 0], precision: 10);

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = i + 1; j < weights.Cols; j++)
            {
                Assert.Equal(0.0, weights[i, j], precision: 12);
            }
        }
    }
}
