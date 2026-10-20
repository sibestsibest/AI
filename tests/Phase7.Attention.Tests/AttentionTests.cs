using Ai.Phase7.Attention;
using Ai.Phase7.Demos;
using Ai.Phase7.LinAlg;

namespace Ai.Phase7.Tests;

/// <summary>
/// Ví dụ 3 token tính tay:
///   Q = K = [[1,0],[0,1],[1,1]] , V = [[10,0],[0,10],[5,5]] , d = 2
///
///   QKᵀ         = [[1,0,1],[0,1,1],[1,1,2]]
///   /√2         = [[0.70711,0,0.70711],[0,0.70711,0.70711],[0.70711,0.70711,1.41421]]
///   softmax hàng 0: e^0.70711 = 2.0281150, tổng = 5.0562300
///                   -> [0.4011121, 0.1977758, 0.4011121]
///   output hàng 0  = 0.4011121·[10,0] + 0.1977758·[0,10] + 0.4011121·[5,5]
///                  = [6.0166815, 3.9833185]
/// </summary>
public class HandCalculatedAttentionTests
{
    [Fact]
    public void Scores_ChiaDungChoCanBacHaiD()
    {
        var (q, k, v) = AttentionDemo.HandExample();

        var scores = ScaledDotProductAttention.Compute(q, k, v).Scores;
        double s = 1.0 / Math.Sqrt(2);

        Assert.Equal(s, scores[0, 0], precision: 10);
        Assert.Equal(0.0, scores[0, 1], precision: 10);
        Assert.Equal(s, scores[0, 2], precision: 10);
        Assert.Equal(2 * s, scores[2, 2], precision: 10);
    }

    [Fact]
    public void Weights_TrungVoiKetQuaTinhTay()
    {
        var (q, k, v) = AttentionDemo.HandExample();

        var w = ScaledDotProductAttention.Compute(q, k, v).Weights;

        Assert.Equal(0.4011121, w[0, 0], precision: 6);
        Assert.Equal(0.1977758, w[0, 1], precision: 6);
        Assert.Equal(0.4011121, w[0, 2], precision: 6);

        // Hàng 2: e^0.7071068 = 2.0281150 (×2), e^1.4142136 = 4.1132504, tổng = 8.1694803
        Assert.Equal(0.2482551, w[2, 0], precision: 6);   // 2.0281150 / 8.1694803
        Assert.Equal(0.5034898, w[2, 2], precision: 6);   // 4.1132504 / 8.1694803
    }

    [Fact]
    public void Output_TrungVoiKetQuaTinhTay()
    {
        var (q, k, v) = AttentionDemo.HandExample();

        var output = ScaledDotProductAttention.Compute(q, k, v).Output;

        Assert.Equal(6.0166814, output[0, 0], precision: 6);
        Assert.Equal(3.9833186, output[0, 1], precision: 6);

        // Mọi hàng V đều có tổng 10, nên mọi hàng đầu ra cũng phải có tổng 10.
        for (int r = 0; r < output.Rows; r++)
        {
            Assert.Equal(10.0, output.Row(r).Sum(), precision: 10);
        }

        // Hàng 2 đối xứng hoàn hảo -> đúng bằng [5, 5].
        Assert.Equal(5.0, output[2, 0], precision: 10);
        Assert.Equal(5.0, output[2, 1], precision: 10);
    }
}

public class AttentionPropertyTests
{
    private static Mat RandomMatrix(int rows, int cols, Random rng)
    {
        var m = new Mat(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++) m[r, c] = rng.NextDouble() * 4 - 2;
        }

        return m;
    }

    [Fact]
    public void MoiHangTrongSoCoTongBangMot()
    {
        var rng = new Random(1);

        for (int trial = 0; trial < 20; trial++)
        {
            var result = ScaledDotProductAttention.Compute(
                RandomMatrix(5, 4, rng), RandomMatrix(5, 4, rng), RandomMatrix(5, 3, rng));

            for (int i = 0; i < result.Weights.Rows; i++)
            {
                Assert.Equal(1.0, result.Weights.Row(i).Sum(), precision: 10);
            }
        }
    }

    [Fact]
    public void MoiTrongSoDeuKhongAm()
    {
        var rng = new Random(2);
        var result = ScaledDotProductAttention.Compute(
            RandomMatrix(4, 8, rng), RandomMatrix(4, 8, rng), RandomMatrix(4, 2, rng));

        for (int r = 0; r < result.Weights.Rows; r++)
        {
            for (int c = 0; c < result.Weights.Cols; c++)
            {
                Assert.True(result.Weights[r, c] >= 0);
            }
        }
    }

    /// <summary>
    /// Đầu ra là TỔ HỢP LỒI của các hàng V, nên không thể vượt ra ngoài
    /// khoảng [min, max] của từng cột V. Attention chỉ trộn, không sáng tạo
    /// giá trị mới.
    /// </summary>
    [Fact]
    public void DauRaLuonNamGiuaMinVaMaxCuaV()
    {
        var rng = new Random(3);
        var v = RandomMatrix(6, 3, rng);

        var result = ScaledDotProductAttention.Compute(
            RandomMatrix(6, 4, rng), RandomMatrix(6, 4, rng), v);

        for (int col = 0; col < v.Cols; col++)
        {
            double min = Enumerable.Range(0, v.Rows).Min(r => v[r, col]);
            double max = Enumerable.Range(0, v.Rows).Max(r => v[r, col]);

            for (int row = 0; row < result.Output.Rows; row++)
            {
                Assert.InRange(result.Output[row, col], min - 1e-9, max + 1e-9);
            }
        }
    }

    [Fact]
    public void MoiHangVGiongNhau_ThiDauRaBangChinhHangDo()
    {
        var v = Mat.FromRows([7, -3], [7, -3], [7, -3]);
        var rng = new Random(4);

        var output = ScaledDotProductAttention
            .Compute(RandomMatrix(3, 2, rng), RandomMatrix(3, 2, rng), v).Output;

        for (int r = 0; r < 3; r++)
        {
            Assert.Equal(7.0, output[r, 0], precision: 10);
            Assert.Equal(-3.0, output[r, 1], precision: 10);
        }
    }

    [Fact]
    public void QueryBangKhong_ChoPhanBoDeu()
    {
        // Mọi score = 0 -> softmax đều -> đầu ra là trung bình cộng của V.
        var v = Mat.FromRows([0, 0], [3, 6], [6, 12]);

        var result = ScaledDotProductAttention.Compute(new Mat(3, 2), Mat.FromRows([1, 1], [2, 2], [3, 3]), v);

        Assert.Equal(1.0 / 3, result.Weights[0, 0], precision: 10);
        Assert.Equal(3.0, result.Output[0, 0], precision: 10);   // (0+3+6)/3
        Assert.Equal(6.0, result.Output[0, 1], precision: 10);   // (0+6+12)/3
    }

    [Fact]
    public void ScoreVuotTroi_ThiGanNhuChiLayVectorDo()
    {
        // Query khớp cực mạnh với key 1 -> gần như copy V hàng 1.
        var q = Mat.FromRows([100, 0]);
        var k = Mat.FromRows([0, 1], [1, 0], [0, 1]);
        var v = Mat.FromRows([1, 1], [99, 99], [1, 1]);

        var result = ScaledDotProductAttention.Compute(q, k, v);

        Assert.True(result.Weights[0, 1] > 0.9999);
        Assert.Equal(99.0, result.Output[0, 0], precision: 2);
    }

    [Fact]
    public void QBangK_ThiMaTranScoreDoiXung()
    {
        var rng = new Random(5);
        var x = RandomMatrix(4, 3, rng);

        var scores = ScaledDotProductAttention.SelfAttention(x).Scores;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.Equal(scores[i, j], scores[j, i], precision: 12);
            }
        }
    }

    [Fact]
    public void SaiKichThuoc_ThiNemLoi()
    {
        Assert.Throws<ArgumentException>(() => ScaledDotProductAttention.Compute(
            Mat.FromRows([1, 2, 3]), Mat.FromRows([1, 2]), Mat.FromRows([1, 2])));

        Assert.Throws<ArgumentException>(() => ScaledDotProductAttention.Compute(
            Mat.FromRows([1, 2]), Mat.FromRows([1, 2], [3, 4]), Mat.FromRows([1, 2])));
    }
}

public class CausalMaskTests
{
    [Fact]
    public void TamGiacTrenBangKhong()
    {
        var x = Mat.FromRows([1.0, 0.2], [0.3, 1.0], [0.8, 0.5], [0.1, 0.1]);

        var weights = ScaledDotProductAttention.SelfAttention(x, causalMask: true).Weights;

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = i + 1; j < weights.Cols; j++)
            {
                Assert.Equal(0.0, weights[i, j], precision: 12);
            }
        }
    }

    [Fact]
    public void MoiHangVanCoTongBangMot()
    {
        var x = Mat.FromRows([1.0, 0.2], [0.3, 1.0], [0.8, 0.5], [0.1, 0.1]);

        var weights = ScaledDotProductAttention.SelfAttention(x, causalMask: true).Weights;

        for (int i = 0; i < weights.Rows; i++)
        {
            Assert.Equal(1.0, weights.Row(i).Sum(), precision: 10);
        }
    }

    [Fact]
    public void TokenDauTienChiNhinDuocChinhNo()
    {
        var x = Mat.FromRows([1.0, 0.2], [0.3, 1.0], [0.8, 0.5]);

        var weights = ScaledDotProductAttention.SelfAttention(x, causalMask: true).Weights;

        Assert.Equal(1.0, weights[0, 0], precision: 12);
    }

    /// <summary>
    /// Điểm cốt lõi: đổi token PHÍA SAU không được làm thay đổi đầu ra của
    /// token phía trước. Nếu có thay đổi thì mô hình đang "nhìn trộm đáp án".
    /// </summary>
    [Fact]
    public void DoiTokenPhiaSau_KhongAnhHuongTokenPhiaTruoc()
    {
        var original = Mat.FromRows([1.0, 0.2], [0.3, 1.0], [0.8, 0.5], [0.1, 0.1]);
        var changed = Mat.FromRows([1.0, 0.2], [0.3, 1.0], [-5.0, 9.0], [7.0, -2.0]);

        var a = ScaledDotProductAttention.SelfAttention(original, causalMask: true).Output;
        var b = ScaledDotProductAttention.SelfAttention(changed, causalMask: true).Output;

        for (int i = 0; i <= 1; i++)
        {
            for (int c = 0; c < a.Cols; c++)
            {
                Assert.Equal(a[i, c], b[i, c], precision: 12);
            }
        }
    }

    [Fact]
    public void KhongMask_ThiDoiTokenPhiaSauCoAnhHuong()
    {
        var original = Mat.FromRows([1.0, 0.2], [0.3, 1.0], [0.8, 0.5]);
        var changed = Mat.FromRows([1.0, 0.2], [0.3, 1.0], [-5.0, 9.0]);

        var a = ScaledDotProductAttention.SelfAttention(original).Output;
        var b = ScaledDotProductAttention.SelfAttention(changed).Output;

        Assert.NotEqual(a[0, 0], b[0, 0], precision: 6);
    }
}

public class AttentionHeadTests
{
    [Fact]
    public void ChieuDonVi_ChoKetQuaGiongSelfAttention()
    {
        var x = Mat.FromRows([1.0, 0.1, 0.0], [0.9, 0.2, 0.0], [0.0, 0.1, 1.0]);
        var identity = AttentionHead.Identity(3);
        var head = new AttentionHead(identity, identity, identity);

        var viaHead = head.Forward(x).Output;
        var viaDirect = ScaledDotProductAttention.SelfAttention(x).Output;

        for (int r = 0; r < viaHead.Rows; r++)
        {
            for (int c = 0; c < viaHead.Cols; c++)
            {
                Assert.Equal(viaDirect[r, c], viaHead[r, c], precision: 12);
            }
        }
    }

    [Fact]
    public void ParameterCount_BangBaMaTranChieu()
    {
        var head = new AttentionHead(modelDimensions: 16, headDimensions: 4, new Random(1));

        Assert.Equal(3 * 16 * 4, head.ParameterCount);
    }

    [Fact]
    public void DauRaCoDungSoChieuCuaHead()
    {
        var head = new AttentionHead(modelDimensions: 6, headDimensions: 3, new Random(1));
        var x = new Mat(5, 6);

        var result = head.Forward(x);

        Assert.Equal(5, result.Output.Rows);
        Assert.Equal(3, result.Output.Cols);
        Assert.Equal(5, result.Weights.Rows);
        Assert.Equal(5, result.Weights.Cols);
    }

    [Fact]
    public void SaiSoChieuDauVao_ThiNemLoi()
    {
        var head = new AttentionHead(modelDimensions: 4, headDimensions: 2, new Random(1));

        Assert.Throws<ArgumentException>(() => head.Forward(new Mat(3, 5)));
    }

    [Fact]
    public void MaskDuocTruyenXuongPhepTinhAttention()
    {
        var head = new AttentionHead(4, 2, new Random(1), causalMask: true);

        var weights = head.Forward(new Mat(3, 4)).Weights;

        Assert.Equal(0.0, weights[0, 1], precision: 12);
        Assert.Equal(0.0, weights[0, 2], precision: 12);
        Assert.Equal(0.0, weights[1, 2], precision: 12);
    }
}

public class MatRowTests
{
    [Fact]
    public void RowVaSetRow_KhepKin()
    {
        var m = Mat.FromRows([1, 2, 3], [4, 5, 6]);

        Assert.Equal([1.0, 2.0, 3.0], m.Row(0).ToArray());

        m.SetRow(0, Vec.Of(9, 8, 7));
        Assert.Equal([9.0, 8.0, 7.0], m.Row(0).ToArray());
        Assert.Equal([4.0, 5.0, 6.0], m.Row(1).ToArray());
    }

    [Fact]
    public void FromRowVectors_DungKichThuoc()
    {
        var m = Mat.FromRowVectors(Vec.Of(1, 2), Vec.Of(3, 4), Vec.Of(5, 6));

        Assert.Equal(3, m.Rows);
        Assert.Equal(2, m.Cols);
        Assert.Equal(5.0, m[2, 0]);
    }

    [Fact]
    public void SetRow_SaiKichThuoc_ThiNemLoi()
    {
        Assert.Throws<ArgumentException>(() => Mat.FromRows([1, 2]).SetRow(0, Vec.Of(1, 2, 3)));
    }
}
