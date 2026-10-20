using Ai.Phase2.LinAlg;

namespace Ai.Phase2.Tests;

/// <summary>Vector và ma trận tự viết — mọi giá trị kỳ vọng đều tính tay.</summary>
public class VecTests
{
    [Fact]
    public void Dot_TinhDungTichVoHuong()
    {
        // 1·4 + 2·5 + 3·6 = 4 + 10 + 18 = 32
        Assert.Equal(32.0, Vec.Of(1, 2, 3).Dot(Vec.Of(4, 5, 6)), precision: 10);
    }

    [Fact]
    public void Dot_VuongGocThiBangKhong()
    {
        Assert.Equal(0.0, Vec.Of(1, 0).Dot(Vec.Of(0, 1)), precision: 10);
    }

    [Fact]
    public void Add_Subtract_Scale_Hadamard_TinhTungPhanTu()
    {
        var a = Vec.Of(1, 2, 3);
        var b = Vec.Of(4, 5, 6);

        Assert.Equal([5.0, 7.0, 9.0], a.Add(b).ToArray());
        Assert.Equal([3.0, 3.0, 3.0], b.Subtract(a).ToArray());
        Assert.Equal([2.0, 4.0, 6.0], a.Scale(2).ToArray());
        Assert.Equal([4.0, 10.0, 18.0], a.Hadamard(b).ToArray());
    }

    [Fact]
    public void Operators_TuongDuongVoiPhuongThuc()
    {
        var a = Vec.Of(1, 2);
        var b = Vec.Of(10, 20);

        Assert.Equal(a.Add(b).ToArray(), (a + b).ToArray());
        Assert.Equal(a.Subtract(b).ToArray(), (a - b).ToArray());
        Assert.Equal(a.Scale(3).ToArray(), (a * 3).ToArray());
        Assert.Equal(a.Scale(3).ToArray(), (3 * a).ToArray());
    }

    [Fact]
    public void Norm_LaDoDaiEuclid()
    {
        // √(3² + 4²) = 5 — tam giác 3-4-5
        Assert.Equal(5.0, Vec.Of(3, 4).Norm(), precision: 10);
    }

    [Fact]
    public void Map_ApHamLenTungPhanTu()
    {
        Assert.Equal([1.0, 4.0, 9.0], Vec.Of(1, 2, 3).Map(v => v * v).ToArray());
    }

    [Fact]
    public void Sum_CongTatCaPhanTu()
    {
        Assert.Equal(6.0, Vec.Of(1, 2, 3).Sum(), precision: 10);
    }

    [Fact]
    public void KhacKichThuoc_ThiNemLoi()
    {
        var a = Vec.Of(1, 2);
        var b = Vec.Of(1, 2, 3);

        Assert.Throws<ArgumentException>(() => a.Dot(b));
        Assert.Throws<ArgumentException>(() => a.Add(b));
        Assert.Throws<ArgumentException>(() => a.Hadamard(b));
    }

    [Fact]
    public void Constructor_SaoChepMang_KhongChiaSeThamChieu()
    {
        var source = new double[] { 1, 2, 3 };
        var v = new Vec(source);

        source[0] = 999;

        Assert.Equal(1.0, v[0]);
    }
}

public class MatTests
{
    [Fact]
    public void MultiplyVector_TinhDungTungHang()
    {
        // W = [[1,2],[3,4]] , x = [10,20]
        // hàng 0: 1·10 + 2·20 = 50 ; hàng 1: 3·10 + 4·20 = 110
        var w = Mat.FromRows([1, 2], [3, 4]);

        var result = w.Multiply(Vec.Of(10, 20));

        Assert.Equal([50.0, 110.0], result.ToArray());
    }

    [Fact]
    public void MultiplyMatrix_TinhDungTichMaTran()
    {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        var a = Mat.FromRows([1, 2], [3, 4]);
        var b = Mat.FromRows([5, 6], [7, 8]);

        var c = a.Multiply(b);

        Assert.Equal(19.0, c[0, 0], precision: 10);
        Assert.Equal(22.0, c[0, 1], precision: 10);
        Assert.Equal(43.0, c[1, 0], precision: 10);
        Assert.Equal(50.0, c[1, 1], precision: 10);
    }

    [Fact]
    public void Transpose_DoiHangThanhCot()
    {
        var m = Mat.FromRows([1, 2, 3], [4, 5, 6]);

        var t = m.Transpose();

        Assert.Equal(3, t.Rows);
        Assert.Equal(2, t.Cols);
        Assert.Equal(2.0, t[1, 0], precision: 10);
        Assert.Equal(4.0, t[0, 1], precision: 10);
    }

    [Fact]
    public void TransposeHaiLan_TraVeChinhNo()
    {
        var m = Mat.FromRows([1, 2, 3], [4, 5, 6]);

        var back = m.Transpose().Transpose();

        for (int r = 0; r < m.Rows; r++)
        {
            for (int c = 0; c < m.Cols; c++)
            {
                Assert.Equal(m[r, c], back[r, c]);
            }
        }
    }

    [Fact]
    public void SaiKichThuoc_ThiNemLoi()
    {
        var w = Mat.FromRows([1, 2], [3, 4]);

        Assert.Throws<ArgumentException>(() => w.Multiply(Vec.Of(1, 2, 3)));
        Assert.Throws<ArgumentException>(() => w.Multiply(Mat.FromRows([1, 2, 3])));
        Assert.Throws<ArgumentException>(() => Mat.FromRows([1, 2], [3]));
    }
}
