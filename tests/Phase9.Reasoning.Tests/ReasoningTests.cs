using Ai.Phase9.Reasoning;

namespace Ai.Phase9.Tests;

public class ParserTests
{
    [Fact]
    public void ParseFact_TachTenVaBieuThuc()
    {
        var fact = Parser.ParseFact("C = A + B");

        Assert.Equal("C", fact.Name);
        Assert.Equal("A + B", fact.Value.ToString());
        Assert.False(fact.IsLiteral);
    }

    [Fact]
    public void ParseFact_NhanBietHangSo()
    {
        var fact = Parser.ParseFact("A = 10");

        Assert.True(fact.IsLiteral);
        Assert.Equal("10", fact.Value.ToString());
    }

    [Theory]
    [InlineData("2 + 3 * 4", "2 + 3 * 4")]
    [InlineData("( 2 + 3 ) * 4", "2 + 3 * 4")]
    public void ParseExpression_GiuNguyenCauTruc(string input, string _)
    {
        Assert.NotNull(Parser.ParseExpression(input));
    }

    /// <summary>Nhân chia phải được nhóm trước cộng trừ.</summary>
    [Fact]
    public void ThuTuUuTien_NhanTruocCong()
    {
        var expression = (BinaryExpression)Parser.ParseExpression("2 + 3 * 4");

        Assert.Equal(Operator.Add, expression.Op);
        Assert.Equal("2", expression.Left.ToString());
        Assert.Equal("3 * 4", expression.Right.ToString());
    }

    [Fact]
    public void DauNgoac_DoiThuTuUuTien()
    {
        var expression = (BinaryExpression)Parser.ParseExpression("( 2 + 3 ) * 4");

        Assert.Equal(Operator.Multiply, expression.Op);
        Assert.Equal("2 + 3", expression.Left.ToString());
    }

    [Fact]
    public void PhepTruLienTiep_NhomTuTraiSangPhai()
    {
        // 10 - 3 - 2 phải là (10-3)-2 = 5, không phải 10-(3-2) = 9
        Assert.Equal(5.0, new ReasoningEngine().Evaluate(Parser.ParseExpression("10 - 3 - 2")), precision: 10);
    }

    [Fact]
    public void DauTruDungDau()
    {
        Assert.Equal(-5.0, new ReasoningEngine().Evaluate(Parser.ParseExpression("-5")), precision: 10);
    }

    [Fact]
    public void Tokenize_TachKyHieuKhongCanKhoangTrang()
    {
        Assert.Equal(["A", "+", "B", "*", "2"], Parser.Tokenize("A+B*2"));
        Assert.Equal(["(", "A", ")"], Parser.Tokenize("(A)"));
    }

    [Theory]
    [InlineData("A + + B")]
    [InlineData("= 10")]
    [InlineData("A =")]
    [InlineData("( A + B")]
    [InlineData("10 = 20 extra")]
    public void CuPhapSai_ThiNemLoi(string statement)
    {
        Assert.ThrowsAny<Exception>(() => Parser.ParseFact(statement));
    }

    [Fact]
    public void TenBienKhongHopLe_ThiNemLoi()
    {
        Assert.Throws<ParseException>(() => Parser.ParseFact("10 = 20"));
    }
}

public class KnowledgeBaseTests
{
    [Fact]
    public void LuuVaTraCuuSuThat()
    {
        var kb = new KnowledgeBase();
        kb.AssertAll("A = 10", "B = 20");

        Assert.Equal(2, kb.Count);
        Assert.True(kb.TryGet("A", out var fact));
        Assert.Equal("10", fact.Value.ToString());
        Assert.False(kb.Contains("Z"));
    }

    [Fact]
    public void KhaiBaoLai_ThiGhiDe()
    {
        var kb = new KnowledgeBase();
        kb.Assert("A = 10");
        kb.Assert("A = 99");

        Assert.Equal(1, kb.Count);
        kb.TryGet("A", out var fact);
        Assert.Equal("99", fact.Value.ToString());
    }

    [Fact]
    public void FindRelevant_LayDeQuyMoiBienPhuThuoc()
    {
        var kb = new KnowledgeBase();
        kb.AssertAll("A = 10", "B = 20", "C = A + B", "D = C * 2", "Unrelated = 999");

        var relevant = kb.FindRelevant(Parser.ParseExpression("D"));

        Assert.Equal(["D", "C", "A", "B"], relevant.Select(f => f.Name).ToArray());
        Assert.DoesNotContain(relevant, f => f.Name == "Unrelated");
    }

    [Fact]
    public void FindRelevant_KhongLapLaiKhiDungChung()
    {
        var kb = new KnowledgeBase();
        kb.AssertAll("A = 1", "B = A + A", "C = B + A");

        var relevant = kb.FindRelevant(Parser.ParseExpression("C"));

        Assert.Equal(relevant.Count, relevant.Select(f => f.Name).Distinct().Count());
    }

    [Fact]
    public void RemoveVaClear()
    {
        var kb = new KnowledgeBase();
        kb.AssertAll("A = 1", "B = 2");

        Assert.True(kb.Remove("A"));
        Assert.False(kb.Remove("A"));
        Assert.Equal(1, kb.Count);

        kb.Clear();
        Assert.Equal(0, kb.Count);
    }
}

public class ReasoningEngineTests
{
    private static ReasoningEngine WithBasicFacts()
    {
        var engine = new ReasoningEngine();
        engine.LearnAll("A = 10", "B = 20", "C = A + B");
        return engine;
    }

    /// <summary>Đúng ví dụ của đề bài.</summary>
    [Fact]
    public void ViDuCuaDeBai_TraLoiDung()
    {
        Assert.Equal(30.0, WithBasicFacts().Ask("C bằng bao nhiêu?").Value);
    }

    [Theory]
    [InlineData("A + B", 30)]
    [InlineData("A - B", -10)]
    [InlineData("A * B", 200)]
    [InlineData("A / B", 0.5)]
    [InlineData("B / A", 2)]
    public void CacPhepToanSoHoc(string question, double expected)
    {
        Assert.Equal(expected, WithBasicFacts().Ask(question).Value!.Value, precision: 10);
    }

    [Theory]
    [InlineData("A = B", false)]
    [InlineData("A > B", false)]
    [InlineData("A < B", true)]
    [InlineData("B > A", true)]
    [InlineData("C = 30", true)]
    public void CacPhepSoSanh(string question, bool expected)
    {
        var answer = WithBasicFacts().Ask(question);

        Assert.True(answer.IsComparison);
        Assert.Equal(expected, answer.Truth);
        Assert.Equal(expected ? "ĐÚNG" : "SAI", answer.Text);
    }

    [Fact]
    public void ThuTuUuTien_DungTrongLucTinh()
    {
        var engine = WithBasicFacts();

        Assert.Equal(14.0, engine.Ask("2 + 3 * 4").Value);
        Assert.Equal(20.0, engine.Ask("( 2 + 3 ) * 4").Value);
        Assert.Equal(50.0, engine.Ask("A + B * 2").Value);
        Assert.Equal(10.0, engine.Ask("( A + B ) / 3").Value);
    }

    /// <summary>Chuỗi phụ thuộc nhiều tầng phải tự lần ra được.</summary>
    [Fact]
    public void SuyLuanNhieuBuoc()
    {
        var engine = new ReasoningEngine();
        engine.LearnAll(
            "Price = 250", "Quantity = 4",
            "Subtotal = Price * Quantity",
            "TaxRate = 10",
            "Tax = Subtotal * TaxRate / 100",
            "Total = Subtotal + Tax");

        Assert.Equal(1000.0, engine.Ask("Subtotal").Value);
        Assert.Equal(100.0, engine.Ask("Tax").Value);
        Assert.Equal(1100.0, engine.Ask("Total").Value);
    }

    /// <summary>Đổi một sự thật thì mọi kết quả phụ thuộc đổi theo ngay.</summary>
    [Fact]
    public void DoiSuThat_KetQuaCapNhatNgay()
    {
        var engine = WithBasicFacts();
        Assert.Equal(30.0, engine.Ask("C").Value);

        engine.Learn("A = 100");
        Assert.Equal(120.0, engine.Ask("C").Value);
    }

    [Fact]
    public void GiaiThichDuocTungBuoc()
    {
        var answer = WithBasicFacts().Ask("C");

        Assert.NotEmpty(answer.Steps);
        Assert.Contains(answer.Steps, s => s.Description.Contains("C = A + B"));
        Assert.Contains(answer.Steps, s => s.Description.Contains("A = 10"));
        Assert.Contains(answer.Steps, s => s.Description.Contains("30"));
    }

    [Fact]
    public void ChiLietKeCacSuThatThucSuDungDen()
    {
        var engine = WithBasicFacts();
        engine.Learn("Unrelated = 999");

        var answer = engine.Ask("C");

        Assert.Equal(["C", "A", "B"], answer.UsedFacts.Select(f => f.Name).ToArray());
    }

    [Fact]
    public void KhongLapLaiBuocGiaiThich()
    {
        var engine = new ReasoningEngine();
        engine.LearnAll("A = 2", "B = A * 3", "C = B + A", "D = B + C");

        var answer = engine.Ask("D");
        var descriptions = answer.Steps.Select(s => s.Description).ToArray();

        Assert.Equal(descriptions.Length, descriptions.Distinct().Count());
    }

    [Fact]
    public void BienChuaBiet_ThiTuChoi()
    {
        var ex = Assert.Throws<ReasoningException>(() => WithBasicFacts().Ask("Z"));

        Assert.Contains("Z", ex.Message);
    }

    [Fact]
    public void ChiaChoKhong_ThiTuChoi()
    {
        var engine = new ReasoningEngine();
        engine.LearnAll("A = 10", "B = 0");

        Assert.Throws<ReasoningException>(() => engine.Ask("A / B"));
    }

    [Fact]
    public void ThamChieuVongTron_ThiTuChoi()
    {
        var engine = new ReasoningEngine();
        engine.LearnAll("X = Y + 1", "Y = X + 1");

        var ex = Assert.Throws<ReasoningException>(() => engine.Ask("X"));

        Assert.Contains("vòng tròn", ex.Message);
    }

    [Fact]
    public void TuThamChieu_ThiTuChoi()
    {
        var engine = new ReasoningEngine();
        engine.Learn("X = X + 1");

        Assert.Throws<ReasoningException>(() => engine.Ask("X"));
    }

    [Fact]
    public void ValueOf_TruyCapNhanh()
    {
        Assert.Equal(30.0, WithBasicFacts().ValueOf("C"), precision: 10);
    }

    [Fact]
    public void CauHoiRong_ThiTuChoi()
    {
        Assert.ThrowsAny<Exception>(() => WithBasicFacts().Ask("   "));
    }
}

public class QuestionNormalizerTests
{
    [Theory]
    [InlineData("C bằng bao nhiêu?", "C")]
    [InlineData("Cho biết giá trị của C", "C")]
    [InlineData("Hãy tính A + B", "A + B")]
    [InlineData("what is C", "C")]
    [InlineData("A plus B", "A + B")]
    [InlineData("C divided by 3", "C / 3")]
    [InlineData("A có lớn hơn B không?", "A > B")]
    [InlineData("A nhỏ hơn B", "A < B")]
    [InlineData("A + B", "A + B")]
    public void GotVeBieuThucThuan(string question, string expected)
    {
        Assert.Equal(expected, QuestionNormalizer.Normalize(question));
    }

    [Fact]
    public void KhongCatNhamTuNamTrongTenBien()
    {
        // "Khong_Gian" chứa chữ "khong" nhưng không phải từ nhiễu riêng lẻ.
        Assert.Equal("Khong_Gian", QuestionNormalizer.Normalize("Khong_Gian"));
    }

    [Fact]
    public void BoDauCauCuoiCau()
    {
        Assert.Equal("A", QuestionNormalizer.Normalize("A?"));
        Assert.Equal("A", QuestionNormalizer.Normalize("A."));
    }

    /// <summary>
    /// Giới hạn thật của cách làm này: chỉ nhận ra đúng những mẫu đã liệt kê.
    /// Một cách diễn đạt mới là chịu. Đây chính là chỗ LLM bù vào ở Phase 11.
    /// </summary>
    [Fact]
    public void CachDienDatLa_ThiKhongHieu()
    {
        string normalized = QuestionNormalizer.Normalize("cậu tính hộ tớ tổng của A với B nhé");

        Assert.NotEqual("A + B", normalized);
    }
}
