using Ai.Phase10.Memory;

namespace Ai.Phase10.Tests;

public abstract class MemoryTestBase
{
    protected static readonly DateTimeOffset Start = new(2026, 1, 1, 9, 0, 0, TimeSpan.Zero);

    protected static (MemoryStore Store, ManualClock Clock) Build(int shortTermCapacity = 5)
    {
        var clock = new ManualClock(Start);
        return (new MemoryStore(clock, shortTermCapacity), clock);
    }
}

public class KeywordsTests
{
    [Fact]
    public void Extract_BoTuDungVaDauCau()
    {
        var keywords = Keywords.Extract("The user is working with Azure!");

        Assert.Contains("user", keywords);
        Assert.Contains("working", keywords);
        Assert.Contains("azure", keywords);
        Assert.DoesNotContain("the", keywords);
        Assert.DoesNotContain("is", keywords);
        Assert.DoesNotContain("with", keywords);
    }

    [Fact]
    public void Extract_GiuNguyenTenNgonNgu()
    {
        // "C#" và "C++" phải sống sót qua bước tách từ.
        Assert.Contains("c#", Keywords.Extract("User uses C#"));
        Assert.Contains("c++", Keywords.Extract("User knows C++"));
    }

    [Fact]
    public void Extract_TuDemChoRaTapRong()
    {
        Assert.Empty(Keywords.Extract("ok"));
        Assert.Empty(Keywords.Extract("haha thanks"));
        Assert.Empty(Keywords.Extract("!!! ???"));
        Assert.Empty(Keywords.Extract(""));
    }

    [Theory]
    [InlineData("Gundam", "User likes Gundam", 1.0)]
    [InlineData("Gundam Azure", "User likes Gundam", 0.5)]
    [InlineData("cooking", "User likes Gundam", 0.0)]
    public void Relevance_LaTiLeTuKhoaCuaCauHoiDuocTimThay(string query, string content, double expected)
    {
        double relevance = Keywords.Relevance(Keywords.Extract(query), Keywords.Extract(content));

        Assert.Equal(expected, relevance, precision: 10);
    }

    [Fact]
    public void Canonical_BatBienVoiThuTuTu()
    {
        Assert.Equal(Keywords.Canonical("User uses C#"), Keywords.Canonical("uses C# User"));
        Assert.Equal(Keywords.Canonical("User uses C#"), Keywords.Canonical("  the USER uses c#!  "));
        Assert.NotEqual(Keywords.Canonical("User uses C#"), Keywords.Canonical("User uses Java"));
    }
}

public class RankerTests : MemoryTestBase
{
    private static MemoryFact MakeFact(string content, double importance, DateTimeOffset created) =>
        new()
        {
            Id = "x",
            Content = content,
            Keywords = Keywords.Extract(content),
            Importance = importance,
            CreatedAt = created,
            LastAccessedAt = created,
        };

    [Fact]
    public void Recency_GiamMotNuaSauMotChuKyBanRa()
    {
        var ranker = new Ranker(recencyHalfLifeHours: 24);
        var fact = MakeFact("User uses C#", 0.5, Start);

        Assert.Equal(1.0, ranker.Recency(fact, Start), precision: 10);
        Assert.Equal(0.5, ranker.Recency(fact, Start.AddHours(24)), precision: 10);
        Assert.Equal(0.25, ranker.Recency(fact, Start.AddHours(48)), precision: 10);
    }

    [Fact]
    public void Frequency_TangTheoLogVaBiChanTren()
    {
        var fact = MakeFact("x y", 0.5, Start);

        Assert.Equal(0.0, Ranker.Frequency(fact), precision: 10);

        fact.AccessCount = 10;
        Assert.Equal(1.0, Ranker.Frequency(fact), precision: 6);

        fact.AccessCount = 1000;
        Assert.Equal(1.0, Ranker.Frequency(fact), precision: 10);   // bị chặn, không nuốt mọi thứ
    }

    [Fact]
    public void KhongLienQuan_ThiDiemBangKhongDuQuanTrongDenDau()
    {
        var ranker = new Ranker();
        var fact = MakeFact("User likes Gundam", importance: 1.0, Start);

        var scored = ranker.Score(fact, Keywords.Extract("cooking recipes"), Start);

        Assert.Equal(0.0, scored.Score);
        Assert.Equal(0.0, scored.Relevance);
    }

    [Fact]
    public void CungLienQuan_ThiQuanTrongHonThangCuoc()
    {
        var ranker = new Ranker();
        var important = MakeFact("User uses C#", 0.9, Start);
        var trivial = MakeFact("User tried C#", 0.3, Start);

        var query = Keywords.Extract("C#");

        Assert.True(ranker.Score(important, query, Start).Score > ranker.Score(trivial, query, Start).Score);
    }

    [Fact]
    public void CungLienQuan_ThiGanDayHonThangCuoc()
    {
        var ranker = new Ranker(recencyHalfLifeHours: 24);
        var fresh = MakeFact("User uses C#", 0.5, Start);
        var stale = MakeFact("User loves C#", 0.5, Start.AddDays(-30));

        var query = Keywords.Extract("C#");

        Assert.True(ranker.Score(fresh, query, Start).Score > ranker.Score(stale, query, Start).Score);
    }

    [Fact]
    public void RetentionScore_KhongDungDenCauHoi()
    {
        var ranker = new Ranker();
        var important = MakeFact("anything here", 0.9, Start);
        var trivial = MakeFact("anything here", 0.1, Start);

        Assert.True(ranker.RetentionScore(important, Start) > ranker.RetentionScore(trivial, Start));
    }
}

public class ShortTermMemoryTests : MemoryTestBase
{
    private static MemoryFact MakeFact(string id) => new()
    {
        Id = id,
        Content = id,
        Keywords = Keywords.Extract(id),
        Importance = 0.5,
        CreatedAt = Start,
        LastAccessedAt = Start,
    };

    [Fact]
    public void GiuThuTuMoiNhatTruoc()
    {
        var stm = new ShortTermMemory(3);
        stm.Add(MakeFact("alpha"));
        stm.Add(MakeFact("beta"));

        Assert.Equal(["beta", "alpha"], stm.Items.Select(f => f.Id).ToArray());
    }

    [Fact]
    public void VuotSucChua_ThiDayMauCuNhatRa()
    {
        var stm = new ShortTermMemory(2);
        stm.Add(MakeFact("a"));
        stm.Add(MakeFact("b"));

        var evicted = stm.Add(MakeFact("c"));

        Assert.Equal("a", evicted?.Id);
        Assert.Equal(2, stm.Count);
        Assert.False(stm.Contains("a"));
        Assert.True(stm.Contains("c"));
    }

    [Fact]
    public void ChuaDay_ThiKhongDayAiRa()
    {
        var stm = new ShortTermMemory(3);

        Assert.Null(stm.Add(MakeFact("a")));
        Assert.False(stm.IsFull);

        stm.Add(MakeFact("b"));
        stm.Add(MakeFact("c"));
        Assert.True(stm.IsFull);
    }

    [Fact]
    public void SucChuaKhongDuong_ThiNemLoi()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ShortTermMemory(0));
    }
}

public class MemoryStoreTests : MemoryTestBase
{
    [Fact]
    public void StoreVaRetrieve_ViDuCuaDeBai()
    {
        var (store, _) = Build();

        store.Store("User likes Gundam", 0.6);
        store.Store("User uses C#", 0.9);
        store.Store("User works with Azure", 0.8);

        Assert.Equal(3, store.LongTerm.Count);
        Assert.Equal("User likes Gundam", store.Retrieve("Gundam")?.Content);
        Assert.Equal("User uses C#", store.Retrieve("C#")?.Content);
        Assert.Equal("User works with Azure", store.Retrieve("Azure")?.Content);
    }

    [Fact]
    public void KhongCoGiLienQuan_ThiTraVeNullChuKhongBia()
    {
        var (store, _) = Build();
        store.Store("User uses C#", 0.9);

        Assert.Null(store.Retrieve("cooking recipes"));
        Assert.Empty(store.Search("cooking recipes"));
    }

    [Fact]
    public void TuDem_ThiTuChoiLuu()
    {
        var (store, _) = Build();

        var result = store.Store("ok", 0.9);

        Assert.Equal(StoreOutcome.Rejected, result.Outcome);
        Assert.Equal(0, store.LongTerm.Count);
    }

    [Fact]
    public void ImportanceThap_ThiTuChoiLuu()
    {
        var (store, _) = Build();

        var result = store.Store("User mentioned the weather is nice", 0.1);

        Assert.Equal(StoreOutcome.Rejected, result.Outcome);
        Assert.Contains("dưới ngưỡng", result.Reason);
    }

    [Fact]
    public void TrungLap_ThiCungCoChuKhongTaoBanSao()
    {
        var (store, _) = Build();
        store.Store("User uses C#", 0.9);

        var result = store.Store("User uses C#", 0.9);

        Assert.Equal(StoreOutcome.Reinforced, result.Outcome);
        Assert.Equal(1, store.LongTerm.Count);
        Assert.Equal(1, result.Fact!.AccessCount);
        Assert.True(result.Fact.Importance > 0.9);
    }

    [Fact]
    public void TrungLapDuDaoThuTuTu()
    {
        var (store, _) = Build();
        store.Store("User uses C#", 0.9);

        Assert.Equal(StoreOutcome.Reinforced, store.Store("uses C# User", 0.9).Outcome);
        Assert.Equal(StoreOutcome.Reinforced, store.Store("the USER uses c#", 0.9).Outcome);
        Assert.Equal(1, store.LongTerm.Count);
    }

    [Fact]
    public void ImportanceKhongVuotQua1()
    {
        var (store, _) = Build();
        store.Store("User uses C#", 0.95);

        for (int i = 0; i < 10; i++) store.Store("User uses C#", 0.95);

        Assert.True(store.LongTerm.Facts.Single().Importance <= 1.0);
    }

    [Fact]
    public void Search_XepTheoDiemGiamDan()
    {
        var (store, _) = Build();
        store.Store("User uses C# and dotnet", 0.9);
        store.Store("User once tried C# scripting", 0.35);

        var results = store.Search("C#", take: 5);

        Assert.Equal(2, results.Count);
        Assert.True(results[0].Score >= results[1].Score);
        Assert.Contains("dotnet", results[0].Fact.Content);
    }

    [Fact]
    public void Search_KhongDanhDauDaDung_NhungRetrieveThiCo()
    {
        var (store, _) = Build();
        store.Store("User uses C#", 0.9);

        store.Search("C#");
        Assert.Equal(0, store.LongTerm.Facts.Single().AccessCount);

        store.Retrieve("C#");
        Assert.Equal(1, store.LongTerm.Facts.Single().AccessCount);
    }

    [Fact]
    public void Rank_ChamDiemToanBoKeCaThuKhongLienQuan()
    {
        var (store, _) = Build();
        store.Store("User uses C#", 0.9);
        store.Store("User likes Gundam", 0.6);

        var ranked = store.Rank("C#");

        Assert.Equal(2, ranked.Count);
        Assert.True(ranked[0].Score > 0);
        Assert.Equal(0.0, ranked[^1].Score);
    }

    [Fact]
    public void NgamHanTroiDi_NhungDaiHanVanNho()
    {
        var (store, _) = Build(shortTermCapacity: 2);

        store.Store("User uses C#", 0.9);
        store.Store("User works with Azure", 0.8);
        store.Store("User likes Gundam", 0.7);

        Assert.Equal(2, store.ShortTerm.Count);
        Assert.False(store.ShortTerm.Contains(store.LongTerm.Facts.First(f => f.Content.Contains("C#")).Id));
        Assert.Equal("User uses C#", store.Retrieve("C#")?.Content);
    }

    [Fact]
    public void MauBiDayRaMaKhongDuGiaTri_ThiQuenHan()
    {
        var clock = new ManualClock(Start);
        var store = new MemoryStore(clock, shortTermCapacity: 1)
        {
            MinimumImportance = 0.0,
            PromotionThreshold = 0.9,   // rất khó được chuyển sang dài hạn
        };

        store.Store("User tried something trivial", 0.1);
        store.Store("User uses C#", 0.95);

        Assert.Equal(1, store.LongTerm.Count);
        Assert.Equal("User uses C#", store.LongTerm.Facts.Single().Content);
    }

    [Fact]
    public void Forget_XoaKhoiCaHaiTang()
    {
        var (store, _) = Build();
        var fact = store.Store("User likes Gundam", 0.6).Fact!;

        Assert.True(store.Forget(fact.Id));
        Assert.False(store.Forget(fact.Id));
        Assert.Null(store.Retrieve("Gundam"));
        Assert.False(store.ShortTerm.Contains(fact.Id));
    }

    /// <summary>Ký ức ít giá trị phải phai đi theo thời gian; ký ức hay dùng thì không.</summary>
    [Fact]
    public void ForgetWeak_ChiQuenThuItGiaTri()
    {
        var (store, clock) = Build(shortTermCapacity: 10);

        store.Store("User uses C# every day", 0.95);
        store.Store("User works with Azure", 0.80);
        store.Store("User tried a new cafe once", 0.35);
        store.Store("User mentioned a random link", 0.30);

        clock.Advance(TimeSpan.FromDays(30));
        store.Retrieve("C#");
        store.Retrieve("Azure");

        var forgotten = store.ForgetWeak(threshold: 0.25);

        Assert.Equal(2, forgotten.Count);
        Assert.All(forgotten, f => Assert.DoesNotContain("C#", f.Content));
        Assert.Equal(2, store.LongTerm.Count);
        Assert.NotNull(store.Retrieve("C#"));
    }

    [Fact]
    public void ForgetWeak_KhongQuenGiKhiMoiThuConTuoi()
    {
        var (store, _) = Build(shortTermCapacity: 10);
        store.Store("User uses C#", 0.9);
        store.Store("User works with Azure", 0.8);

        Assert.Empty(store.ForgetWeak(threshold: 0.25));
        Assert.Equal(2, store.LongTerm.Count);
    }

    [Fact]
    public void DungKyUcThuongXuyen_ThiGiuDuocLau()
    {
        var (store, clock) = Build(shortTermCapacity: 10);
        store.Store("User uses Kubernetes", 0.4);
        store.Store("User uses Terraform", 0.4);

        // Nhắc lại Kubernetes đều đặn, bỏ quên Terraform.
        for (int day = 0; day < 10; day++)
        {
            clock.Advance(TimeSpan.FromDays(3));
            store.Retrieve("Kubernetes");
        }

        var kubernetes = store.LongTerm.Facts.First(f => f.Content.Contains("Kubernetes"));
        var terraform = store.LongTerm.Facts.First(f => f.Content.Contains("Terraform"));

        Assert.True(store.RetentionScore(kubernetes) > store.RetentionScore(terraform));

        store.ForgetWeak(threshold: 0.25);
        Assert.NotNull(store.Retrieve("Kubernetes"));
        Assert.Null(store.Retrieve("Terraform"));
    }

    [Fact]
    public void Clear_XoaSach()
    {
        var (store, _) = Build();
        store.Store("User uses C#", 0.9);

        store.Clear();

        Assert.Equal(0, store.LongTerm.Count);
        Assert.Equal(0, store.ShortTerm.Count);
    }
}
