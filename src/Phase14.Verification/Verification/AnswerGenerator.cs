using System.Text;
using Ai.Phase14.Grounding;
using Ai.Phase14.Memory;

namespace Ai.Phase14.Verification;

public interface IAnswerGenerator
{
    /// <summary>Rút các khẳng định từ tập căn cứ. KHÔNG sinh chữ mới.</summary>
    IReadOnlyList<Claim> ExtractClaims(EvidenceSet evidence, IReadOnlyList<Evidence> cited);

    /// <summary>Ghép thành văn bản câu trả lời cho một kết luận đã chốt.</summary>
    string Compose(string question, AnswerVerdict verdict, double confidence,
        IReadOnlyList<Claim> claims, IReadOnlyList<Evidence> cited,
        IReadOnlyList<Contradiction> contradictions, bool requiresDisclosure);
}

/// <summary>
/// SINH CÂU TRẢ LỜI TỪ CĂN CỨ — theo khuôn, tất định, không có mô hình sinh chữ.
///
/// Đây là lựa chọn có chủ ý, không phải sự thiếu sót. Cả project này có sẵn
/// một mô hình ngôn ngữ (Phase 6) và một transformer (Phase 8) — nhưng dùng
/// chúng để viết câu trả lời sẽ phá đúng cái mà Phase 14 đang cố dựng: một mô
/// hình sinh chữ có thể diễn đạt lại căn cứ thành một câu mà căn cứ KHÔNG hề
/// nói, và không có cách nào biết nó đã làm thế.
///
/// Nên nguyên tắc ở đây là: MỖI CÂU TRONG CÂU TRẢ LỜI ĐẾN TỪ MỘT MẨU CĂN CỨ.
/// Phần agent tự viết chỉ là khung đỡ — câu dẫn, trích dẫn nguồn, lời cảnh
/// báo. Đổi lại là văn phong cứng; nhưng thà đọc cứng mà đúng nguồn.
///
/// FACT vs INFERENCE được quyết ở đây:
///
///     một mẩu căn cứ            -> Fact      (nhắc lại)
///     nhiều mẩu, hoặc Derived   -> Inference (ghép hoặc suy ra)
/// </summary>
public sealed class AnswerGenerator : IAnswerGenerator
{
    /// <summary>Tối đa bao nhiêu khẳng định trong một câu trả lời.</summary>
    public int MaxClaims { get; init; } = 3;

    public IReadOnlyList<Claim> ExtractClaims(EvidenceSet evidence, IReadOnlyList<Evidence> cited)
    {
        ArgumentNullException.ThrowIfNull(evidence);
        ArgumentNullException.ThrowIfNull(cited);

        var claims = new List<Claim>();
        int index = 1;

        foreach (var item in cited.Take(MaxClaims))
        {
            // Nội dung căn cứ được dùng NGUYÊN VĂN làm nội dung khẳng định.
            // Đây là lý do Support luôn cao: không có bước diễn đạt lại nào để
            // làm trôi nghĩa. Validator vẫn kiểm lại, vì nó không tin chỗ này.
            claims.Add(new Claim
            {
                Id = $"c{index++}",
                Text = item.Content,
                Kind = item.Kind == SourceKind.Reasoning ? ClaimKind.Inference : ClaimKind.Fact,
                EvidenceIds = [item.Id],
                Support = Keywords.Relevance(Keywords.Extract(item.Content), item.Keywords()),
                Reliability = item.Reliability,
                Agreement = item.Agreement,
                HasVerifiedSupport = item.Trust == TrustLevel.Verified,
            });
        }

        return claims;
    }

    public string Compose(string question, AnswerVerdict verdict, double confidence,
        IReadOnlyList<Claim> claims, IReadOnlyList<Evidence> cited,
        IReadOnlyList<Contradiction> contradictions, bool requiresDisclosure)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(claims);
        ArgumentNullException.ThrowIfNull(cited);
        ArgumentNullException.ThrowIfNull(contradictions);

        return verdict switch
        {
            AnswerVerdict.Insufficient => ComposeInsufficient(claims),
            AnswerVerdict.Conflicted => ComposeConflicted(contradictions, cited),
            _ => ComposeAnswer(verdict, confidence, claims, cited, contradictions, requiresDisclosure),
        };
    }

    /// <summary>
    /// Không đủ căn cứ -> NÓI THẲNG là không biết.
    ///
    /// Đây là nhánh quan trọng nhất của cả Phase 14. Một hệ thống không có
    /// nhánh này sẽ luôn trả lời một điều gì đó, và điều nó trả lời khi không
    /// biết gì chính là điều nó bịa ra.
    /// </summary>
    private static string ComposeInsufficient(IReadOnlyList<Claim> claims)
    {
        var builder = new StringBuilder();

        builder.Append("Tôi KHÔNG ĐỦ CĂN CỨ để trả lời câu này, nên tôi không trả lời. ");

        if (claims.Count == 0)
        {
            builder.Append("Không kênh nào (kho kiến thức, bộ nhớ, Internet) đưa ra được căn cứ liên quan.");
        }
        else
        {
            builder.Append("Có tìm được ").Append(claims.Count)
                .Append(" mẩu liên quan nhưng chúng quá yếu để kết luận:");

            foreach (var claim in claims)
            {
                builder.AppendLine().Append("  • ").Append(Shorten(claim.Text, 140));
            }
        }

        return builder.ToString();
    }

    /// <summary>
    /// Các căn cứ nói ngược nhau -> TRÌNH BÀY CẢ HAI, không chọn bên.
    ///
    /// Chọn một phía rồi im lặng về phía kia là hành vi tệ nhất: người đọc
    /// nhận một câu trả lời trông dứt khoát, và mất luôn thông tin quan trọng
    /// nhất — rằng chính các nguồn cũng không thống nhất.
    /// </summary>
    private static string ComposeConflicted(IReadOnlyList<Contradiction> contradictions, IReadOnlyList<Evidence> cited)
    {
        var builder = new StringBuilder();

        builder.Append("Các căn cứ tôi tìm được NÓI NGƯỢC NHAU, nên tôi không chọn bên nào. ")
            .AppendLine("Đây là cả hai phía:");

        var conflict = contradictions[0];

        builder.AppendLine()
            .Append("  Phía A — ").Append(conflict.Left.Source.Cite()).AppendLine()
            .Append("    ").AppendLine(Shorten(conflict.Left.Content, 220))
            .AppendLine()
            .Append("  Phía B — ").Append(conflict.Right.Source.Cite()).AppendLine()
            .Append("    ").AppendLine(Shorten(conflict.Right.Content, 220));

        builder.AppendLine()
            .Append("  Dấu hiệu mâu thuẫn: ").Append(conflict.Explanation);

        if (conflict.BetweenVerifiedSources)
        {
            builder.AppendLine().Append("  Đáng lưu ý: cả hai đều là nguồn đã kiểm chứng, nên đây là tranh chấp thật.");
        }

        if (contradictions.Count > 1)
        {
            builder.AppendLine().Append("  (còn ").Append(contradictions.Count - 1).Append(" mâu thuẫn khác nữa)");
        }

        return builder.ToString();
    }

    /// <summary>Có căn cứ đủ -> trả lời, kèm trích dẫn từng mẩu và nhãn Fact/Inference.</summary>
    private static string ComposeAnswer(AnswerVerdict verdict, double confidence,
        IReadOnlyList<Claim> claims, IReadOnlyList<Evidence> cited,
        IReadOnlyList<Contradiction> contradictions, bool requiresDisclosure)
    {
        var builder = new StringBuilder();

        if (verdict == AnswerVerdict.Uncertain)
        {
            builder.Append("Tôi trả lời được nhưng KHÔNG CHẮC (độ tự tin ")
                .Append(confidence.ToString("P0")).AppendLine("), vì căn cứ còn yếu:");
        }
        else
        {
            builder.Append("Theo các căn cứ tôi có (độ tự tin ")
                .Append(confidence.ToString("P0")).AppendLine("):");
        }

        var byId = cited.ToDictionary(e => e.Id, StringComparer.Ordinal);

        foreach (var claim in claims)
        {
            string label = claim.Kind == ClaimKind.Fact ? "SỰ THẬT" : "SUY RA ";

            builder.AppendLine()
                .Append("  [").Append(label).Append("] ").AppendLine(Shorten(claim.Text, 240));

            foreach (string id in claim.EvidenceIds)
            {
                if (!byId.TryGetValue(id, out var evidence)) continue;

                builder.Append("    ↳ ").Append(evidence.Source.Cite())
                    .Append("  [tin cậy ").Append(claim.Reliability.ToString("F2")).Append(']');

                if (claim.Agreement > 1)
                {
                    builder.Append(", ").Append(claim.Agreement).Append(" kênh cùng xác nhận");
                }

                builder.AppendLine();
            }
        }

        // Mâu thuẫn NHẸ (dưới ngưỡng đổi kết luận) vẫn phải được nêu ra.
        //
        // Đây là chỗ dễ bỏ qua nhất và bỏ qua thì thành đúng cái hành vi đã bị
        // phê ở ComposeConflicted: chọn một phía rồi im lặng về phía kia. Mâu
        // thuẫn nhẹ nghĩa là một bên đáng tin hơn hẳn — nên agent được phép
        // trả lời theo bên đó, nhưng KHÔNG được phép giấu rằng có bên khác.
        if (contradictions.Count > 0)
        {
            var weakest = contradictions[0];
            var dissenting = weakest.Left.Reliability <= weakest.Right.Reliability ? weakest.Left : weakest.Right;

            builder.AppendLine()
                .Append("  CÓ NGUỒN NÓI NGƯỢC LẠI (").Append(dissenting.Source.Cite())
                .Append(", tin cậy ").Append(dissenting.Reliability.ToString("F2")).AppendLine("):")
                .Append("    ").AppendLine(Shorten(dissenting.Content, 200))
                .Append("    Tôi không trả lời theo nguồn này vì nó kém tin cậy hơn, nhưng bạn nên biết là có.");
        }

        if (requiresDisclosure)
        {
            builder.AppendLine()
                .Append("  LƯU Ý: toàn bộ căn cứ trên đến từ nguồn ngoài CHƯA KIỂM CHỨNG. ")
                .Append("Hãy tự đối chiếu trước khi dựa vào.");
        }

        return builder.ToString().TrimEnd();
    }

    private static string Shorten(string text, int maxLength) =>
        text.Length <= maxLength ? text : string.Concat(text.AsSpan(0, maxLength).TrimEnd(), "…");
}
