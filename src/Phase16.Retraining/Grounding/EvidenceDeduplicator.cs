using Ai.Phase16.Internet;
using Ai.Phase16.Memory;

namespace Ai.Phase16.Grounding;

/// <summary>
/// Gộp evidence trùng lặp — và đây là bước khó hơn bỏ trùng của Phase 13.
///
/// Phase 12 bỏ trùng theo URL: hai dòng cùng URL là cùng một trang, xong.
/// Ở Phase 13 thì các mẩu đến từ NĂM kênh khác nhau, nên cùng một thông tin
/// xuất hiện dưới những hình thức chẳng giống nhau chút nào:
///
///   Knowledge: "Softmax biến một vector logits thành phân bố xác suất."
///   Internet : "Softmax converts logits into a probability distribution."
///   Memory   : "tôi đã đọc về softmax và phân bố xác suất"
///
/// VÌ SAO PHẢI GỘP? Không phải cho gọn. Mà vì <see cref="Evidence.Agreement"/>
/// là con số Phase 14 dùng để tính độ tự tin. Ba mẩu trên KHÔNG phải ba nguồn
/// độc lập cùng xác nhận — nhưng nếu không gộp thì hệ thống đếm ra 3, rồi kết
/// luận "rất nhiều nguồn đồng thuận". Đếm sai chỗ này dẫn thẳng tới tự tin sai.
///
/// GIỮ MẨU NÀO khi gộp? Mẩu có mức TIN CẬY cao nhất, không phải mẩu dài nhất
/// hay đến trước nhất. Nếu knowledge và internet nói cùng một điều thì phải
/// trích dẫn knowledge — nó kiểm chứng được, và nó không mang theo nghĩa vụ
/// phải cảnh báo người dùng.
/// </summary>
public sealed class EvidenceDeduplicator
{
    /// <summary>Giống nhau từ mức này trở lên thì coi là cùng một thông tin.</summary>
    public double NearDuplicateThreshold { get; init; } = 0.70;

    public EvidenceDeduplicationReport Deduplicate(IEnumerable<Evidence> evidence)
    {
        ArgumentNullException.ThrowIfNull(evidence);

        var groups = new List<Group>();
        int removed = 0;

        foreach (var item in evidence)
        {
            var group = FindGroup(groups, item);

            if (group is null)
            {
                groups.Add(new Group(item));
                continue;
            }

            group.Absorb(item);
            removed++;
        }

        var merged = groups.Select(g => g.Best with
        {
            // Agreement = số KÊNH độc lập, không phải số mẩu. Hai mẩu cùng từ
            // Internet chỉ là một tiếng nói; internet + knowledge mới là hai.
            Agreement = g.DistinctChannels,
        }).ToList();

        return new EvidenceDeduplicationReport(merged, removed);
    }

    private Group? FindGroup(List<Group> groups, Evidence item)
    {
        foreach (var group in groups)
        {
            // ----------------------------------------------------------
            // CHẶN TRƯỚC: chiều khẳng định/phủ định ngược nhau -> KHÔNG gộp
            // ----------------------------------------------------------
            //
            // Đây là bản sửa cho một lỗi thật, và lỗi đó im lặng đến mức đáng
            // sợ. Hai câu này có tập từ khoá gần như y hệt (vì "không" là stop
            // word của Phase 10, bị lọc mất):
            //
            //     "SSRF là lỗ hổng cho phép gửi request nội bộ"
            //     "SSRF KHÔNG phải lỗ hổng cho phép gửi request nội bộ"
            //
            // Nên Jaccard của chúng vượt ngưỡng gần-trùng, và bộ bỏ trùng GỘP
            // chúng lại — làm một mâu thuẫn trực tiếp biến mất hoàn toàn
            // trước khi ContradictionDetector của Phase 14 kịp nhìn thấy.
            // Gộp xong thì Agreement còn tăng lên 2, tức là hệ thống kết luận
            // "hai kênh cùng xác nhận" về đúng chỗ chúng phủ định nhau.
            //
            // Cặp từ đối nghịch ("tăng"/"giảm") bị gộp y như vậy, và còn dễ
            // hơn: chúng chỉ khác nhau ĐÚNG MỘT TỪ trong cả câu.
            if (Polarity.Conflict(group.Best.Content, item.Content)) continue;

            // --- Mức 1: cùng URL chuẩn hoá -> chắc chắn cùng một trang ---
            if (item.Url is not null && group.Best.Url is not null &&
                ResultDeduplicator.CanonicalUrl(item.Url)
                    .Equals(ResultDeduplicator.CanonicalUrl(group.Best.Url), StringComparison.Ordinal))
            {
                return group;
            }

            // --- Mức 2: cùng nội dung sau khi chuẩn hoá ---
            if (group.Canonical.Equals(Keywords.Canonical(item.Content), StringComparison.OrdinalIgnoreCase))
            {
                return group;
            }

            // --- Mức 3: gần trùng theo từ khoá ---
            if (ResultDeduplicator.Jaccard(group.Keywords, item.Keywords()) >= NearDuplicateThreshold)
            {
                return group;
            }
        }

        return null;
    }

    private sealed class Group
    {
        private readonly HashSet<SourceKind> _channels;

        public Group(Evidence first)
        {
            Best = first;
            Canonical = Memory.Keywords.Canonical(first.Content);
            Keywords = first.Keywords();
            _channels = [first.Kind];
        }

        public Evidence Best { get; private set; }

        public string Canonical { get; private set; }

        public HashSet<string> Keywords { get; }

        public int DistinctChannels => _channels.Count;

        public void Absorb(Evidence other)
        {
            _channels.Add(other.Kind);
            Keywords.UnionWith(other.Keywords());

            // Ưu tiên mức tin cậy (enum nhỏ = tin hơn), rồi tới Reliability,
            // rồi tới nội dung dài hơn.
            bool better = other.Trust < Best.Trust
                || (other.Trust == Best.Trust && other.Reliability > Best.Reliability)
                || (other.Trust == Best.Trust && other.Reliability.Equals(Best.Reliability)
                    && other.Content.Length > Best.Content.Length);

            if (better)
            {
                Best = other;
                Canonical = Memory.Keywords.Canonical(other.Content);
            }
        }
    }
}

public sealed record EvidenceDeduplicationReport(IReadOnlyList<Evidence> Unique, int DuplicatesRemoved);
