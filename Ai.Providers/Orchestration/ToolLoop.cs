using Ai.Providers.Contracts;
using Ai.Providers.Routing;
using Ai.Providers.Tools;

namespace Ai.Providers.Orchestration;

/// <summary>Kết quả chạy hết vòng lặp công cụ: câu trả lời cuối, kèm mọi lượt đã chạy.</summary>
public sealed record ToolLoopResult(AiResponse Response, IReadOnlyList<AiToolExchange> Exchanges)
{
    /// <summary>Vòng lặp dừng vì CHẠM TRẦN chứ không phải vì model đã nói xong.</summary>
    public bool HitLimit { get; init; }

    public int ToolCallCount => Exchanges.Count;

    public int RefusedCount => Exchanges.Count(e => e.Result.WasRefused);

    public override string ToString() =>
        $"{Response} — {ToolCallCount} lượt công cụ" + (HitLimit ? " (CHẠM TRẦN)" : "");
}

/// <summary>
/// VÒNG LẶP GỌI CÔNG CỤ — model xin, MiniAI quyết, model nhận kết quả.
///
///     model -> xin công cụ -> ToolRegistry (4 chốt) -> chạy -> kết quả
///        ↑                                                       │
///        └───────────────────────────────────────────────────────┘
///                     lặp, tối đa MaxToolCalls lần
///
/// BA ĐIỀU KHIẾN VÒNG LẶP NÀY KHÔNG CHẠY MÃI, và cả ba đều cần:
///
///   1. TRẦN SỐ LƯỢT (<c>MaxToolCalls</c>). Chạm trần thì hỏi model lần cuối
///      KHÔNG kèm công cụ nào — nên nó buộc phải trả lời bằng chữ.
///   2. MỖI LƯỢT ĐỀU CÓ KẾT QUẢ GỬI VỀ, kể cả lượt bị từ chối. Model đang chờ
///      một kết quả mà không nhận được sẽ xin lại y hệt.
///   3. HUỶ ĐƯỢC. Token huỷ đi xuyên suốt, nên màn hình đóng là vòng lặp dừng.
///
/// Vòng lặp KHÔNG giữ trạng thái ở provider: mỗi lượt gửi lại toàn bộ
/// <see cref="AiRequest.ToolHistory"/>. Nhờ vậy một provider hỏng giữa chừng
/// có thể được thay bằng provider dự phòng mà không mất ngữ cảnh — điều không
/// làm được nếu trạng thái nằm bên kia.
/// </summary>
public sealed class ToolLoop
{
    private readonly IAiRouter _router;
    private readonly IToolRegistry _registry;

    public ToolLoop(IAiRouter router, IToolRegistry registry)
    {
        ArgumentNullException.ThrowIfNull(router);
        ArgumentNullException.ThrowIfNull(registry);

        _router = router;
        _registry = registry;
    }

    public async Task<ToolLoopResult> RunAsync(AiRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        // BẮT ĐẦU TỪ NHỮNG LƯỢT ĐÃ CHẠY, không bắt đầu từ rỗng.
        //
        // Người gọi có thể đưa vào một yêu cầu đã mang sẵn kết quả công cụ —
        // đúng điều bộ điều phối làm khi nó VIẾT LẠI câu trả lời. Bỏ qua
        // chúng ở đây thì lần viết lại sẽ bảo model "hãy dùng kết quả công cụ"
        // mà không đưa kết quả nào cho nó, và bộ kiểm cũng mất luôn thứ nó cần
        // đối chiếu.
        var exchanges = new List<AiToolExchange>(request.ToolHistory);

        int budget = Math.Max(0, request.MaxToolCalls) + exchanges.Count;

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            bool exhausted = exchanges.Count >= budget;

            // Chạm trần -> bỏ hẳn danh sách công cụ khỏi yêu cầu. Gửi kèm công
            // cụ rồi mong model tự kiềm chế là mong hão; không gửi thì nó không
            // có cách nào xin nữa.
            var attempt = request with
            {
                ToolHistory = exchanges,
                Tools = exhausted ? [] : request.Tools,
            };

            var routed = await _router.GenerateAsync(attempt, cancellationToken).ConfigureAwait(false);

            if (!routed.Response.Success)
            {
                return new ToolLoopResult(routed.Response, exchanges) { HitLimit = exhausted };
            }

            if (!routed.Response.WantsTools)
            {
                return new ToolLoopResult(routed.Response, exchanges) { HitLimit = exhausted };
            }

            if (exhausted)
            {
                // Model vẫn xin công cụ dù không được đưa công cụ nào. Không
                // chạy, và trả về đúng những gì nó nói — thà một câu trả lời
                // thiếu còn hơn một vòng lặp không có điểm dừng.
                return new ToolLoopResult(routed.Response, exchanges) { HitLimit = true };
            }

            foreach (var call in routed.Response.ToolCalls)
            {
                if (exchanges.Count >= budget) break;

                var result = await _registry.ExecuteAsync(call, cancellationToken).ConfigureAwait(false);

                exchanges.Add(new AiToolExchange(call, result));
            }
        }
    }
}
