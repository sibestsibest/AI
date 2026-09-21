namespace Ai.Providers.Contracts;

/// <summary>
/// MỘT NHÀ CUNG CẤP MODEL — cửa duy nhất để gọi ra ngoài.
///
/// Giao diện này cố tình NHỎ. Ba thành viên, không hơn:
///
///   GenerateAsync      gửi yêu cầu, nhận câu trả lời đã chuẩn hoá
///   IsAvailableAsync   provider này có dùng được lúc này không
///   ProviderName       tên trong cấu hình và trong log
///   SupportedModels    những model provider này nhận
///
/// VÌ SAO NHỎ ĐẾN VẬY? Vì mỗi thành viên thêm vào là một thứ MỌI provider
/// tương lai phải hiện thực. Thêm <c>StreamAsync</c>, <c>EmbedAsync</c>,
/// <c>CountTokensAsync</c> vào đây thì một provider chỉ làm được chat sẽ phải
/// ném <c>NotSupportedException</c> ở ba chỗ — và chỗ gọi lại phải bọc try/catch
/// để biết provider nào làm được gì. Năng lực khác nhau thì khai báo bằng cấu
/// hình (xem <c>ProviderOptions</c>), không bằng cách phình giao diện.
///
/// <see cref="GenerateAsync"/> KHÔNG được ném lỗi cho lỗi của provider: nó trả
/// về <see cref="AiResponse"/> với <c>Success = false</c>. Lý do rất thực dụng
/// — bộ định tuyến phải thử provider tiếp theo, và một luồng điều khiển dựa
/// trên ngoại lệ sẽ khiến "thử provider khác" trở thành bắt ngoại lệ trong
/// vòng lặp. Ngoại lệ chỉ dành cho lỗi LẬP TRÌNH (tham số null) và cho
/// <see cref="OperationCanceledException"/> khi người gọi huỷ.
/// </summary>
public interface IAiProvider
{
    string ProviderName { get; }

    IReadOnlyList<string> SupportedModels { get; }

    Task<AiResponse> GenerateAsync(AiRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Provider có sẵn sàng không. Phải NHANH và không tốn tiền.
    ///
    /// Với Ollama là một lời gọi <c>/api/tags</c>; với endpoint trả tiền thì
    /// KHÔNG gọi thử một lượt sinh chữ — như vậy mỗi lần kiểm tra sẵn sàng lại
    /// tốn tiền. Khi không kiểm được rẻ thì trả về true và để lần gọi thật báo
    /// lỗi; <c>ProviderHealthTracker</c> sẽ ghi nhận và tạm loại nó ra.
    /// </summary>
    Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default);
}
