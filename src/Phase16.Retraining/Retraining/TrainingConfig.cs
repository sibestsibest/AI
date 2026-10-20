using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace Ai.Phase16.Retraining;

/// <summary>
/// CẤU HÌNH HUẤN LUYỆN — mọi thứ cần để dựng lại y hệt một model.
///
/// Đây là nửa còn lại của lời hứa tái lập. Nửa kia là vân tay tập dữ liệu
/// (Phase 15). Biết cả hai thì train lại phải ra ĐÚNG bộ trọng số cũ, đến từng
/// chữ số — và <c>RetrainingPipelineTests</c> kiểm chính điều đó.
///
/// <see cref="Seed"/> nằm trong cấu hình chứ không phải một tham số truyền
/// riêng, và đó là chỗ hay bị bỏ sót nhất: trọng số khởi tạo ngẫu nhiên quyết
/// định model cuối cùng cũng nhiều như dữ liệu. Không ghi seed lại thì "cùng
/// dữ liệu, cùng siêu tham số" vẫn cho ra hai model khác nhau, và không ai giải
/// thích nổi vì sao.
/// </summary>
public sealed record TrainingConfig
{
    /// <summary>Hạt giống cho trọng số khởi tạo. Phải ghi lại, nếu không thì không tái lập được.</summary>
    public int Seed { get; init; } = 42;

    public int Epochs { get; init; } = 3000;

    public double LearningRate { get; init; } = 0.3;

    public int HiddenSize { get; init; } = 16;

    /// <summary>
    /// Phần dữ liệu để dành cho đánh giá, 0..0.5.
    ///
    /// Không cho vượt 0.5: quá nửa dữ liệu đem đi đánh giá thì phần train còn
    /// lại quá ít, và ta đo một model tệ hơn hẳn model sẽ thật sự chạy.
    /// </summary>
    public double ValidationShare { get; init; } = 0.2;

    /// <summary>Kiểm tra cấu hình hợp lệ. Ném ngay thay vì để một lượt train dài chạy rồi mới hỏng.</summary>
    public void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(Epochs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(LearningRate, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(HiddenSize, 1);

        if (ValidationShare is < 0 or > 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(ValidationShare),
                ValidationShare, "phần dữ liệu đánh giá phải nằm trong [0, 0.5]");
        }
    }

    /// <summary>
    /// Vân tay cấu hình — 8 ký tự hex, tính giống cách Phase 15 tính vân tay
    /// tập dữ liệu (SHA-256, không dùng GetHashCode vì nó ngẫu nhiên hoá theo
    /// tiến trình).
    ///
    /// Định dạng số dùng <see cref="CultureInfo.InvariantCulture"/>: máy đặt
    /// locale tiếng Việt in "0,3" thay vì "0.3", và khi đó cùng một cấu hình
    /// lại ra hai vân tay khác nhau ở hai máy.
    /// </summary>
    public string Fingerprint
    {
        get
        {
            string payload = string.Format(CultureInfo.InvariantCulture,
                "seed={0};epochs={1};lr={2:R};hidden={3};val={4:R}",
                Seed, Epochs, LearningRate, HiddenSize, ValidationShare);

            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(payload));
            return Convert.ToHexString(hash)[..8].ToLowerInvariant();
        }
    }

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture,
            "seed {0}, {1} epoch, lr {2}, ẩn {3}, đánh giá {4:P0} [{5}]",
            Seed, Epochs, LearningRate, HiddenSize, ValidationShare, Fingerprint);
}
