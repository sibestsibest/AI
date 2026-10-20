using System.Text.Json;

namespace Ai.Providers.Configuration;

/// <summary>
/// CHỖ DUY NHẤT đọc được giá trị của một khoá API.
///
/// Ba nguồn, xét theo thứ tự:
///
///   1. giá trị được TIÊM vào lúc chạy (dùng cho test, và cho nơi đã có sẵn
///      một kho bí mật như Key Vault — chỉ cần nạp vào đây)
///   2. biến môi trường
///   3. file user-secrets của .NET, nếu có <see cref="UserSecretsId"/>
///
/// VÌ SAO ĐỌC FILE USER-SECRETS TRỰC TIẾP thay vì dùng
/// <c>Microsoft.Extensions.Configuration.UserSecrets</c>? Vì cả dự án này
/// không có PackageReference nào ngoài bộ test, và user-secrets chỉ là một file
/// JSON ở một đường dẫn đã được quy ước
/// (<c>%APPDATA%\Microsoft\UserSecrets\&lt;id&gt;\secrets.json</c>). Đọc nó bằng
/// System.Text.Json là mười dòng; kéo thêm ba package vào để làm đúng mười
/// dòng đó thì đắt hơn cái nó mua được.
///
/// KHÔNG có hàm nào in ra hay trả về danh sách bí mật. <see cref="Describe"/>
/// chỉ nói bí mật CÓ hay KHÔNG, và nếu có thì dài bao nhiêu ký tự — đủ để dò
/// lỗi cấu hình ("tôi đặt biến rồi mà" → "biến rỗng"), không đủ để lộ gì.
/// </summary>
public sealed class SecretResolver
{
    private readonly Dictionary<string, string> _injected;
    private readonly Func<string, string?> _environment;
    private readonly Lazy<Dictionary<string, string>> _userSecrets;

    public SecretResolver(
        IReadOnlyDictionary<string, string>? injected = null,
        Func<string, string?>? environment = null,
        string? userSecretsId = null)
    {
        _injected = injected is null
            ? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            : new Dictionary<string, string>(injected, StringComparer.OrdinalIgnoreCase);

        _environment = environment ?? Environment.GetEnvironmentVariable;
        UserSecretsId = userSecretsId;
        _userSecrets = new Lazy<Dictionary<string, string>>(LoadUserSecrets);
    }

    public string? UserSecretsId { get; }

    /// <summary>Lấy bí mật theo tên. Trả về null nếu không có ở nguồn nào — KHÔNG ném.</summary>
    public string? Resolve(string? name)
    {
        if (string.IsNullOrWhiteSpace(name)) return null;

        if (_injected.TryGetValue(name, out string? injected) && !string.IsNullOrWhiteSpace(injected))
        {
            return injected;
        }

        string? fromEnvironment = _environment(name);
        if (!string.IsNullOrWhiteSpace(fromEnvironment)) return fromEnvironment;

        return _userSecrets.Value.TryGetValue(name, out string? secret) && !string.IsNullOrWhiteSpace(secret)
            ? secret
            : null;
    }

    public bool Has(string? name) => Resolve(name) is not null;

    /// <summary>Mô tả an toàn để ghi log: có/không và độ dài, KHÔNG có giá trị.</summary>
    public string Describe(string? name)
    {
        if (string.IsNullOrWhiteSpace(name)) return "không cần khoá";

        string? value = Resolve(name);

        return value is null
            ? $"{name}: CHƯA ĐẶT"
            : $"{name}: đã đặt ({value.Length} ký tự)";
    }

    /// <summary>
    /// Đọc file user-secrets. Mọi lỗi đều bị nuốt có chủ ý: thiếu file, JSON
    /// sai, không có quyền đọc — tất cả đều chỉ có nghĩa là "không có bí mật ở
    /// nguồn này", và không được làm sập ứng dụng.
    /// </summary>
    private Dictionary<string, string> LoadUserSecrets()
    {
        var empty = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (string.IsNullOrWhiteSpace(UserSecretsId)) return empty;

        try
        {
            string root = OperatingSystem.IsWindows()
                ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                    "Microsoft", "UserSecrets")
                : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    ".microsoft", "usersecrets");

            string path = Path.Combine(root, UserSecretsId, "secrets.json");

            if (!File.Exists(path)) return empty;

            using var document = JsonDocument.Parse(File.ReadAllText(path));

            if (document.RootElement.ValueKind != JsonValueKind.Object) return empty;

            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.String)
                {
                    empty[property.Name] = property.Value.GetString() ?? "";
                }
            }

            return empty;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            return empty;
        }
    }
}
