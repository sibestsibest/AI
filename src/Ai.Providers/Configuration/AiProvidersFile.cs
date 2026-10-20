using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace Ai.Providers.Configuration;

/// <summary>Kết quả đọc cấu hình: luôn có một bộ cấu hình dùng được, kèm các vấn đề đã phát hiện.</summary>
public sealed record ConfigLoadResult(AiProvidersOptions Options, IReadOnlyList<string> Problems, string Source)
{
    public bool HasProblems => Problems.Count > 0;

    public override string ToString() =>
        $"{Source}: {Options.Providers.Count} provider, {Problems.Count} vấn đề";
}

/// <summary>
/// ĐỌC CẤU HÌNH TỪ FILE JSON — và từ chối file có vẻ chứa khoá thật.
///
/// VÌ SAO KHÔNG DÙNG Microsoft.Extensions.Configuration?
///
/// Vì thứ duy nhất cần ở đây là "đọc một file JSON thành object", và
/// System.Text.Json trong BCL làm đúng việc đó. Cả dự án không có
/// PackageReference nào ngoài bộ test (xem README), nên kéo vào ba package cấu
/// hình để đọc một file là đổi một thứ lớn để mua một thứ nhỏ. Bù lại, phần
/// ghép bí mật từ nhiều nguồn được làm tường minh trong
/// <see cref="SecretResolver"/> — mà đó lại là chỗ ta MUỐN đọc rõ từng dòng.
///
/// KIỂM TRA AN TOÀN QUAN TRỌNG NHẤT của lớp này không phải là kiểm cú pháp, mà
/// là <see cref="RejectLiteralSecrets"/>: nếu trong file có trường trông như
/// một khoá API thật, cấu hình đó bị TỪ CHỐI kèm lý do. Lý do rất thực tế —
/// file cấu hình bị commit, còn biến môi trường thì không. Một cảnh báo nhẹ sẽ
/// bị bỏ qua; một lần từ chối thì không.
/// </summary>
public static class AiProvidersFile
{
    /// <summary>Tên file mặc định, tìm cạnh file thực thi.</summary>
    public const string DefaultFileName = "aiproviders.json";

    /// <summary>Biến môi trường trỏ tới file cấu hình, nếu muốn đặt chỗ khác.</summary>
    public const string PathEnvironmentVariable = "AI_PROVIDERS";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        Converters = { new JsonStringEnumConverter() },
    };

    /// <summary>
    /// Những mẫu trông như khoá thật. Danh sách này KHÔNG cần đầy đủ để có ích:
    /// nó chặn đúng các tiền tố mà người ta hay dán vào, và chặn cả chuỗi dài
    /// ngẫu nhiên không có khoảng trắng — hình dạng chung của mọi khoá API.
    /// </summary>
    private static readonly Regex[] LiteralSecretPatterns =
    [
        new(@"\bsk-[A-Za-z0-9_\-]{16,}", RegexOptions.CultureInvariant),
        new(@"\bgsk_[A-Za-z0-9_\-]{16,}", RegexOptions.CultureInvariant),
        new(@"\bhf_[A-Za-z0-9]{16,}", RegexOptions.CultureInvariant),
        new(@"\bAIza[0-9A-Za-z_\-]{20,}", RegexOptions.CultureInvariant),
        new(@"\bBearer\s+[A-Za-z0-9._\-]{20,}", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
    ];

    /// <summary>
    /// Đọc cấu hình theo thứ tự: đường dẫn truyền vào → biến môi trường → file
    /// cạnh exe → mặc định chỉ-local.
    ///
    /// KHÔNG có file cũng KHÔNG phải lỗi: hệ thống phải chạy được ngay khi vừa
    /// tải về, và khi đó nó dùng <see cref="AiProvidersOptions.LocalOnlyDefault"/>
    /// — tức là chỉ thử Ollama tại máy, không gửi gì ra Internet.
    /// </summary>
    public static ConfigLoadResult Load(string? path = null, Func<string, string?>? environment = null)
    {
        environment ??= Environment.GetEnvironmentVariable;

        string? resolved = FirstExisting(
            path,
            environment(PathEnvironmentVariable),
            Path.Combine(AppContext.BaseDirectory, DefaultFileName));

        if (resolved is null)
        {
            return new ConfigLoadResult(AiProvidersOptions.LocalOnlyDefault, [],
                "mặc định (không tìm thấy file cấu hình) — chỉ Ollama tại máy");
        }

        try
        {
            return Parse(File.ReadAllText(resolved), resolved);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new ConfigLoadResult(AiProvidersOptions.LocalOnlyDefault,
                [$"không đọc được '{resolved}': {ex.Message}"], "mặc định (đọc file thất bại)");
        }
    }

    /// <summary>Đọc cấu hình từ một chuỗi JSON. Tách riêng để test được mà không cần file thật.</summary>
    public static ConfigLoadResult Parse(string json, string source = "chuỗi JSON")
    {
        ArgumentNullException.ThrowIfNull(json);

        var problems = RejectLiteralSecrets(json);

        if (problems.Count > 0)
        {
            // Từ chối TOÀN BỘ file, không chỉ trường có vấn đề: nếu đã có một
            // khoá thật trong đó thì file này không được tin, và cũng không nên
            // được dùng tiếp như thể chỉ sai một chỗ nhỏ.
            return new ConfigLoadResult(AiProvidersOptions.LocalOnlyDefault, problems,
                $"{source} BỊ TỪ CHỐI — dùng mặc định chỉ-local");
        }

        AiProvidersOptions? options;

        try
        {
            options = JsonSerializer.Deserialize<AiProvidersOptions>(json, JsonOptions);
        }
        catch (JsonException ex)
        {
            return new ConfigLoadResult(AiProvidersOptions.LocalOnlyDefault,
                [$"JSON sai cú pháp: {ex.Message}"], $"{source} (không đọc được)");
        }

        if (options is null || options.Providers.Count == 0)
        {
            return new ConfigLoadResult(AiProvidersOptions.LocalOnlyDefault,
                ["cấu hình không khai provider nào"], $"{source} (rỗng)");
        }

        return new ConfigLoadResult(options, Validate(options), source);
    }

    /// <summary>
    /// Kiểm tra tính hợp lệ. Trả về danh sách vấn đề thay vì ném ngoại lệ, vì
    /// một provider khai sai KHÔNG được làm hỏng cả cấu hình — những provider
    /// còn lại vẫn dùng được, và bộ định tuyến chỉ cần biết cái nào đừng thử.
    /// </summary>
    public static IReadOnlyList<string> Validate(AiProvidersOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        var problems = new List<string>();

        foreach (string duplicate in options.Providers
                     .GroupBy(p => p.Name, StringComparer.OrdinalIgnoreCase)
                     .Where(g => g.Count() > 1)
                     .Select(g => g.Key))
        {
            problems.Add($"tên provider '{duplicate}' bị khai hai lần — tên là khoá tra cứu, phải duy nhất");
        }

        foreach (var provider in options.Providers)
        {
            if (string.IsNullOrWhiteSpace(provider.Name))
            {
                problems.Add("có provider không có tên");
            }

            // Endpoint của loại Responses được phép BỎ TRỐNG: SDK chính thức đã
            // biết địa chỉ mặc định của OpenAI. Khai vào chỉ cần khi trỏ tới
            // một cổng khác (Azure, proxy nội bộ) — và lúc đó nó vẫn bị kiểm
            // như mọi endpoint khác.
            bool endpointOptional = provider.Type is ProviderKind.Scripted or ProviderKind.OpenAiResponses;

            if (endpointOptional && string.IsNullOrWhiteSpace(provider.Endpoint))
            {
                // không có gì để kiểm
            }
            else if (provider.Type != ProviderKind.Scripted && string.IsNullOrWhiteSpace(provider.Endpoint))
            {
                problems.Add($"{provider.Name}: thiếu Endpoint");
            }
            else if (provider.Type != ProviderKind.Scripted
                     && !Uri.TryCreate(provider.Endpoint, UriKind.Absolute, out var uri))
            {
                problems.Add($"{provider.Name}: Endpoint '{provider.Endpoint}' không phải URL tuyệt đối");
            }
            else if (provider.Type != ProviderKind.Scripted
                     && Uri.TryCreate(provider.Endpoint, UriKind.Absolute, out var checkedUri)
                     && checkedUri.Scheme == Uri.UriSchemeHttp
                     && !IsLoopback(checkedUri))
            {
                // HTTP trần ra ngoài máy nghĩa là khoá API đi trên đường không
                // mã hoá. Với localhost thì không có đường nào để nghe, nên đó
                // là ngoại lệ duy nhất — và Ollama chạy đúng như vậy.
                problems.Add($"{provider.Name}: Endpoint dùng http:// ra ngoài máy — khoá sẽ đi không mã hoá, hãy dùng https://");
            }

            if (provider.Type is ProviderKind.OpenAiCompatible or ProviderKind.OpenAiResponses
                && string.IsNullOrWhiteSpace(provider.ApiKeyEnvironmentVariable))
            {
                problems.Add($"{provider.Name}: thiếu ApiKeyEnvironmentVariable (TÊN biến môi trường, không phải khoá)");
            }

            if (provider.MaxOutputTokens is < 0 or > 200_000)
            {
                problems.Add($"{provider.Name}: MaxOutputTokens = {provider.MaxOutputTokens}, phải trong 0..200000");
            }

            if (provider.ReasoningEffort is { } effort
                && effort is not ("minimal" or "low" or "medium" or "high"))
            {
                problems.Add($"{provider.Name}: ReasoningEffort = '{effort}', " +
                             "chỉ nhận minimal / low / medium / high");
            }

            if (provider.TimeoutSeconds is < 1 or > 300)
            {
                problems.Add($"{provider.Name}: TimeoutSeconds = {provider.TimeoutSeconds}, phải trong 1..300");
            }

            if (provider.MaxRetries is < 0 or > 5)
            {
                problems.Add($"{provider.Name}: MaxRetries = {provider.MaxRetries}, phải trong 0..5 — " +
                             "thử lại không giới hạn là cách làm nghẽn chính mình");
            }
        }

        if (options.DefaultProvider is { } preferred && options.Find(preferred) is null)
        {
            problems.Add($"DefaultProvider = '{preferred}' nhưng không có provider nào tên như vậy");
        }

        return problems;
    }

    /// <summary>Tìm khoá thật bị dán vào file cấu hình.</summary>
    public static IReadOnlyList<string> RejectLiteralSecrets(string json)
    {
        ArgumentNullException.ThrowIfNull(json);

        var problems = new List<string>();

        foreach (var pattern in LiteralSecretPatterns)
        {
            var match = pattern.Match(json);

            if (!match.Success) continue;

            // KHÔNG in giá trị khớp ra thông điệp lỗi — thông điệp này sẽ vào
            // log. Chỉ nói dạng khoá nào và ở vị trí nào.
            problems.Add($"file chứa chuỗi trông như khoá API thật (dạng '{Shape(match.Value)}', " +
                         $"vị trí {match.Index}) — khoá phải đặt qua biến môi trường, " +
                         "cấu hình chỉ ghi TÊN biến");
        }

        return problems;
    }

    /// <summary>Hình dạng của một chuỗi bí mật: giữ tiền tố, thay phần còn lại bằng dấu.</summary>
    private static string Shape(string secret) =>
        secret.Length <= 4 ? "…" : string.Concat(secret.AsSpan(0, 4), "…");

    private static bool IsLoopback(Uri uri) =>
        uri.IsLoopback
        || uri.Host.Equals("localhost", StringComparison.OrdinalIgnoreCase)
        || uri.Host is "127.0.0.1" or "::1";

    private static string? FirstExisting(params string?[] candidates) =>
        candidates.FirstOrDefault(c => !string.IsNullOrWhiteSpace(c) && File.Exists(c));
}
