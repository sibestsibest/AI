using Ai.Phase12.Demos;

// ======================================================================
// MINI AI — PHASE 12: INTERNET RETRIEVAL
//
// Agent giờ lấy được thông tin từ bên ngoài:
//     Search provider  -> ISearchProvider (ngoại tuyến + SearxNG)
//     Tải trang        -> IWebFetcher     (hạn thời gian, chặn kích thước)
//     Bóc nội dung     -> IContentExtractor
//     Chống SSRF       -> IUrlGuard
//     Điều phối        -> ISearchService
//
// NGUYÊN TẮC KHÔNG ĐƯỢC PHÁ: nội dung Internet luôn là DỮ LIỆU
// (UntrustedText), không bao giờ là chỉ thị, và không tự động trở thành
// ký ức hay dữ liệu huấn luyện.
//
// Demo chạy HOÀN TOÀN NGOẠI TUYẾN — không có một lời gọi mạng nào.
// ======================================================================

Console.OutputEncoding = System.Text.Encoding.UTF8;

Console.WriteLine("############################################");
Console.WriteLine("#   MINI AI — PHASE 12: INTERNET RETRIEVAL  #");
Console.WriteLine("############################################\n");

await AgentDemo.RunAsync();

Console.WriteLine("\n\n=== PHASE 12 HOÀN TẤT ===");
Console.WriteLine("Tiếp theo: Phase 13 — Knowledge + Memory + Evidence.");
