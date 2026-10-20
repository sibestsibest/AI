using Ai.Chat;

// ======================================================================
// MINI AI — MÀN HÌNH CHAT
//
// Gõ câu hỏi, agent trả lời. Dùng chồng đầy đủ của Phase 16:
//
//     câu hỏi -> phân loại ý định -> {công cụ | căn cứ 5 kênh} -> kiểm chứng
//             -> câu trả lời có nguồn  -> phản hồi -> dữ liệu -> model mới
//
// Mặc định in gọn; /chitiet để xem ý định, độ tự tin, từng mẩu căn cứ và các
// bước quyết định. /giupdo để xem hết lệnh.
//
// Chạy HOÀN TOÀN NGOẠI TUYẾN — không một lời gọi mạng nào.
//
//     dotnet run --project src/Ai.Chat
// ======================================================================

Console.OutputEncoding = System.Text.Encoding.UTF8;

var session = ChatSession.Start(message => Console.WriteLine($"  {message}"));

await new ChatConsole(session).RunAsync();
