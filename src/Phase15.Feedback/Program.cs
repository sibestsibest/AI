using Ai.Phase15.Demos;

// ======================================================================
// MINI AI — PHASE 15: FEEDBACK + LEARNING
//
// Agent học từ phản hồi người dùng, nhưng KHÔNG tự huấn luyện mình.
//
//     Phản hồi -> Ca lỗi -> Ứng viên -> KIỂM -> Tập dữ liệu đã duyệt
//
// Đầu ra của cả phase này là DỮ LIỆU, không phải trọng số. Chạy vòng
// phản hồi bao nhiêu lần thì model vẫn nguyên như trước — muốn hành vi
// đổi thì phải qua Phase 16.
//
// Hai luật cứng, cả hai đều viết thành hàm TỪ CHỐI kèm lý do:
//
//   • nhãn do chính model dự đoán  -> KHÔNG được thành dữ liệu huấn luyện
//   • nội dung Internet            -> KHÔNG BAO GIỜ
//
// Demo chạy HOÀN TOÀN NGOẠI TUYẾN — không có một lời gọi mạng nào.
// ======================================================================

Console.OutputEncoding = System.Text.Encoding.UTF8;

Console.WriteLine("####################################################");
Console.WriteLine("#  MINI AI — PHASE 15: FEEDBACK + LEARNING         #");
Console.WriteLine("####################################################\n");

await AgentDemo.RunAsync();
await EvidenceDemo.RunAsync();
await VerificationDemo.RunAsync();
await FeedbackDemo.RunAsync();

Console.WriteLine("\n\n=== PHASE 15 HOÀN TẤT ===");
Console.WriteLine("Tiếp theo: Phase 16 — Controlled Retraining.");
