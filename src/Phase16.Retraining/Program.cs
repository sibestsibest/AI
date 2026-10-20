using Ai.Phase16.Demos;

// ======================================================================
// MINI AI — PHASE 16: CONTROLLED RETRAINING
//
// Vòng khép lại: dữ liệu đã duyệt của Phase 15 trở thành TRỌNG SỐ — nhưng
// chỉ khi vượt được cửa đánh giá.
//
//     Tập dữ liệu -> Chia -> Train -> Đánh giá -> So sánh -> Cửa -> Model
//
// Luật lớn nhất: MODEL MỚI KHÔNG TỰ ĐỘNG THAY MODEL ĐANG CHẠY. Nó phải
// đạt sàn accuracy, không thụt lùi macro F1, không làm hỏng ví dụ nào
// model cũ đang làm đúng, và đúng 100% bộ vàng.
//
// Tái lập: cùng tập dữ liệu + cùng cấu hình -> ĐÚNG bộ trọng số ấy, đến
// từng chữ số. Quay lui: model cũ dựng lại từ trọng số đã lưu, hành vi
// trở lại y như trước.
//
// Demo chạy HOÀN TOÀN NGOẠI TUYẾN — không có một lời gọi mạng nào.
// ======================================================================

Console.OutputEncoding = System.Text.Encoding.UTF8;

Console.WriteLine("####################################################");
Console.WriteLine("#  MINI AI — PHASE 16: CONTROLLED RETRAINING       #");
Console.WriteLine("####################################################\n");

await AgentDemo.RunAsync();
await EvidenceDemo.RunAsync();
await VerificationDemo.RunAsync();
await FeedbackDemo.RunAsync();
RetrainingDemo.Run();

Console.WriteLine("\n\n=== PHASE 16 HOÀN TẤT ===");
Console.WriteLine("Toàn bộ lộ trình đã xong: USER -> AGENT -> MEMORY/INTERNET -> EVIDENCE");
Console.WriteLine("-> REASONING -> VERIFICATION -> ANSWER -> FEEDBACK -> LEARNING -> RETRAINING.");
