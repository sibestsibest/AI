using Ai.Phase14.Demos;

// ======================================================================
// MINI AI — PHASE 14: VERIFICATION + ANSWER GENERATION
//
// Agent không còn "sinh ra một câu trả lời" nữa. Nó đi qua dây chuyền:
//
//     Câu hỏi -> Retrieve -> Collect Evidence -> Reason -> Verify
//             -> Generate Answer -> Validate -> GroundedAnswer
//
// Bốn kết luận có thể ra: Answered / Uncertain / Conflicted / Insufficient.
//
// Khi căn cứ không đủ, agent NÓI THẲNG là không biết. Khi các căn cứ nói
// ngược nhau, nó trình bày CẢ HAI phía thay vì chọn bên.
//
// Tóm tắt suy luận lưu QUYẾT ĐỊNH và THAM CHIẾU CĂN CỨ — không lưu và
// không in ra dòng suy nghĩ nội bộ.
//
// Demo chạy HOÀN TOÀN NGOẠI TUYẾN — không có một lời gọi mạng nào.
// ======================================================================

Console.OutputEncoding = System.Text.Encoding.UTF8;

Console.WriteLine("########################################################");
Console.WriteLine("#  MINI AI — PHASE 14: VERIFICATION + ANSWER GENERATION  #");
Console.WriteLine("########################################################\n");

await AgentDemo.RunAsync();
await EvidenceDemo.RunAsync();
await VerificationDemo.RunAsync();

Console.WriteLine("\n\n=== PHASE 14 HOÀN TẤT ===");
Console.WriteLine("Tiếp theo: Phase 15 — Feedback + Learning.");
