using Ai.Phase13.Demos;

// ======================================================================
// MINI AI — PHASE 13: KNOWLEDGE + MEMORY + EVIDENCE
//
// Agent giờ trả lời CÓ CĂN CỨ, và căn cứ nào cũng truy được về nguồn:
//     Knowledge  -> KnowledgeStore       (đã biên tập, Verified)
//     Memory     -> MemoryStore          (Phase 10, UserProvided)
//     Context    -> ConversationContext  (lượt vừa rồi, UserProvided)
//     Reasoning  -> ReasoningEngine      (Phase 9, Derived)
//     Internet   -> SearchService        (Phase 12, Untrusted)
//            ↓
//     EvidenceCollector -> bỏ trùng chéo kênh -> xếp hạng -> EvidenceSet
//
// LUẬT CỨNG: nội dung Internet KHÔNG BAO GIỜ tự động thành kiến thức lâu
// dài hay dữ liệu huấn luyện. Chốt thực thi nằm ở KnowledgeStore.TryPromote.
//
// Demo chạy HOÀN TOÀN NGOẠI TUYẾN — không có một lời gọi mạng nào.
// ======================================================================

Console.OutputEncoding = System.Text.Encoding.UTF8;

Console.WriteLine("##################################################");
Console.WriteLine("#  MINI AI — PHASE 13: KNOWLEDGE + MEMORY + EVIDENCE  #");
Console.WriteLine("##################################################\n");

await AgentDemo.RunAsync();
await EvidenceDemo.RunAsync();

Console.WriteLine("\n\n=== PHASE 13 HOÀN TẤT ===");
Console.WriteLine("Tiếp theo: Phase 14 — Verification + Answer Generation.");
