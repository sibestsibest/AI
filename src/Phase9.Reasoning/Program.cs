using Ai.Phase9.Demos;

// ======================================================================
// MINI AI — PHASE 9: REASONING ENGINE
//
// Không ép mạng neural làm tất cả. Phase 1–8 giỏi nhận mẫu và xử lý cái
// mơ hồ, nhưng không tính đúng 10 + 20 một cách bảo đảm và không giải
// thích được vì sao. Suy luận ký hiệu làm được cả hai — bằng cách khác hẳn.
//
//     Parse facts -> Store facts -> Find relevant facts
//     -> Apply rule -> Calculate -> Return answer
// ======================================================================

Console.WriteLine("############################################");
Console.WriteLine("#   MINI AI — PHASE 9: REASONING ENGINE    #");
Console.WriteLine("############################################\n");

ReasoningDemo.Run();

Console.WriteLine("\n\n=== PHASE 9 HOÀN TẤT ===");
