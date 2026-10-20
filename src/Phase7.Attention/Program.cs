using Ai.Phase7.Demos;

// ======================================================================
// MINI AI — PHASE 7: ATTENTION
//
//     Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
//
// Mô hình ở Phase 6 nhét ngữ cảnh vào một vector nối cứng, mọi từ đóng
// góp như nhau. Attention cho phép mỗi token TỰ CHỌN nên lấy thông tin
// từ token nào, và lấy bao nhiêu.
// ======================================================================

Console.WriteLine("############################################");
Console.WriteLine("#   MINI AI — PHASE 7: ATTENTION           #");
Console.WriteLine("############################################\n");

AttentionDemo.Run();

Console.WriteLine("\n\n=== PHASE 7 HOÀN TẤT ===");
