using Ai.Phase3.Demos;

// ======================================================================
// MINI AI — PHASE 3: BACKPROPAGATION
//
// Phase 2 train được nhưng bằng numerical gradient: mỗi bước học tốn
// 2·P lần forward pass. Phase 3 thay bằng backpropagation — cùng kết quả,
// chỉ tốn 1 forward + 1 backward, không phụ thuộc số tham số.
//
// Chu trình đầy đủ:
//   Forward -> Prediction -> Loss -> Backpropagation -> Gradients
//   -> Update weights -> Forward again
// ======================================================================

Console.WriteLine("############################################");
Console.WriteLine("#   MINI AI — PHASE 3: BACKPROPAGATION     #");
Console.WriteLine("############################################\n");

HandCalculationDemo.Run();
GradientCheckDemo.Run();
SpeedComparisonDemo.Run();
DeepTrainingDemo.Run();

Console.WriteLine("\n\n=== PHASE 3 HOÀN TẤT ===");
Console.WriteLine("Backpropagation đã đối chiếu khớp với đạo hàm số trên mọi kiến trúc.");
Console.WriteLine("Phase 4 sẽ dùng nó để dạy mạng học XOR.");
