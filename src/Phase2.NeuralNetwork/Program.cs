using Ai.Phase2.Demos;

// ======================================================================
// MINI AI — PHASE 2: NEURAL NETWORK (FORWARD PASS)
//
// Phase 1: 1 neuron, 1 đầu vào, không activation.
// Phase 2: nhiều neuron xếp thành layer, nhiều layer nối tiếp, có activation.
//
// Backpropagation KHÔNG có ở phase này — đó là Phase 3. Để chứng minh
// kiến trúc đã đúng, mạng được train bằng numerical gradient.
// ======================================================================

Console.WriteLine("########################################");
Console.WriteLine("#   MINI AI — PHASE 2: NEURAL NETWORK  #");
Console.WriteLine("########################################\n");

LinAlgDemo.Run();
ForwardTraceDemo.Run();
Phase1EquivalenceDemo.Run();
NonlinearFitDemo.Run();

Console.WriteLine("\n\n=== PHASE 2 HOÀN TẤT ===");
Console.WriteLine("Mạng đã forward đúng và học được quan hệ phi tuyến.");
Console.WriteLine("Phase 3 sẽ thay numerical gradient bằng backpropagation.");
