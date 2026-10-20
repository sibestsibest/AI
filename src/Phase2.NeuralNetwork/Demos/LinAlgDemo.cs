using Ai.Phase2.LinAlg;

namespace Ai.Phase2.Demos;

/// <summary>Demo 1 — các phép toán vector/ma trận tự viết, đối chiếu với tính tay.</summary>
public static class LinAlgDemo
{
    public static void Run()
    {
        Console.WriteLine("=== DEMO 1: VECTOR & MATRIX TỰ VIẾT ===\n");

        var a = Vec.Of(1, 2, 3);
        var b = Vec.Of(4, 5, 6);

        Console.WriteLine($"  a = {a}");
        Console.WriteLine($"  b = {b}\n");
        Console.WriteLine($"  a · b (dot)      = {a.Dot(b),8:F4}   tính tay: 1·4 + 2·5 + 3·6 = 32");
        Console.WriteLine($"  a + b            = {a.Add(b)}   tính tay: [5, 7, 9]");
        Console.WriteLine($"  b − a            = {b.Subtract(a)}   tính tay: [3, 3, 3]");
        Console.WriteLine($"  a × 2            = {a.Scale(2)}   tính tay: [2, 4, 6]");
        Console.WriteLine($"  a ⊙ b (Hadamard) = {a.Hadamard(b)}   tính tay: [4, 10, 18]");
        Console.WriteLine($"  ‖a‖              = {a.Norm(),8:F4}   tính tay: √(1+4+9) = √14 = 3.7417");

        Console.WriteLine("\n  Ma trận W (2x2) nhân vector x:");
        var w = Mat.FromRows([1, 2], [3, 4]);
        var x = Vec.Of(10, 20);

        Console.WriteLine($"{w}");
        Console.WriteLine($"  x = {x}");
        Console.WriteLine($"  W · x = {w.Multiply(x)}   tính tay: [1·10+2·20, 3·10+4·20] = [50, 110]");

        Console.WriteLine("\n  Wᵀ (chuyển vị — Phase 3 sẽ cần để lan truyền lỗi ngược):");
        Console.WriteLine($"{w.Transpose()}");

        Console.WriteLine("\n  Kiểm tra bắt lỗi kích thước:");
        try
        {
            w.Multiply(Vec.Of(1, 2, 3));
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"    Đã chặn đúng: {ex.Message.Split(" (Parameter")[0]}");
        }
    }
}
