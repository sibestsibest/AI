using Ai.Phase4.Network;

namespace Ai.Phase4.Training;

/// <summary>
/// Biến đầu ra liên tục của mạng thành QUYẾT ĐỊNH rời rạc.
///
/// Mạng phân loại không trả về "0" hay "1" — nó trả về một xác suất, ví dụ
/// 0.973. Bước cuối cùng là so với ngưỡng 0.5. Đây là chỗ hồi quy và phân
/// loại tách đôi: cùng một mạng, chỉ khác cách đọc đầu ra.
/// </summary>
public static class Classifier
{
    public const double Threshold = 0.5;

    public static int Classify(double probability) => probability >= Threshold ? 1 : 0;

    /// <summary>Tỉ lệ mẫu được phân loại đúng, từ 0 đến 1.</summary>
    public static double Accuracy(NeuralNetwork network, TrainingData data)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(data);

        int correct = 0;
        foreach (var sample in data)
        {
            var predicted = network.Forward(sample.Input);

            bool allMatch = true;
            for (int i = 0; i < predicted.Length; i++)
            {
                if (Classify(predicted[i]) != Classify(sample.Target[i]))
                {
                    allMatch = false;
                    break;
                }
            }

            if (allMatch) correct++;
        }

        return (double)correct / data.Count;
    }
}
