# AI — Tự xây AI bằng C# từ số 0

Không dùng OpenAI / Azure OpenAI / ML.NET / bất kỳ thư viện AI nào.
Mọi thuật toán — matrix, vector, forward, loss, backpropagation, gradient descent,
activation, softmax, cross entropy, embedding, attention — đều tự viết bằng C# thuần.

Phase 1–16 **không có một PackageReference nào** ngoài bộ test. Riêng lớp
[Ai.Providers](src/Ai.Providers/) là phần nối tới LLM ngoài (Ollama tại máy, hoặc
endpoint kiểu OpenAI) — nó **mặc định TẮT**, chỉ làm khâu diễn đạt câu trả lời, và cũng
không dùng SDK của nhà cung cấp nào (chỉ `HttpClient` + `System.Text.Json`).

Yêu cầu: **.NET 10 SDK** (LTS).

```bash
dotnet test                                      # 2491 unit tests, 19 project
dotnet run --project src/Phase1.LinearRegression # chạy từng phase độc lập
dotnet run --project src/Ai.Chat             # màn hình chat (hỏi – đáp)
```

## Lộ trình

| Phase | Nội dung | Tests | Kết quả đo được |
|---|---|---|---|
| [1](src/Phase1.LinearRegression/) | Linear Regression | 36 | tự học `y = 2x + 1`, loss `3.9e-16` |
| [2](src/Phase2.NeuralNetwork/) | Neural Network (forward pass) | 79 | học `y = x²`, tốt hơn model tuyến tính 1282× |
| [3](src/Phase3.Backpropagation/) | Backpropagation | 43 | nhanh hơn numerical gradient 24×, gradient check đạt |
| [4](src/Phase4.Classification/) | Classification — XOR | 31 | 100% accuracy, perceptron chỉ đạt 50% |
| [5](src/Phase5.Embedding/) | Embedding + cosine similarity | 35 | `cos(cat, dog) = 0.988`, `cos(cat, car) = −0.14` |
| [6](src/Phase6.LanguageModel/) | Mini Language Model | 37 | học đúng phân bố tần suất của ngữ liệu |
| [7](src/Phase7.Attention/) | Attention | 24 | `softmax(QKᵀ/√d)V`, ví dụ 3 token tính tay |
| [8](src/Phase8.Transformer/) | Mini Transformer | 33 | 4616 tham số, gradient check lệch `5.8e-11` |
| [9](src/Phase9.Reasoning/) | Reasoning Engine (ký hiệu) | 55 | suy luận nhiều tầng, giải thích từng bước |
| [10](src/Phase10.Memory/) | Memory | 34 | store/retrieve/search/rank/forget có chọn lọc |
| [11](src/Phase11.Agent/) | AI Agent | 36 | ghép neural + memory + reasoning + 3 tools |
| [12](src/Phase12.Internet/) | Internet Retrieval | 237 | tra web có chống SSRF, bỏ trùng, xếp hạng nguồn |
| [13](src/Phase13.Evidence/) | Knowledge + Memory + Evidence | 323 | 5 kênh căn cứ, truy được nguồn, web không thành kiến thức |
| [14](src/Phase14.Verification/) | Verification + Answer Generation | 385 | 4 kết luận, không đủ căn cứ thì nói thẳng là không biết |
| [15](src/Phase15.Feedback/) | Feedback + Learning | 474 | phản hồi sinh ra DỮ LIỆU có phiên bản, không sinh trọng số |
| [16](src/Phase16.Retraining/) | Controlled Retraining | 539 | train lại tái lập bit-by-bit, model tệ hơn không được triển khai |

Ngoài 16 phase còn hai project công cụ:

| Project | Nội dung | Tests |
|---|---|---|
| [Ai.Chat](src/Ai.Chat/) | màn hình chat hỏi–đáp, chạy được cả vòng phản hồi → train lại | 14 |
| [Ai.Providers](src/Ai.Providers/) | nối LLM ngoài vào **khâu diễn đạt** (mặc định tắt) | 76 |

Mỗi phase là **một project độc lập**, chạy được riêng, có unit test riêng.
Phase sau mang theo bản sao core của phase trước rồi mở rộng — đổi lại là trùng lặp
code, nhưng mỗi phase đọc được từ đầu đến cuối như một tài liệu hoàn chỉnh.

---

## Phase 1 — Linear Regression

Model nhỏ nhất có thể: 1 neuron, 1 input, 2 tham số.

```
ŷ = w·x + b
L = (1/n)·Σ(ŷᵢ − yᵢ)²
∂L/∂w = (2/n)·Σ eᵢ·xᵢ      ∂L/∂b = (2/n)·Σ eᵢ
w ← w − lr·∂L/∂w
```

**Learning rate**: Hessian là `[[22,6],[6,2]]`, `λ_max ≈ 23.66`, nên điều kiện hội tụ
là `lr < 2/λ_max ≈ 0.0845`. Dùng `0.05`. Với `lr = 0.1` model phân kỳ (loss `5.3e+272`).

## Phase 2 — Neural Network

```
Neuron:  z = w·x + b , a = f(z)
Layer:   output = f(W·x + b)     W là OutputSize × InputSize, mỗi hàng một neuron
Network: đầu ra layer trước là đầu vào layer sau
```

Không có activation thì `W₂(W₁x + b₁) + b₂ = W'x + b'` — xếp bao nhiêu layer cũng
chỉ là một đường thẳng. Có `IActivation` + `Identity`, `Sigmoid`, `Tanh`, `ReLU`.

Backprop chưa có ở phase này; mạng được train bằng **numerical gradient**
(`(J(p+h) − J(p−h))/2h`) để chứng minh kiến trúc đúng mà không đi tắt.

## Phase 3 — Backpropagation

Ba công thức chain rule cho mỗi layer:

```
(1)  δᵢ      = ∂L/∂aᵢ · f'(zᵢ)
(2)  ∂L/∂wᵢⱼ = δᵢ · xⱼ  ,  ∂L/∂bᵢ = δᵢ
(3)  ∂L/∂xⱼ  = Σᵢ wᵢⱼ · δᵢ            (= Wᵀ·δ)
```

| | Numerical gradient | Backpropagation |
|---|---|---|
| Forward pass / bước | `2·P` | 1 |
| Đo thật (25 tham số, 3000 epoch) | 3.150.000 forward, 4.49s | 63.000 forward, 0.19s |

**Tiêu chí gradient check**: đạt khi lệch tương đối `< 1e-6` **hoặc** tuyệt đối `< 1e-8`.
Cần cả hai vì sai phân trung tâm có sàn nhiễu `ε/h ≈ 10⁻¹¹`.

## Phase 4 — Classification (XOR)

Sigmoid ở layer ra + Binary Cross Entropy + ngưỡng 0.5.

`L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]`. Ghép với sigmoid thì `δ = ŷ − y` —
gradient không bao giờ biến mất, khác hẳn MSE (khi sigmoid bão hoà, `δ < 1e-4`).

XOR không khả phân tuyến tính: perceptron kẹt ở 50%, mạng `[2→4→1]` đạt 100%.

## Phase 5 — Embedding

`cos(a,b) = (a·b)/(‖a‖·‖b‖)`. One-hot cho cosine = 0 giữa **mọi** cặp từ khác nhau —
nó không diễn đạt được "cat gần dog hơn car".

Học bằng skip-gram + negative sampling. **Phải dùng hai bảng embedding** (center và
context): với một bảng, cặp (cat, dog) không bao giờ đồng hiện nên bị lấy làm cặp âm
và bị đẩy xa nhau — đo ra `cos = −0.23`. Với hai bảng: `cos = 0.988`.

## Phase 6 — Mini Language Model

`tokenize → embedding → hidden → logits → softmax → cross entropy`

```
softmax(z)ᵢ = e^(zᵢ−max) / Σ e^(zⱼ−max)      trừ max để không tràn số
L = −log(p_đúng)                             perplexity = e^L
∂L/∂logits = p − y                           softmax + CE rút gọn còn một dòng
```

Model học đúng **phân bố tần suất**: ngữ liệu có "I like cats"×3, "dogs"×2,
"programming"×1 → dự đoán `0.5000 / 0.3331 / 0.1667`.

## Phase 7 — Attention

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

Query = "tôi cần gì", Key = "tôi nói về gì", Value = "nội dung của tôi".

Chia `√d` vì `q·k` là tổng `d` số hạng nên phương sai bằng `d`; không chia thì softmax
bão hoà (đo được: `d = 512` → trọng số lớn nhất `0.997` thay vì `0.349`).

Causal mask đặt score `= −∞` ở tam giác trên, vì `e^(−∞) = 0`.

## Phase 8 — Mini Transformer

```
Embedding → Positional Encoding → [LayerNorm → Self Attention → +residual
                                   LayerNorm → Feed Forward   → +residual] ×2
          → LayerNorm → Output head → softmax
```

Backprop chạy xuyên qua tất cả, gồm Jacobian của softmax và hai số hạng hiệu chỉnh
của LayerNorm. **Gradient check trên toàn bộ 4616 tham số: lệch lớn nhất `5.8e-11`.**

## Phase 9 — Reasoning Engine

Không dùng neural network. `Parse → Store → Find relevant → Apply rule → Calculate → Answer`.

Recursive descent parser có thứ tự ưu tiên và dấu ngoặc. Hỗ trợ `= > < + - * /`.
Suy luận bắc cầu nhiều tầng, trả về **đầy đủ các bước giải thích**, phát hiện tham
chiếu vòng tròn và chia cho 0.

Khác biệt lớn nhất so với mạng neural: engine **biết khi nào nó không biết** và nói thẳng.

## Phase 10 — Memory

`ShortTermMemory` (dung lượng cứng, FIFO) + `LongTermMemory` (phai theo giá trị).

```
score = 0.55·relevance + 0.20·importance + 0.15·recency + 0.10·frequency
recency = 0.5 ^ (giờ kể từ lần dùng cuối / halfLife)
```

Bốn quy tắc lọc khi `Store()`: bỏ câu không có từ khoá → củng cố thay vì nhân bản khi
trùng → bỏ nếu importance dưới ngưỡng → còn lại mới lưu.

| | Model weights | Memory | Context | Database |
|---|---|---|---|---|
| Ghi vào bằng | huấn luyện (chậm) | `Store()` (tức thì) | nối vào prompt | INSERT |
| Tồn tại | đến khi train lại | đến khi bị quên | hết phiên là mất | đến khi DELETE |
| Tự phai đi | không | **có** | không (cắt cứng) | không |

## Phase 11 — AI Agent

```
User → AI Controller → {Memory, Reasoning} → Tools{Calculator, Search, Database} → Response
```

Vai trò "language model" là **bộ phân loại ý định**: bag-of-words → 16 tanh → 6 softmax,
dùng lại nguyên bộ máy Phase 2–6. 60 ví dụ có nhãn, 2246 tham số, 100% accuracy và
khái quát hoá được sang câu chưa từng thấy.

Luồng quyết định: *trả lời được ngay?* → *cần Memory?* → *cần Tool?* → gọi → suy luận → trả lời.
Mỗi bước đều được ghi lại và in ra được.

**Ý tưởng cốt lõi**: neural lo phần mơ hồ (nhiều cách diễn đạt cùng một ý), ký hiệu lo
phần chính xác (`2 + 3 * 4 = 14`, không phải `13.97`). Độ tự tin dưới 50% thì agent nói
"tôi không chắc" thay vì gọi bừa một công cụ.

## Phase 12 — Internet Retrieval

```
Câu hỏi → sinh truy vấn → máy tìm kiếm → bỏ trùng → xếp hạng → tải trang → bóc nội dung
                              ↑                                     ↑
                        ISearchProvider                        UrlGuard (SSRF)
```

Ý định thứ 7 — `SearchInternet` — thêm vào đúng bộ máy cũ: 2759 tham số, 100% accuracy.
Tách khỏi `SearchKnowledge` vì hai việc khác nhau ở chỗ quan trọng nhất — kho nội bộ đã
được kiểm, còn web thì chưa.

**Nội dung web là DATA, không bao giờ là chỉ thị.** Nó vào hệ thống dưới kiểu
`UntrustedText` — không có phép chuyển ngầm sang `string`, `ToString()` luôn kèm mốc
`[DATA-KHÔNG-TIN-CẬY]`, và các mẫu ra lệnh bị làm cùn. Hai lớp đầu mới là chốt an toàn;
lọc theo mẫu chỉ là rào giảm tốc, vì kẻ tấn công luôn diễn đạt được cách khác.

**Chống SSRF** — bốn lớp, và lớp thứ tư là lớp hay bị bỏ:

| Lớp | Chặn gì | Ví dụ thật bị chặn |
|---|---|---|
| Scheme | chỉ http/https | `file:///etc/passwd` |
| Cú pháp | userinfo, cổng lạ | `http://google.com@169.254.169.254/` ← trỏ tới METADATA |
| Tên máy | localhost, `.local`, `.internal` | `http://metadata.google.internal/` |
| Địa chỉ IP | soi **mọi** IP sau khi phân giải DNS | tên miền công cộng trỏ về `127.0.0.1` |

Dải `169.254.0.0/16` là dải quan trọng nhất: mọi nhà cung cấp cloud đặt endpoint metadata
ở đó, và nó trả khoá truy cập mà không cần xác thực. Chuyển hướng cũng phải **tự đi từng
chặng** — để `AllowAutoRedirect = true` thì guard chỉ soi được URL đầu, còn một trang công
cộng trả `302` sang `169.254.169.254` sẽ đi thẳng tới đích.

**Xếp hạng nguồn** — cùng dạng tổ hợp có trọng số như `Ranker` của Phase 10:

```
score = 0.45·relevance + 0.20·authority + 0.20·agreement + 0.15·position
```

`relevance` nặng nhất vì nguồn uy tín mà nói chuyện khác thì vô dụng; `position` nhẹ nhất
vì nó chính là ý kiến của máy tìm kiếm — thứ ta đang cố không phụ thuộc vào. Đây là mức
**đáng đọc trước**, không phải mức **đáng tin**.

Bỏ trùng không phải để cho gọn: mười dòng cùng trỏ một trang trông như "mười nguồn cùng
khẳng định", trong khi chỉ có MỘT. Phase 14 đếm số nguồn độc lập để tính độ tự tin, nên
con số đó phải đúng.

**Ranh giới không được phá**: nội dung Internet **không** tự động thành ký ức hay dữ liệu
huấn luyện. `SearchService` trả về thông tin và không ghi gì vào memory; cache có TTL vì
cache không hết hạn thì nó biến thành "kiến thức". Có unit test giữ đúng ranh giới này.

Toàn bộ demo và 237 test chạy **hoàn toàn ngoại tuyến** — `StaticSearchProvider` cho kết
quả tất định, không cần mạng, không cần API key. Bộ test phụ thuộc Internet là bộ test
hỏng: hôm nay xanh, mai đỏ, mà chẳng ai sửa gì.

## Phase 13 — Knowledge + Memory + Evidence

Agent trả lời **có căn cứ**, và căn cứ nào cũng truy được về nguồn.

```
Câu hỏi → (+ ngữ cảnh nếu câu hỏi không tự đứng được)
   ↓
Knowledge · Memory · Context · Reasoning · Internet     ← 5 kênh
   ↓
bỏ trùng CHÉO KÊNH → xếp hạng → EvidenceSet
```

Mỗi mẩu `Evidence` mang `Content`, `Source`, `Url`, `RetrievedAt`, `Relevance`,
`Reliability`. **`Relevance` và `Reliability` phải tách rời** — gộp thành một con số thì
"blog đúng chủ đề" và "tài liệu NIST nói chuyện khác" trông giống nhau, và agent mất khả
năng nói *"tôi tìm đúng chỗ nhưng nguồn không đủ chắc"*.

Mức tin cậy gắn theo **kênh**, không theo nội dung:

| Kênh | Trust | Thành kiến thức được? |
|---|---|---|
| Knowledge | `Verified` | đã là rồi |
| Memory / Context | `UserProvided` | được |
| Reasoning | `Derived` | **không** |
| Internet | `Untrusted` | **không bao giờ** |

`Derived` bị chặn vì nó là đầu ra do chính agent sinh ra: cho nó tự thành kiến thức là mở
vòng tự khuếch đại — suy ra → lưu → lấy chính nó làm căn cứ → suy ra tiếp, không bước nào
đối chiếu lại thực tế. Cùng vòng lặp đó, nếu ăn vào trọng số, là lý do Phase 15 không
được huấn luyện từ câu trả lời tự sinh.

Chốt thực thi là `KnowledgeStore.TryPromote` — một hàm **từ chối tường minh** kèm lý do,
chứ không phải "chỉ là không viết hàm nhận web". Cái không tồn tại thì không test được;
một hàm trả lý do thì viết được test khẳng định luật vẫn còn đó, và test sẽ đỏ ngay nếu
sau này có ai nới luật. Kể cả `nist.gov` với `reliability = 1.0` cũng bị chặn.

**Bỏ trùng chéo kênh** khó hơn bỏ trùng URL của Phase 12: cùng một thông tin xuất hiện
dưới những hình thức không giống nhau chút nào. Khi gộp thì mẩu **đã kiểm thắng**, và
`Agreement` đếm **kênh** chứ không đếm mẩu — ba trang web cùng nói một điều vẫn chỉ là
*một* tiếng nói.

```
score = 0.45·relevance + 0.30·reliability + 0.15·agreement + 0.10·freshness
```

`relevance` nặng nhất vì nguồn uy tín mà lạc đề thì vô dụng. `reliability` chỉ 0.30 —
nặng hơn nữa thì kênh knowledge thắng tuyệt đối và agent bỏ qua thông tin mới đúng lúc
cần nhất. `freshness` nhẹ nhất vì **mới không có nghĩa là đúng**.

Tìm trong kho kiến thức đo **cả hai chiều**, và đó là điều cần thiết:

```
score = (body + 0.5·topic + 1.0·topicCoverage) / 2.5
```

Chỉ đo một chiều thì sai ở cả hai đầu: hỏi *"softmax hoạt động thế nào"* (4 từ) khiến tài
liệu **Softmax** chỉ khớp `1/4`, còn hỏi *"cách nấu phở bò Hà Nội"* lại lôi về tài liệu
**SSRF** vì nó tình cờ chứa "cách" và "nội" (trong "nội bộ"). `topicCoverage` cứu trường
hợp đầu và dìm trường hợp sau.

Ngữ cảnh hội thoại cũng vậy — điều kiện bổ sung ngữ cảnh **không phải đếm số từ khoá**:

```
"còn nó"             2 từ khoá, KHÔNG có chủ đề nào  → cần bổ sung
"overfitting là gì"  1 từ khoá, chủ đề rất rõ        → để nguyên
```

Đếm từ cho kết luận ngược ở cả hai câu. Thước đo đúng là có từ nào **mang chủ đề** hay
không, chứ không phải có bao nhiêu từ.

Kết quả đo được: hỏi *"tìm trên mạng thông tin về SSRF"* thì Phase 12 luôn kèm cảnh báo
"chưa kiểm chứng", còn Phase 13 thấy kho nội bộ cũng có tài liệu về SSRF → hai kênh độc
lập xác nhận → xếp nguồn đã kiểm lên trước và **không cảnh báo nữa**. Cùng câu hỏi, cùng
Internet; khác biệt duy nhất là biết nguồn nào đáng tin hơn — và nói ra được vì sao.

## Phase 14 — Verification + Answer Generation

Agent không còn "sinh ra một câu trả lời". Nó đi qua dây chuyền, và ba bước cuối **không
đảo được**:

```
Câu hỏi → Retrieve → Collect Evidence → Reason → Verify → Generate → Validate
```

*Verify trước Generate* — vì kết luận quyết định VIẾT GÌ. Sinh câu trả lời trước rồi mới
kiểm thì đã có một câu dứt khoát tồn tại, và sức ép sẽ là sửa nó cho đỡ dứt khoát, thay vì
ngay từ đầu không viết nó ra. *Validate sau Generate* — vì nó kiểm chính văn bản sắp phát
ra, không kiểm ý định của bộ sinh.

Bốn kết luận: `Answered` / `Uncertain` / `Conflicted` / `Insufficient`. Căn cứ nói ngược
nhau thì trình bày **cả hai phía**, không chọn bên.

```
base = 0.35·reliability + 0.25·support + 0.20·agreement + 0.20·relevance
     × (1 − 0.6·severity)   nếu có mâu thuẫn
     × 0.65                 nếu không có nguồn nào đã kiểm
     × 0.80                 nếu chủ yếu là suy ra, không phải sự thật
```

Nhân chứ không trừ, vì phạt phải theo tỉ lệ: trừ một lượng cố định thì câu vốn yếu (0.2)
bị trừ xuống âm, còn câu rất mạnh (0.95) gần như không suy suyển — ngược hẳn điều ta muốn.

`AnswerValidator` được viết **độc lập với bộ sinh**: nó chỉ nhận văn bản cùng danh sách căn
cứ rồi tự kiểm, nên bắt được cả một bộ sinh viết sau này. Không qua được thì câu trả lời bị
**hạ xuống "không đủ căn cứ" và viết lại**, chứ không phát ra.

Phase 14 lưu **tóm tắt suy luận** — quyết định, ngưỡng nào đã kích hoạt, id căn cứ — chứ
không lưu và không in dòng suy nghĩ nội bộ. Người dùng cần biết kết luận dựa trên cái gì;
độc thoại thô thì dài, lẫn cả những nhánh đã bị loại, và đọc nó dễ dẫn tới tin vào một lập
luận mà chính hệ thống đã bỏ đi.

## Phase 15 — Feedback + Learning

```
Phản hồi → Ca lỗi → Ứng viên → KIỂM (6 cửa) → Phiên bản tập dữ liệu mới
```

**Không có mũi tên nào đi từ phản hồi thẳng tới trọng số.** Đầu ra của cả phase này là DỮ
LIỆU. `LearningPipeline` không giữ tham chiếu nào tới network hay optimizer, và có unit
test chạy cả vòng rồi khẳng định bộ phân loại trả lời **giống hệt đến 12 chữ số thập phân**.

**Sáu loại ca lỗi, và chỉ một loại train được**:

| Loại | Sửa ở đâu |
|---|---|
| `WrongIntent` | bộ phân loại ý định — **loại duy nhất có trọng số để sửa** |
| `Overconfident` | ngưỡng/công thức độ tự tin |
| `WrongFact` | kho kiến thức hoặc khâu chọn căn cứ |
| `MissedAnswer` | thiếu kiến thức hoặc thiếu kênh truy hồi |
| `Incomplete` / `Unhelpful` | số căn cứ được dẫn, khâu diễn đạt |

Ý định sai được xét **trước tất cả**, vì nó là lỗi thượng nguồn: hiểu sai câu hỏi thì mọi
khâu sau đều chạy đúng trên một câu hỏi SAI. Xét theo thứ tự khác thì ca đó bị dán nhãn
`WrongFact` và sẽ được đem đi sửa kho kiến thức — trong khi kho kiến thức không có lỗi gì.

**Hai cửa TỪ CHỐI, viết thành hàm trả về lý do** (cùng lý lẽ với `TryPromote` của Phase 13:
cái không tồn tại thì không test được):

* **nhãn do chính model dự đoán** — kể cả khi người dùng đã bấm "đúng". Họ chấm *câu trả
  lời*, không xác nhận *cái nhãn ý định* bên trong. Lấy nhãn model tự đoán dạy lại chính nó
  là đóng một vòng không có thực tế nào đi vào — và accuracy trên tập tự gán nhãn chỉ có
  thể tăng suốt quá trình đó.
* **nội dung Internet** — không ngoại lệ, kể cả `nist.gov` với `reliability = 1.0`.

Sáu cửa kiểm chất lượng: nguồn nhãn → nhãn tồn tại → văn bản rỗng/quá dài → dấu hiệu ra
lệnh → trùng → **xung đột nhãn**. Cửa cuối là cửa đáng nói nhất: cùng một câu với nhãn khác
không phải lỗi gõ, mà là hai người không đồng ý câu đó nghĩa là gì. Nhận cả hai thì model
học được đúng một điều — rằng câu đó là 50/50 — loss không bao giờ xuống, mà chẳng có gì
báo là dữ liệu sai. Nó cũng là đường tấn công thẳng nhất, nên hệ thống **từ chối và giao
người quyết**. Kiểm phải **tuần tự**: hai ứng viên xung đột với *nhau* mà kiểm độc lập thì
cả hai cùng lọt.

Tập dữ liệu **append-only, có vân tay** (SHA-256, 12 hex). Vân tay phải xác định được
*trọng số*, không chỉ *nội dung* — hai tập cùng ví dụ nhưng khác thứ tự sẽ train ra hai
model khác nhau, nên `DatasetVersion.Create` sắp ví dụ về thứ tự chuẩn trước khi băm.

Một hệ quả đáng biết: `"tính 2 cộng 3"` và `"tính 2 cộng 4"` là **một ví dụ**. Khoá chuẩn
hoá dùng chính bộ tách từ mà bag-of-words dùng, và bộ đó bỏ token một ký tự — hai câu ấy
encode thành cùng một vector, model không phân biệt được chúng.

## Phase 16 — Controlled Retraining

```
Tập dữ liệu → Chia tất định → Train → Đánh giá → So sánh → Cửa triển khai → Sổ model
```

**Kết quả đo được, và nó đảo lại một con số cũ của chính dự án này:**

| | Phase 11 công bố | Phase 16 đo |
|---|---|---|
| Accuracy trên tập train | 100% | 100% |
| Accuracy trên **dữ liệu chưa thấy** | *chưa từng đo* | **58.8%** (10/17) |

Cùng model, cùng dữ liệu. Khác nhau chỉ ở chỗ **đo ở đâu**. Trong 17 câu đánh giá, 4 câu
không có lấy một từ nào từng xuất hiện khi train và 3 trong số đó sai — bỏ chúng ra thì còn
69.2%. Đó là phần do **thiếu dữ liệu**, không phải do model: với 70 ví dụ, giữ lại 20% để
đánh giá là lấy đi ví dụ *duy nhất* chứa những từ ấy.

**Tái lập được, bit-by-bit**: cùng tập dữ liệu + cùng cấu hình → cùng vân tay trọng số
(`2dc2a960e4ac` qua mọi lần chạy). Ba nguồn bất định phải bịt hết:

1. trọng số khởi tạo → seed nằm **trong** cấu hình và được lưu cùng model
2. thứ tự từ điển → sắp `Ordinal`, không duyệt `HashSet` (chỉ số của từ quyết định vị trí trọng số)
3. thứ tự ví dụ → tập dữ liệu Phase 15 đã sắp chuẩn — cộng số thực dấu phẩy động không có
   tính kết hợp, lệch 1e-16 mỗi bước qua 3000 epoch là hai model khác nhau

Và **không dùng `string.GetHashCode()`** ở bất cứ đâu: .NET ngẫu nhiên hoá nó theo từng
tiến trình, nên cách chia dữ liệu sẽ đổi mỗi lần khởi động lại chương trình. Dùng FNV-1a.

**Sáu điều kiện của cửa triển khai**, mỗi điều kiện trả về con số thật kèm lý do:

```
1. có dữ liệu để đánh giá          không đo được thì không triển khai
2. mọi nhãn đều có mặt khi train    model không học được nhãn nó chưa thấy
3. accuracy ≥ sàn (0.50)            chặn thảm hoạ; đoán bừa là 1/7 ≈ 14%
4. macro F1 không thụt lùi          KHÔNG dùng accuracy — xem dưới
5. KHÔNG ví dụ nào từ đúng thành sai
6. bộ vàng đúng 100%
```

Điều kiện 4 dùng macro F1 vì accuracy đánh trọng số theo số mẫu: một model **bỏ hẳn** ý
định hiếm nhất vẫn có thể có accuracy cao hơn. Với người dùng thì đó không phải "sai 8%",
mà là "chức năng tra Internet đã chết".

Điều kiện 5 không cho đánh đổi, và lý do nằm ở một trường hợp mà số trung bình che mất: một
model **sửa 3 câu và làm hỏng 3 câu** có chênh lệch accuracy bằng 0 — trông như không đổi
gì. Nên `ModelComparison` so **từng ví dụ**, không so hiệu số.

Model tệ hơn thì **không thay được model đang chạy**: cấu hình 1 epoch cho ra `m3` với
accuracy 23.5% và bộ vàng 3/7 → bị 4 điều kiện chặn cùng lúc, model đang chạy vẫn là `m2`.
`m3` không bị xoá — model bị từ chối là dữ liệu quý, nó cho biết cấu hình nào đã thử và
hỏng ra sao.

**Vòng tròn khép lại**: `"overfitting là gì"` → `m1` đoán `Calculate` (57.2%, sai) → người
dùng sửa → Phase 15 duyệt vào `v2` → `m2` đoán `SearchKnowledge` (99.8%), macro F1 +0.044,
sửa được 1 câu, làm hỏng 0 → **triển khai**. Quay lui về `m1` thì câu đó lại ra `Calculate`
— hành vi trở lại thật, vì phiên bản model lưu cả trọng số.

**Một lỗi thật đã phải sửa ở đây.** Cách chia tất định băm câu `"overfitting là gì"` vào
tập **đánh giá** (bucket 70 < ngưỡng 200), nên model mới không hề học nó và vẫn đoán sai y
như cũ. Từ "overfitting" chỉ xuất hiện trong đúng ví dụ ấy nên không có đường nào khác để
học — và vì cách chia là *tất định*, nó sẽ nằm đó **mãi mãi**: người dùng sửa bao nhiêu lần
cũng vô ích. Nên ví dụ đến từ lời sửa của người dùng **luôn vào tập train**, không bốc thăm.
Giá phải trả được nói thẳng: chúng không còn được đo trên dữ liệu chưa thấy. Bù lại bằng
hai chốt độc lập — **bộ vàng** (7 câu viết tay, một câu mỗi ý định, có test khẳng định nó
không trùng tập huấn luyện) và điều kiện không làm hỏng thứ model cũ đang làm đúng.

---

## Ai.Providers — nối LLM ngoài vào KHÂU DIỄN ĐẠT

Lớp này **không** thay bộ máy của dự án. Nó thay đúng một khâu: viết lại câu trả lời cho
gọn. Căn cứ, độ tự tin, kết luận và việc truy nguồn vẫn hoàn toàn của Phase 13–14.

```
Câu hỏi → Evidence (P13) → Reason → Verify (P14)
                                       ↓
                        LlmAnswerGenerator  ← Router → {Ollama, endpoint kiểu OpenAI}
                                       ↓
                        AnswerValidator (P14)  ← chốt chặn KHÔNG đổi
                             ↓ đạt          ↓ không đạt
                        văn bản LLM      văn bản khuôn mẫu (như cũ)
```

Chỗ cắm là `IAnswerGenerator` — chính giao diện mà Phase 14 đã tách ra, và chú thích của
`AnswerValidator` đã dự đoán trước ngày này: *"bộ sinh sẽ được sửa, sẽ được thay… lúc đó
tính chất kia biến mất lặng lẽ, không có gì báo"*. Validator được viết độc lập với bộ sinh
chính vì vậy, nên nó bắt được cả một bộ sinh viết sau — kể cả một LLM.

**Bốn giới hạn cứng của LLM trong kiến trúc này:**

| LLM KHÔNG được | Vì sao |
|---|---|
| rút khẳng định | độ tự tin, nhãn sự-thật/suy-ra và id căn cứ đều dựa vào đó |
| quyết định có trả lời hay không | `Insufficient` và `Conflicted` không được đưa cho nó diễn đạt |
| gắn nguồn | trích dẫn sai còn tệ hơn không trích dẫn; hệ thống tự gắn |
| phát văn bản chưa qua kiểm | không qua `AnswerValidator` thì dùng văn khuôn mẫu |

Giới hạn thứ tư làm cho việc bật LLM **không thể làm câu trả lời tệ hơn**: tệ nhất là quay
về đúng hành vi cũ. Có test cho đúng điều đó — LLM bịa "mọi tường lửa thương mại đều đã
chặn kiểu tấn công này từ 2019" thì câu đó bị chặn, kết luận vẫn `Answered`, và người dùng
vẫn nhận được câu trả lời có nguồn.

**Khoá API không bao giờ nằm trong source hay file cấu hình.** Cấu hình chỉ ghi TÊN biến
môi trường; `SecretResolver` là chỗ duy nhất đọc giá trị, và nó không có hàm nào in ra bí
mật. Nếu file cấu hình chứa chuỗi trông như khoá thật (`sk-…`, `AIza…`, `Bearer …`) thì cấu
hình đó **bị từ chối toàn bộ** — vì file cấu hình bị commit, còn biến môi trường thì không.

Prompt gửi đi cũng đi qua bộ lọc: khoá, JWT, chuỗi kết nối, mật khẩu bị thay bằng
`[ĐÃ-LỌC]`. Cần thiết vì căn cứ có thể chứa ký ức người dùng — ai đó từng bảo agent "nhớ
token của tôi là…" thì đó là một mẩu ký ức hợp lệ, và nó sẽ lặng lẽ rời khỏi máy.

```bash
dotnet run --project src/Ai.Chat    # rồi gõ:  /provider   /llm on
```

Mặc định không có file cấu hình thì hệ thống chỉ thử **Ollama tại máy** — không có Ollama
thì `/llm on` từ chối kèm lý do, và màn hình chạy y như trước. Xem
[aiproviders.sample.json](src/Ai.Providers/aiproviders.sample.json) để thêm một
endpoint kiểu OpenAI (chỉ cần một mục cấu hình, không sửa code).

---

## Ai.Providers.OpenAi — OpenAI qua SDK chính thức

Khâu diễn đạt ở trên cho LLM **viết lại** căn cứ đã có. Lớp này mở thêm một đường **khác
hẳn**: để model **gọi công cụ của MiniAI** rồi tự trả lời. Hai đường tồn tại song song vì
chúng trả lời hai câu hỏi khác nhau:

| Đường | Lệnh | Trả lời câu hỏi | Bảo đảm |
|---|---|---|---|
| Kiểm chứng (P13–14) | gõ thẳng câu hỏi | "điều này có đúng không" | mọi câu truy được về căn cứ |
| Điều phối (mới) | `/hoi <câu hỏi>` | "làm hộ tôi việc này" | câu trả lời phải DÙNG kết quả công cụ |

```
Câu hỏi
   ↓
TaskClassifier       luật từ khoá — KHÔNG gọi model để hỏi "câu này loại gì"
   ↓
Modes.Policy         chế độ: auto | fast | reasoning | research | coding
   ↓
ContextBuilder       nhớ LIÊN QUAN + vài lượt gần nhất, có hạn ký tự
   ↓
AiProviderRouter     chọn provider/model, thử lại, lui provider khi hỏng
   ↓
ToolLoop  ⇄  ToolRegistry (4 chốt)     ← model ĐỀ NGHỊ, MiniAI QUYẾT ĐỊNH
   ↓
ResponseValidator    không đạt → viết lại, TỐI ĐA 2 lần rồi dừng
   ↓
ChatResult
```

**Gói `OpenAI` nằm riêng một project.** `Ai.Providers` vẫn **không có PackageReference
nào** — đó là tính chất của nó từ đầu, và một SDK của riêng một hãng không được bắt cả
Ollama tại máy kéo theo. Phần còn lại của MiniAI chỉ thấy `IAiProvider`; không có kiểu nào
của OpenAI rời khỏi thư mục `src/Ai.Providers.OpenAi`.

**Bốn chốt trước khi một công cụ được chạy** (`ToolRegistry`), theo thứ tự:

| # | Chốt | Chặn được gì |
|---|---|---|
| 1 | có đăng ký không | model bịa ra tên công cụ |
| 2 | có bị chặn không | công cụ đã tắt nhưng vẫn bị gọi |
| 3 | tham số có đọc được không | JSON sai khuôn, sai kiểu |
| 4 | có được duyệt không | công cụ ghi/xoá chưa được người dùng cho phép |

Mặc định là **từ chối**: một công cụ quên khai quyền là công cụ *không chạy được*, chứ
không phải công cụ chạy tự do. Khai báo gửi cho model (`AiToolDefinition`) **không mang
theo code** — delegate nằm lại trong sổ đăng ký, không bao giờ rời khỏi máy.

**Về phiên bản SDK.** Gói `OpenAI` 2.14.0 đặt tên **khác phần lớn ví dụ trên mạng**, và
hợp đồng Responses trong đó vẫn được đánh dấu **thử nghiệm** (`OPENAI001` — mặc định là
lỗi biên dịch, được tắt ở đúng project này):

| Ví dụ cũ hay dùng | Thực tế 2.14.0 |
|---|---|
| `OpenAIResponseClient` | `ResponsesClient` |
| `ResponseCreationOptions` | `CreateResponseOptions` |
| `OpenAIResponse` | `ResponseResult` |

**Không có tên model nào bị ghi cứng.** `TaskModels` trong cấu hình chọn model theo từng
loại việc (`Chat`, `Reasoning`, `Research`, `Coding`, `Vision`, `Embedding`,
`ImageGeneration`); bỏ trống thì dùng `DefaultModel`. Khai đúng một model vẫn chạy được mọi
loại việc.

```bash
$env:OPENAI_API_KEY = "..."        # khoá KHÔNG vào file cấu hình
dotnet run --project src/Ai.Chat   # rồi gõ:  /che research   /hoi <câu hỏi>
```

**Mặc định tắt, và mặc định không gửi gì ra ngoài.** Provider OpenAI trong file mẫu có
`"Enabled": false`; `AllowProviderSideStorage` cũng mặc định `false`, nên prompt và câu trả
lời không được lưu lại ở phía nhà cung cấp trừ khi bật tường minh. `MaxOutputTokens` trong
cấu hình **cắt** giá trị mà chỗ gọi yêu cầu — hạn mức chi phí là của người vận hành, không
phải của người viết chỗ gọi.

**Chưa có trong dự án này:** module Health Coach (dinh dưỡng, BMR/TDEE) và provider sinh
ảnh. Khuôn mẫu mà chúng cần — *LLM rút cấu trúc → dịch vụ tất định tính toán* — đã có ở
`StructuredOutput` và được kiểm bằng test; cắm dịch vụ vào là đủ, không phải sửa kiến trúc.
