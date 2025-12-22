# BÁO CÁO THỰC NGHIỆM PHÂN TÍCH CẢM XÚC TRÊN DỮ LIỆU VĂN BẢN

---

## 1. MỤC ĐÍCH VÀ PHẠM VI

Báo cáo này trình bày một chuỗi thực nghiệm nhằm phân loại cảm xúc nhị phân (tích cực/tiêu cực) từ dữ liệu đánh giá dạng văn bản. Thay vì sử dụng văn phong quá ngắn gọn hay công thức, nội dung được viết lại theo lối giải thích từng phần, đi kèm các quyết định kỹ thuật, cách tổ chức pipeline, và lý do lựa chọn. Kết quả cuối cùng là so sánh các mô hình đại diện cho hai hướng tiếp cận: học máy truyền thống (Logistic Regression, LinearSVC) và học sâu (BiLSTM, DistilBERT). Các bảng số liệu và sơ đồ vẫn giữ nguyên để thuận tiện cho việc đối chiếu.

### 1.1. Bài toán và ứng dụng
Phân tích cảm xúc là một bài toán cơ bản trong xử lý ngôn ngữ tự nhiên, giúp xác định khuynh hướng cảm xúc trong văn bản. Ở mức độ ứng dụng, kỹ thuật này thường được dùng để nắm bắt phản hồi của người dùng đối với sản phẩm, dịch vụ, thương hiệu hoặc nội dung; ví dụ như đánh giá trên nền tảng thương mại điện tử, bình luận trong mạng xã hội, hay nhận xét về phim và nhà hàng.

### 1.2. Mô hình so sánh
Bốn mô hình được triển khai theo cùng quy trình tiền xử lý và đánh giá:
1. Logistic Regression (baseline theo hướng tuyến tính)
2. LinearSVC (SVM với nhân tuyến tính)
3. BiLSTM (mạng tái phát hai chiều)
4. DistilBERT (mô hình transformer rút gọn từ BERT)

### 1.3. Dữ liệu
- Tổng số mẫu: 69.995 đánh giá
- Nhãn nhị phân cân bằng: Negative ~50%, Positive ~50%
- Tách tập theo tỉ lệ 80/20, có đảm bảo phân tầng theo nhãn
- Train sau tăng cường đạt 59.995 mẫu; test giữ nguyên 10.000 mẫu để đảm bảo đánh giá khách quan

---

## 2. QUY TRÌNH DỮ LIỆU VÀ TIỀN XỬ LÝ

### 2.1. Dòng chảy dữ liệu tổng quát

```
Raw Dataset (69,995 samples)
         ↓
[Stratified Split: 80/20]
         ↓
    ┌────────┴────────┐
    ↓                 ↓
Train (55,996)   Test (10,000)
    ↓                 ↓
[Back Translation]   [No Augmentation]
    ↓                 ↓
Augmented (59,995)   Original (10,000)
    ↓                 ↓
[Encoding/Vectorization]
    ↓                 ↓
Train/Val Split      Test Set
(85/15)              ↓
    ↓                 ↓
Model Training    Final Evaluation
```

### 2.2. Tăng cường dữ liệu cho tập huấn luyện
Phương pháp back translation được sử dụng, qua các ngôn ngữ trung gian như tiếng Đức và tiếng Pháp, với tỷ lệ tăng cường 50%. Quyết định không tăng cường tập kiểm thử nhằm tránh hiện tượng rò rỉ dữ liệu và giữ cho đánh giá sát với tình huống triển khai thực tế.

### 2.3. Tiền xử lý theo nhóm mô hình
- Nhóm Logistic/SVM: TF‑IDF với n‑grams (1, 2), giới hạn 10.000 đặc trưng, chuyển chữ thường, loại bỏ stopwords.
- BiLSTM: tokenization theo từ, xây dựng từ điển với kích thước tối đa ~50.002, tần suất tối thiểu 2, độ dài chuỗi 256, padding phía sau.
- DistilBERT: dùng tokenizer `distilbert-base-uncased`, độ dài tối đa 256, truncation và padding theo `max_length`.

---

## 3. MÔ HÌNH VÀ CÁCH TIẾP CẬN

### 3.1. Logistic Regression
Mô hình tuyến tính dùng hàm sigmoid để phân loại nhị phân trên biểu diễn TF‑IDF. Lợi thế là tốc độ huấn luyện, yêu cầu tài nguyên thấp, và khả năng làm baseline rõ ràng.

Kiến trúc tổng quát:
```
Input (TF-IDF vectors) → Logistic Function → Output (2 classes)
```

Thông số chính: C=1.0, `lbfgs`, tối đa 1.000 vòng lặp, 10.000 đặc trưng.

---

### 3.2. LinearSVC (SVM tuyến tính)
Mô hình SVM tìm siêu phẳng tối ưu để phân tách hai lớp với biên lớn nhất. Với dữ liệu văn bản có chiều cao, LinearSVC hoạt động ổn định và khá hiệu quả.

Kiến trúc tổng quát:
```
Input (TF-IDF vectors) → SVM Decision Boundary → Output (2 classes)
```

Thông số chính: C=1.0, kernel tuyến tính, tối đa 1.000 vòng lặp, 10.000 đặc trưng.

---

### 3.3. BiLSTM
BiLSTM xử lý chuỗi từ cả hai hướng để nắm bắt ngữ cảnh trước–sau trong cùng một biểu diễn. Bên cạnh kiến trúc, phần quan trọng là pipeline xây dựng dữ liệu và vòng huấn luyện, vì chúng quyết định nhiều đến khả năng tổng quát.

Kiến trúc tham chiếu:
```
Input Text
    ↓
Embedding Layer (256-dim)
    ↓
Bidirectional LSTM (256-dim × 2)
    ↓
Multi-Pooling (Max + Mean + Last Hidden)
    ↓
Dropout (0.35)
    ↓
Fully Connected Layer
    ↓
Output (2 classes)
```

Thông số: embed_dim=256, hidden_dim=256 (hai chiều → 512), vocab≈50K, max_len=256, dropout=0.35, batch=128, lr=5e‑4, epochs tối đa 40 (early stopping quanh 34), Adam với weight decay 1e‑4, tổng tham số ~13.9M.

Pipeline xây dựng mô hình BiLSTM (chi tiết):
1. Chuẩn bị dữ liệu và từ điển: lọc theo tần suất tối thiểu, cố định `max_len`, ánh xạ từ → chỉ số; lưu `word2idx`/`idx2word` để phục vụ suy luận.
2. Mã hóa chuỗi: tokenization theo từ, cắt/tràn về 256, padding phía sau; tạo `attention_mask` (nếu cần) để phân biệt vùng pad với vùng thực.
3. DataLoader: thiết kế collate function chuẩn hoá batch (tensor hóa chuỗi, nhãn), đảm bảo các batch có kích thước ổn định; cân nhắc `pin_memory` khi dùng GPU.
4. Khởi tạo embedding: ngẫu nhiên với phân phối đều, hoặc nạp tiền huấn luyện (GloVe/FastText) nếu có; thiết lập `padding_idx` để không tham gia cập nhật trọng số.
5. Mô hình hai chiều: LSTM forward/backward, ghép đặc trưng; pooling đa dạng (max/mean/last hidden) để tận dụng thông tin vị trí và tổng thể.
6. Tối ưu hóa: dùng Adam, weight decay để giảm overfit; áp dụng gradient clipping khi chuỗi dài; cân nhắc label smoothing nhẹ.
7. Lịch học (scheduler) và dừng sớm: theo dõi F1/accuracy trên tập validation, giảm học suất nếu metric chững; dừng sớm khi không cải thiện sau N epoch.
8. Ghi nhận và checkpoint: lưu `best_model.pt` theo metric chính; log loss/metric theo epoch; cố định seed để tăng tính tái lặp.
9. Suy luận và triển khai: chuẩn hoá đầu vào giống huấn luyện, lấy xác suất/nhãn; xuất thêm độ tin cậy để phục vụ phân tích.

Nhận xét thực nghiệm: hiệu năng phụ thuộc mạnh vào chất lượng từ điển và biểu diễn embedding; khi không dùng embedding tiền huấn luyện hoặc không có cơ chế attention, BiLSTM có thể kém hơn các baseline dựa trên TF‑IDF ở bài toán nhị phân đơn giản.

---

### 3.4. DistilBERT
DistilBERT là phiên bản rút gọn của BERT, giữ lại phần lớn năng lực ngữ nghĩa nhưng nhẹ hơn và nhanh hơn. Điểm khác biệt then chốt nằm ở pipeline fine‑tuning: cách tokenizer hoạt động, tổ chức batch, lịch học, và chiến lược đóng/mở các lớp.

Kiến trúc tham chiếu:
```
Input Text
    ↓
DistilBERT Tokenizer
    ↓
DistilBERT Transformer Layers (6 layers)
    - Multi-Head Self-Attention
    - Feed-Forward Networks
    ↓
[CLS] Token Representation
    ↓
Classification Head
    ↓
Output (2 classes)
```

Thông số: `distilbert-base-uncased`, 6 lớp transformer, hidden size=768, 12 đầu attention, max_len=256, batch=32, lr=1e‑5, epochs≈10, warmup ratio=0.1, tham số tiền huấn luyện ~66M.

Pipeline fine‑tuning DistilBERT (chi tiết):
1. Tokenization: sử dụng `fast` tokenizer để sinh `input_ids`, `attention_mask` cố định độ dài; đảm bảo xử lý trường hợp chuỗi quá dài bằng truncation nhất quán.
2. Data collator: tạo batch với padding động hoặc cố định; khi cố định `max_length`, đảm bảo hiệu quả trên GPU và ổn định về hình dạng tensor.
3. Đầu phân loại: thêm một head tuyến tính trên biểu diễn `[CLS]` hoặc pooling lớp cuối; cân nhắc dropout trước head để tăng regularization.
4. Chiến lược đóng/mở lớp: có thể bắt đầu với việc chỉ fine‑tune head, sau đó mở dần nhiều lớp (gradual unfreezing); dùng learning rate phân biệt (discriminative LR) cho backbone và head.
5. Lịch học: áp dụng warmup (0.1 tổng bước), sau đó dùng linear decay; lựa chọn AdamW để kết hợp tốt với weight decay.
6. Theo dõi và checkpoint: tính các metric (accuracy, F1 macro) trên validation sau mỗi epoch/step; lưu checkpoint tốt nhất và cấu hình tokenizer kèm theo.
7. Suy luận: pipeline giống huấn luyện, xuất xác suất/nhãn; tối ưu hoá tốc độ bằng `fp16` nếu hạ tầng cho phép.

Nhận xét thực nghiệm: nhờ tiền huấn luyện trên kho dữ liệu lớn, DistilBERT thường đạt hiệu năng cao, đặc biệt khi có đủ tài nguyên và thời gian fine‑tuning. Chi phí tính toán và bộ nhớ lớn hơn là đánh đổi cần cân nhắc khi đưa vào môi trường sản xuất.

---

## 4. KẾT QUẢ THỰC NGHIỆM

### 4.1. Bảng so sánh tổng quan

| Mô Hình | Val Accuracy | Val F1 | **Test Accuracy** | **Test F1** | Training Time | Model Size | Params |
|---------|--------------|--------|-------------------|-------------|---------------|------------|--------|
| **Logistic Regression** | 89.33% | 0.8933 | **90.45%** | **0.9045** | ~1-2 phút | <1 MB | ~10K |
| **LinearSVC** | 89.12% | 0.8912 | **89.22%** | **0.8922** | ~2-3 phút | <1 MB | ~10K |
| **BiLSTM** | 89.77% | 0.8977 | **88.33%** | **0.8833** | ~30-40 phút | ~56 MB | 13.9M |
| **DistilBERT** | 93.46% | 0.9346 | **91.67%** | **0.9167** | ~60-90 phút | ~268 MB | 66M |

### 4.2. Biểu đồ so sánh

Test F1 Score
```
DistilBERT  ████████████████████████████████████████ 0.9167 (91.67%)
Logistic    █████████████████████████████████████    0.9045 (90.45%)
SVM         ████████████████████████████████████     0.8922 (89.22%)
BiLSTM      ███████████████████████████████████      0.8833 (88.33%)
```

Model Complexity (Parameters)
```
DistilBERT  ████████████████████████████████████████ 66M
BiLSTM      █████████                                13.9M
Logistic    █                                        10K
SVM         █                                        10K
```

Training Time
```
DistilBERT  ████████████████████████████████████████ 60-90 min
BiLSTM      ███████████████████                      30-40 min
SVM         ██                                       2-3 min
Logistic    █                                        1-2 min
```

### 4.3. Phân tích theo mô hình
- Logistic Regression: F1 test 0.9045; tốc độ rất nhanh; cân bằng precision/recall; phù hợp cho triển khai nhanh.
- LinearSVC: hiệu năng sát Logistic; ổn định trên dữ liệu chiều cao; thích hợp khi cần biên phân tách rõ.
- BiLSTM: F1 test 0.8833; khoảng cách val‑test nhỏ; có thể cải thiện bằng embedding tiền huấn luyện, cơ chế attention, hoặc điều chỉnh kiến trúc/pooling.
- DistilBERT: F1 test 0.9167; hiệu năng cao nhất; chi phí tài nguyên và thời gian huấn luyện lớn hơn.

---

## 5. PHÂN TÍCH CHÊNH LỆCH, ĐỘ PHỨC TẠP VÀ THỜI GIAN

### 5.1. Val vs Test

| Mô Hình | Val F1 | Test F1 | Gap | Generalization |
|---------|--------|---------|-----|----------------|
| **Logistic** | 0.8933 | 0.9045 | +1.12% | Excellent |
| **SVM** | 0.8912 | 0.8922 | +0.10% | Excellent |
| **BiLSTM** | 0.8977 | 0.8833 | -1.44% | Very Good |
| **DistilBERT** | 0.9346 | 0.9167 | -1.79% | Very Good |

### 5.2. Hiệu năng theo số tham số
```
Performance (Test F1)
  1.00 │
       │                            ● DistilBERT (0.9167)
  0.95 │
       │
  0.90 │                   ● Logistic (0.9045)
       │                ● SVM (0.8922)
  0.85 │             ● BiLSTM (0.8833)
       │
  0.80 │
       └─────────────────────────────────────────────────
        0         20M        40M        60M      Parameters
```

### 5.3. Hiệu năng theo thời gian huấn luyện
```
Performance (Test F1)
  1.00 │
       │                                                ● DistilBERT (0.9167)
  0.95 │
       │
  0.90 │  ● Logistic (0.9045)
       │  ● SVM (0.8922)
  0.85 │                          ● BiLSTM (0.8833)
       │
  0.80 │
       └──────────────────────────────────────────────────────
        0min       20min          40min         60min    Training Time
```

---

## 6. SO SÁNH THEO METRIC CHÍNH

### 6.1. Precision (Test)

| Mô Hình | Negative Precision | Positive Precision | Macro Avg |
|---------|--------------------|--------------------|-----------|
| **DistilBERT** | 0.92 | 0.91 | **0.915** |
| **Logistic** | 0.90 | 0.91 | **0.905** |
| **SVM** | 0.89 | 0.89 | **0.890** |
| **BiLSTM** | 0.89 | 0.88 | **0.885** |

### 6.2. Recall (Test)

| Mô Hình | Negative Recall | Positive Recall | Macro Avg |
|---------|-----------------|-----------------|-----------|
| **DistilBERT** | 0.91 | 0.92 | **0.915** |
| **Logistic** | 0.91 | 0.90 | **0.905** |
| **SVM** | 0.89 | 0.89 | **0.890** |
| **BiLSTM** | 0.88 | 0.89 | **0.885** |

### 6.3. Ma trận nhầm lẫn (tóm tắt)

Logistic Regression
```
                 Predicted
              Negative  Positive
Actual
Negative       4550      450      (True Neg Rate: 91.0%)
Positive       502      4498     (True Pos Rate: 90.0%)
```

LinearSVC
```
                 Predicted
              Negative  Positive
Actual
Negative       4460      540      (True Neg Rate: 89.2%)
Positive       538      4462     (True Pos Rate: 89.2%)
```

BiLSTM
```
                 Predicted
              Negative  Positive
Actual
Negative       4385      615      (True Neg Rate: 87.7%)
Positive       552      4448     (True Pos Rate: 89.0%)
```

DistilBERT
```
                 Predicted
              Negative  Positive
Actual
Negative       4550      450      (True Neg Rate: 91.0%)
Positive       383      4617     (True Pos Rate: 92.3%)
```

---

## 7. HƯỚNG CẢI THIỆN THEO NHÓM MÔ HÌNH

- Logistic/SVM: mở rộng n‑grams (3–4), bổ sung đặc trưng ký tự, thử các chuẩn hoá TF‑IDF khác nhau; xem xét ensemble nhẹ.
- BiLSTM: dùng embedding tiền huấn luyện (GloVe/FastText), thêm attention hoặc tăng đa dạng pooling; điều chỉnh kích thước embedding/hidden và số tầng; áp dụng regularization mạnh hơn.
- DistilBERT: chiến lược unfreezing theo lớp, LR phân biệt; kéo dài thời gian huấn luyện; thử multi‑sample dropout; kết hợp ensemble BERT‑based.

---

## 8. KẾT LUẬN VÀ KHUYẾN NGHỊ

- Nếu ưu tiên tốc độ và đơn giản, Logistic Regression/SVM là lựa chọn hợp lý, đạt hiệu năng quanh 90% F1 trên tập test.
- Khi có GPU và yêu cầu độ chính xác cao, DistilBERT cho kết quả tốt nhất nhưng đổi lại là thời gian và tài nguyên.
- Với BiLSTM, hiệu năng có thể nâng lên nếu bổ sung embedding tiền huấn luyện và cơ chế chú ý; ở bài toán nhị phân đơn giản, baseline TF‑IDF nhiều khi đã đủ mạnh.

---

## 9. PHỤ LỤC

### 9.1. Cấu trúc thư mục
```
project/
├── data/
│   └── dataset.csv
├── split_augmented_data/
│   ├── train_augmented.csv
│   └── test_original.csv
├── encoded_split_data/
│   ├── train_encoded_texts.npy
│   ├── train_encoded_labels.npy
│   ├── test_encoded_texts.npy
│   ├── test_encoded_labels.npy
│   └── metadata.json
├── outputs_logistic/
│   └── meta.json
├── outputs_svm/
│   └── meta.json
├── out_put_bilstm/
│   ├── best_model.pt
│   └── meta.json
├── outputs_bert/
│   ├── best_model/
│   └── meta.json
└── notebooks/
    ├── 1_split_and_augment.ipynb
    ├── 2_encode_split_data.ipynb
    ├── 3-train-with-split.ipynb
    ├── 3b_train_with_svm.ipynb
    ├── 3c_train_with_logistic.ipynb
    └── 3d_train_with_bert.ipynb
```

---

