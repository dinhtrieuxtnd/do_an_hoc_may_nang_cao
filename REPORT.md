# Báo cáo chi tiết — Sentiment Analysis

Tài liệu này chứa phân tích chi tiết về pipeline dữ liệu, cấu hình mô hình, kết quả thực nghiệm và ghi chú lý thuyết. Để xem hướng dẫn cài đặt và chạy, xem [README.md](README.md).

---

## Pipeline dữ liệu

```
Dataset gốc (50,000 mẫu)
         │
   Stratified Split 80/20
         │
    ┌─────┴──────┐
    ▼            ▼
 Train          Test
 40,000         10,000
    │              │
 Back Translation  (giữ nguyên)
 +50% mỗi class    │
    │              │
 Train augmented   Test original
 59,995            10,000
    │              │
 Train/Val 85/15   │
    │     │        │
 50,995  9,000    10,000
 train   val      test
```

### Nguyên tắc chống data leakage

- Test set chia **trước** khi augment.
- Vocabulary (cho BiLSTM) chỉ xây dựng từ train.
- TF-IDF vectorizer chỉ `fit` trên train.
- Test set không bao giờ tham gia vào quá trình huấn luyện.

---

## Chi tiết các mô hình

### Logistic Regression

- Input: TF-IDF vectors (max 10,000 features, unigram + bigram).
- Solver: `lbfgs`, C=1.0, `class_weight='balanced'`.

### LinearSVC

- Input: TF-IDF vectors (max 10,000 features, unigram + bigram).
- C=1.0, `class_weight='balanced'`, `dual=False`.

### BiLSTM

- Embedding 256-dim (học từ scratch), BiLSTM 256 hidden (→ 512 bidirectional).
- Multi-pooling: max + mean + last hidden → 1,536-dim.
- Dropout 0.35, label smoothing 0.1, gradient clipping 0.5.
- Optimizer: Adam (lr=5e-4, weight_decay=1e-4).
- Early stopping patience=8, tổng 34 epochs.
- Tổng tham số: **13,856,258**.

### DistilBERT

- Pre-trained `distilbert-base-uncased`, fine-tune toàn bộ.
- Max length 256, batch 32, lr=1e-5.
- Warmup 10%, linear decay, label smoothing 0.1.
- Early stopping patience=3.
- Tổng tham số: **66,955,010**.

---

## Kết quả thực nghiệm

### Bảng so sánh (trên test set — 10,000 mẫu, không augment)

| Mô hình | Val F1 | Test F1 | Test Acc | Gap | Params |
|---------|--------|---------|----------|-----|--------|
| **DistilBERT** | 0.9346 | **0.9167** | 91.67% | −1.79% | ~67M |
| **Logistic Regression** | 0.8933 | **0.9045** | 90.45% | +1.12% | ~10K |
| **LinearSVC** | 0.8912 | **0.8922** | 89.22% | +0.10% | ~10K |
| **BiLSTM** | 0.8977 | **0.8833** | 88.33% | −1.44% | ~13.9M |

> **Nhận xét**: DistilBERT đạt kết quả tốt nhất. Logistic Regression đạt hiệu quả rất cao so với chi phí. BiLSTM cho kết quả thấp hơn baseline — xem phần [Ghi chú lý thuyết](#ghi-chú-lý-thuyết) để hiểu nguyên nhân.