# 🎬 Sentiment Analysis — ML truyền thống vs Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.x-FFD21E)](https://huggingface.co/transformers)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Đồ án môn **Học Máy Nâng Cao** — Phân loại cảm xúc nhị phân (positive / negative) trên tập **IMDB 50K Movie Reviews**, so sánh 4 mô hình từ baseline truyền thống đến transformer hiện đại.

> 📄 Phân tích chi tiết về pipeline, lý thuyết và nhận xét kết quả xem tại [REPORT.md](REPORT.md).

---

## ✨ Tính năng nổi bật

- **So sánh 4 mô hình** trên cùng bộ dữ liệu: Logistic Regression, LinearSVC, BiLSTM, DistilBERT.
- **Data Augmentation** bằng Back Translation (EN → DE/FR → EN) tăng ~50% dữ liệu train.
- **Chống data leakage** nghiêm ngặt: test set chia trước khi augment, vocabulary/TF-IDF chỉ fit trên train.
- **Pipeline tái tạo hoàn toàn** (reproducible): seed cố định, output lưu sẵn.
- **Hỗ trợ GPU & CPU**: mô hình nhẹ chạy local, mô hình nặng chạy trên Kaggle.

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python 3.8+ |
| ML truyền thống | scikit-learn (TF-IDF + Logistic Regression, LinearSVC) |
| Deep Learning | PyTorch (BiLSTM) |
| Transformer | HuggingFace Transformers (DistilBERT) |
| Data Augmentation | googletrans (Back Translation) |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter / Kaggle |

---

## 🚀 Hướng dẫn bắt đầu

### Yêu cầu hệ thống

- **Python** ≥ 3.8
- **GPU** (khuyến nghị): NVIDIA GPU với ≥ 16GB VRAM cho BiLSTM và DistilBERT.  
  Hoặc chạy trực tiếp trên **Kaggle** (Tesla T4 miễn phí).
- **CPU**: đủ cho Logistic Regression và LinearSVC.

### Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/<username>/do_an_hoc_may_nang_cao.git
cd do_an_hoc_may_nang_cao

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Cài đặt thư viện
pip install pandas numpy scikit-learn torch transformers googletrans==4.0.0rc1 tqdm matplotlib seaborn joblib
```

### Cấu trúc thư mục

```
.
├── data/
│   └── dataset.csv                     # Dataset IMDB gốc (50,000 mẫu)
├── split_augmented_data/               # Dữ liệu đã chia & augment
├── encoded_split_data/                 # Dữ liệu đã mã hóa (cho BiLSTM)
├── outputs_logistic/                   # Kết quả Logistic Regression
├── outputs_svm/                        # Kết quả LinearSVC
├── outputs_bilstm/                     # Kết quả BiLSTM
├── outputs_bert/                       # Kết quả DistilBERT
├── 1_split_and_augment.ipynb           # Bước 1: Chia dữ liệu + augment
├── 2_encode_split_data.ipynb           # Bước 2: Mã hóa (cho BiLSTM)
├── 3a_train_with_bilstm.ipynb          # Bước 3: Train BiLSTM
├── 3b_train_with_svm.ipynb             # Bước 3: Train LinearSVC
├── 3c_train_with_logistic.ipynb        # Bước 3: Train Logistic Regression
├── 3d-train-with-bert.ipynb            # Bước 3: Fine-tune DistilBERT
├── REPORT.md                           # Báo cáo chi tiết
└── README.md                           # File này
```

---

## 📖 Cách sử dụng

Chạy các notebook **theo thứ tự**. Mỗi bước phụ thuộc output của bước trước.

### Bước 1 — Chia dữ liệu & Augment

```bash
jupyter notebook 1_split_and_augment.ipynb
```
- **Input**: `data/dataset.csv`
- **Output**: `split_augmented_data/train_augmented.csv`, `test_original.csv`
- ⚠️ Thời gian chạy: ~19 tiếng (do Google Translate API). File output **đã có sẵn** trong repo.

### Bước 2 — Mã hóa dữ liệu (chỉ cần cho BiLSTM)

```bash
jupyter notebook 2_encode_split_data.ipynb
```
- **Input**: `split_augmented_data/*.csv`
- **Output**: `encoded_split_data/*.npy`, `*.json`

### Bước 3 — Train mô hình

Chọn 1 hoặc nhiều notebook để chạy:

| Notebook | Mô hình | Chạy trên |
|----------|---------|-----------|
| `3a_train_with_bilstm.ipynb` | BiLSTM | Kaggle GPU |
| `3b_train_with_svm.ipynb` | LinearSVC | Local CPU |
| `3c_train_with_logistic.ipynb` | Logistic Regression | Local CPU |
| `3d-train-with-bert.ipynb` | DistilBERT | Kaggle GPU |

```bash
# Ví dụ: chạy Logistic Regression trên máy local
jupyter notebook 3c_train_with_logistic.ipynb
```

Kết quả (model, metrics, plots) tự động lưu vào thư mục `outputs_*/` tương ứng.

---

## 📊 Kết quả

| Mô hình | Test F1 | Test Accuracy | Tham số |
|---------|---------|---------------|---------|
| **DistilBERT** | **0.9167** | 91.67% | ~67M |
| Logistic Regression | 0.9045 | 90.45% | ~10K |
| LinearSVC | 0.8922 | 89.22% | ~10K |
| BiLSTM | 0.8833 | 88.33% | ~13.9M |

> Chi tiết phân tích, nhận xét và giải thích kết quả xem tại [REPORT.md](REPORT.md).

---

## 👥 Tác giả

| Thành viên | Vai trò |
|------------|---------|
| *Cập nhật tên thành viên* | *Cập nhật vai trò* |

Đồ án môn **Học Máy Nâng Cao** — Học kỳ 1, năm học 2025–2026.

---

## 📄 License

Dự án được phân phối dưới giấy phép [MIT License](LICENSE). Xem file `LICENSE` để biết chi tiết.

---

## 🔗 Tham khảo

- [IMDB Dataset — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT*
- Edunov et al. (2018). *Understanding Back-Translation at Scale*
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization*
