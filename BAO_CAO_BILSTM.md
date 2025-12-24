# BÁO CÁO BTL
# PHÂN LOẠI CẢM XÚC QUA VĂN BẢN MÃ HÓA SỬ DỤNG MẠNG BILSTM

## MỤC LỤC

1. [GIỚI THIỆU](#chương-1-giới-thiệu)
2. [CƠ SỞ LÝ THUYẾT](#chương-2-cơ-sở-lý-thuyết)
3. [DỮ LIỆU VÀ TIỀN XỬ LÝ](#chương-3-dữ-liệu-và-tiền-xử-lý)
4. [PHƯƠNG PHÁP ĐỀ XUẤT](#chương-4-phương-pháp-đề-xuất)
5. [THỰC NGHIỆM VÀ KẾT QUẢ](#chương-5-thực-nghiệm-và-kết-quả)
6. [KẾT LUẬN](#chương-6-kết-luận)
7. [TÀI LIỆU THAM KHẢO](#tài-liệu-tham-khảo)

---

## TÓM TẮT

Phân loại cảm xúc (sentiment analysis) là một trong những bài toán quan trọng trong xử lý ngôn ngữ tự nhiên với nhiều ứng dụng thực tế như phân tích đánh giá sản phẩm, giám sát dư luận mạng xã hội, và hệ thống khuyến nghị. Báo cáo này trình bày việc xây dựng và đánh giá mô hình Bidirectional Long Short-Term Memory (BiLSTM) kết hợp với chiến lược multi-pooling để phân loại cảm xúc văn bản thành hai lớp: positive và negative.

Nghiên cứu sử dụng dataset gồm 69,995 đánh giá, được chia thành train set (80%) và test set (20%). Để tăng cường dữ liệu huấn luyện, chúng tôi áp dụng kỹ thuật back translation [Edunov et al., 2018] với hai ngôn ngữ trung gian (German và French), tăng train set lên 59,995 mẫu. Mô hình BiLSTM được thiết kế với embedding layer 256 chiều, LSTM layer hai chiều với hidden dimension 512, và multi-pooling strategy kết hợp max pooling, mean pooling, và last hidden states [Wang et al., 2016].

Kết quả thực nghiệm cho thấy mô hình đạt **Test Macro-F1 Score 88.33%** và **Test Accuracy 88.33%** trên tập test, với sự cân bằng tốt giữa precision và recall cho cả hai lớp. Khoảng cách giữa validation F1 (89.77%) và test F1 (88.33%) chỉ 1.44%, chứng tỏ khả năng tổng quát hóa tốt của mô hình. Các kỹ thuật regularization như dropout (0.35), weight decay (1e-4), gradient clipping (0.5), và label smoothing (0.1) đã hiệu quả trong việc giảm overfitting.

**Từ khóa:** Sentiment Analysis, BiLSTM, Multi-Pooling, Back Translation, Deep Learning, Natural Language Processing

---

# CHƯƠNG 1: GIỚI THIỆU

## 1.1. Đặt Vấn Đề

Trong kỷ nguyên số hóa, lượng dữ liệu văn bản trên internet tăng trưởng với tốc độ chóng mặt, đặc biệt là các đánh giá, bình luận và ý kiến của người dùng trên các nền tảng thương mại điện tử và mạng xã hội. Phân loại cảm xúc tự động từ các văn bản này trở thành một nhu cầu thiết yếu cho các doanh nghiệp và tổ chức trong việc hiểu khách hàng, cải thiện sản phẩm, và đưa ra quyết định kinh doanh [Liu, 2012].

Tuy nhiên, phân loại cảm xúc là một bài toán phức tạp do những thách thức sau:

1. **Ngữ cảnh và ngữ nghĩa:** Cảm xúc phụ thuộc vào ngữ cảnh và có thể thay đổi theo cách diễn đạt
2. **Phủ định và ngôn ngữ mỉa mai:** Các cấu trúc như "not good" hoặc "not bad" đảo ngược cảm xúc
3. **Độ dài văn bản biến đổi:** Reviews có thể ngắn (vài từ) hoặc dài (hàng trăm từ)
4. **Dữ liệu không cân bằng:** Phân phối giữa các lớp có thể lệch
5. **Long-term dependencies:** Cần hiểu mối quan hệ giữa các từ cách xa nhau

Các phương pháp truyền thống như Naive Bayes, SVM với bag-of-words thường gặp khó khăn trong việc capture semantic relationships và long-term dependencies [Zhang et al., 2018]. Deep learning, đặc biệt là Recurrent Neural Networks (RNNs), đã cho thấy khả năng vượt trội trong xử lý sequential data [Goldberg, 2017].

## 1.2. Mục Tiêu Nghiên Cứu

Mục tiêu chính của đồ án này là:

1. **Xây dựng mô hình BiLSTM hiệu quả** cho bài toán phân loại cảm xúc nhị phân (positive/negative)
2. **Áp dụng data augmentation** thông qua back translation để tăng cường dữ liệu huấn luyện
3. **Thiết kế multi-pooling strategy** để capture thông tin toàn diện từ sequence representations
4. **Đánh giá hiệu quả** của các kỹ thuật regularization trong việc cải thiện generalization
5. **Phân tích kết quả** và so sánh với các phương pháp baseline

## 1.3. Đóng Góp Chính

Các đóng góp chính của đồ án bao gồm:

1. **Pipeline hoàn chỉnh:** Quy trình từ xử lý dữ liệu, augmentation, encoding đến training và evaluation
2. **Multi-pooling strategy:** Kết hợp ba chiến lược pooling (max, mean, last hidden) để tận dụng thông tin đa dạng
3. **Data augmentation hiệu quả:** Back translation với hai ngôn ngữ trung gian tăng 7.14% dữ liệu train
4. **Comprehensive regularization:** Kết hợp nhiều techniques (dropout, weight decay, gradient clipping, label smoothing)
5. **Kết quả ấn tượng:** Đạt 88.33% F1-score trên test set với generalization tốt

## 1.4. Cấu Trúc Báo Cáo

Báo cáo được tổ chức như sau:

- **Chương 2** trình bày cơ sở lý thuyết về LSTM, BiLSTM, word embeddings, và các kỹ thuật liên quan
- **Chương 3** mô tả dataset, các bước tiền xử lý, data augmentation, và encoding
- **Chương 4** chi tiết kiến trúc mô hình BiLSTM đề xuất và các thành phần
- **Chương 5** trình bày quá trình huấn luyện, kết quả thực nghiệm và phân tích
- **Chương 6** tóm tắt kết quả, thảo luận hạn chế và hướng phát triển

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1. Recurrent Neural Networks và Vấn Đề Vanishing Gradient

### 2.1.1. Recurrent Neural Networks

Recurrent Neural Networks (RNNs) là một lớp neural networks được thiết kế để xử lý sequential data bằng cách duy trì hidden state qua các timesteps [Elman, 1990]. Tại mỗi timestep $t$, RNN nhận input $x_t$ và hidden state trước đó $h_{t-1}$, sau đó tính toán hidden state mới:

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

Trong đó $W_{hh}$, $W_{xh}$ là weight matrices và $b_h$ là bias vector.

### 2.1.2. Vanishing Gradient Problem

Bengio et al. [1994] chỉ ra rằng RNNs gặp phải vấn đề **vanishing gradient** khi training với sequences dài. Khi backpropagation through time (BPTT), gradient được tính theo chain rule:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

Nếu $||\frac{\partial h_t}{\partial h_{t-1}}|| < 1$, gradient sẽ giảm exponentially theo độ dài sequence, dẫn đến network không học được long-term dependencies. Ngược lại, nếu $||\frac{\partial h_t}{\partial h_{t-1}}|| > 1$, xảy ra **exploding gradient** [Pascanu et al., 2013].

**Hệ quả:** RNNs truyền thống chỉ có thể nhớ thông tin trong khoảng 5-10 timesteps [Hochreiter, 1991], không phù hợp cho các tasks cần long-term memory như sentiment analysis.

## 2.2. Long Short-Term Memory (LSTM)

### 2.2.1. Kiến Trúc LSTM

Hochreiter & Schmidhuber [1997] đề xuất LSTM để giải quyết vanishing gradient problem thông qua **cell state** và **gate mechanisms**. LSTM cell bao gồm:

#### a) Forget Gate
Quyết định thông tin nào từ cell state cũ cần loại bỏ:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Trong đó $\sigma$ là sigmoid function ($\sigma(x) = \frac{1}{1+e^{-x}}$), output trong khoảng [0,1].

#### b) Input Gate và Candidate Cell State
Quyết định thông tin mới nào cần thêm vào cell state:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

#### c) Cell State Update
Cập nhật cell state bằng cách kết hợp thông tin cũ và mới:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Trong đó $\odot$ là element-wise multiplication (Hadamard product).

#### d) Output Gate và Hidden State
Quyết định output dựa trên cell state:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### 2.2.2. Tại Sao LSTM Giải Quyết Vanishing Gradient?

Gers et al. [2000] giải thích rằng cell state $C_t$ cho phép gradient flow trực tiếp qua time:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Vì forget gate $f_t$ được học, network có thể maintain gradient flow bằng cách setting $f_t \approx 1$ khi cần nhớ long-term information. Điều này cho phép LSTM học dependencies hàng trăm timesteps [Hochreiter & Schmidhuber, 1997].

**Ưu điểm của LSTM:**
- Giải quyết vanishing gradient problem
- Học được long-term dependencies (100+ timesteps)
- Selective memory thông qua learnable gates
- Robust với nhiều loại sequential tasks

## 2.3. Bidirectional LSTM (BiLSTM)

### 2.3.1. Motivation

Schuster & Paliwal [1997] chỉ ra rằng LSTM unidirectional chỉ xử lý sequence theo một hướng (left-to-right), dẫn đến việc mất mát context information từ phía sau. Trong nhiều NLP tasks, context từ cả hai phía đều quan trọng [Graves & Schmidhuber, 2005].

**Ví dụ:** Trong câu "The movie was not ___ at all", để dự đoán từ còn thiếu cần context từ cả "not" (trước) và "at all" (sau).

### 2.3.2. Kiến Trúc BiLSTM

BiLSTM xử lý sequence theo **hai hướng song song** [Graves et al., 2013]:

**Forward LSTM:** Xử lý từ trái sang phải
$$\overrightarrow{h}_t = \text{LSTM}_{forward}(x_t, \overrightarrow{h}_{t-1}, \overrightarrow{C}_{t-1})$$

**Backward LSTM:** Xử lý từ phải sang trái
$$\overleftarrow{h}_t = \text{LSTM}_{backward}(x_t, \overleftarrow{h}_{t+1}, \overleftarrow{C}_{t+1})$$

**Output Concatenation:** Kết hợp outputs từ cả hai hướng
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

Trong đó $[\cdot; \cdot]$ là concatenation operation.

### 2.3.3. Ứng Dụng trong Sentiment Analysis

BiLSTM đặc biệt hiệu quả cho sentiment analysis vì [Tang et al., 2015]:

1. **Negation handling:** Hiểu được cấu trúc phủ định như "not good", "not bad"
2. **Context dependency:** Words như "good" có cảm xúc khác nhau tùy context
3. **Full sentence understanding:** Capture meaning từ toàn bộ câu, không chỉ phần đầu

**Ví dụ:** "The first half was boring but the ending was amazing"
- Forward LSTM bias về "boring" (negative)
- Backward LSTM bias về "amazing" (positive)
- BiLSTM capture được contrast và overall positive sentiment

## 2.4. Word Embeddings

### 2.4.1. Distributed Word Representations

Mikolov et al. [2013] đề xuất **distributed representations** thay thế one-hot encoding. Thay vì represent mỗi word bằng sparse vector dimension $|V|$, word embeddings map words sang dense vectors dimension $d$ ($d << |V|$):

$$w_i \rightarrow \mathbf{e}_i \in \mathbb{R}^d$$

**Distributional Hypothesis** [Harris, 1954]: Words xuất hiện trong contexts giống nhau có meanings tương tự.

### 2.4.2. Embedding Layer

Embedding layer là lookup table $E \in \mathbb{R}^{|V| \times d}$ [Collobert et al., 2011]:

$$\mathbf{e}_i = E[w_i]$$

Trong đó $w_i$ là word index và $\mathbf{e}_i$ là embedding vector tương ứng.

**Ưu điểm:**
- **Semantic similarity:** Words tương tự có embeddings gần nhau (cosine similarity cao)
- **Dimensionality reduction:** $d$ (256-512) << $|V|$ (50,000)
- **Learnable:** Embeddings được học jointly với task-specific objective
- **Transfer learning:** Có thể khởi tạo với pre-trained embeddings (Word2Vec, GloVe, FastText)

### 2.4.3. Pre-trained vs Task-specific Embeddings

**Pre-trained embeddings** [Pennington et al., 2014]:
- Trained trên large corpus (Wikipedia, Common Crawl)
- Capture general semantic relationships
- Có thể không optimal cho specific domain

**Task-specific embeddings:**
- Learned từ training data
- Optimized cho specific task
- Cần sufficient training data

Trong project này, chúng tôi sử dụng **task-specific embeddings** được học từ scratch.

## 2.5. Pooling Strategies

### 2.5.1. Max Pooling

Collobert et al. [2011] đề xuất **max pooling** cho NLP:

$$h_{max}^{(i)} = \max_{t=1}^{T} h_t^{(i)}$$

Trong đó $h_t^{(i)}$ là feature thứ $i$ tại timestep $t$.

**Ưu điểm:**
- Capture most salient features
- Invariant to sequence length
- Effective cho sentiment analysis (từ quan trọng nhất quyết định cảm xúc)

**Ví dụ:** "The movie was boring, tedious, and absolutely terrible"
- Max pooling capture "terrible" (strongest negative word)

### 2.5.2. Average/Mean Pooling

Average pooling tính trung bình features qua toàn bộ sequence [Kalchbrenner et al., 2014]:

$$h_{mean}^{(i)} = \frac{1}{T} \sum_{t=1}^{T} h_t^{(i)}$$

**Ưu điểm:**
- Capture overall sentiment distribution
- Less sensitive to outliers
- Smooth representation

### 2.5.3. Last Hidden State

Sử dụng hidden states cuối cùng từ BiLSTM [Sutskever et al., 2014]:

$$h_{last} = [\overrightarrow{h}_T; \overleftarrow{h}_1]$$

**Ưu điểm:**
- Capture final context sau khi xử lý toàn bộ sequence
- Natural choice cho sequence-to-sequence tasks

### 2.5.4. Multi-Pooling Strategy

Wang et al. [2016] đề xuất kết hợp nhiều pooling strategies:

$$h_{combined} = [h_{max}; h_{mean}; h_{last}]$$

**Ưu điểm:**
- Complementary information từ nhiều perspectives
- Richer representation
- Improved performance [Chen et al., 2017]

**Trade-off:** Tăng dimensionality (3× features) nhưng cải thiện accuracy đáng kể.

## 2.6. Regularization Techniques

### 2.6.1. Dropout

Srivastava et al. [2014] đề xuất **dropout** để prevent overfitting:

$$h_{drop} = h \odot m, \quad m_i \sim \text{Bernoulli}(1-p)$$

Trong đó $p$ là dropout probability.

**Cơ chế:**
- **Training:** Randomly "drop" neurons với probability $p$
- **Inference:** Scale outputs by $(1-p)$ hoặc sử dụng inverted dropout

**Hiệu ứng:** Dropout ngăn co-adaptation giữa neurons, forcing network học robust features [Hinton et al., 2012].

### 2.6.2. Weight Decay (L2 Regularization)

Krizhevsky et al. [2012] sử dụng **weight decay** để prevent large weights:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \sum_{w \in W} w^2$$

Trong đó $\lambda$ là regularization coefficient (typically $10^{-4}$ to $10^{-5}$).

**Hiệu ứng:** Encourage smaller weights, reducing model complexity và improving generalization.

### 2.6.3. Gradient Clipping

Pascanu et al. [2013] đề xuất **gradient clipping** để prevent exploding gradients trong RNNs:

$$\mathbf{g} \leftarrow \begin{cases}
\frac{\text{threshold}}{||\mathbf{g}||} \cdot \mathbf{g} & \text{if } ||\mathbf{g}|| > \text{threshold} \\
\mathbf{g} & \text{otherwise}
\end{cases}$$

**Variants:**
- **Gradient clipping by value:** Clip mỗi gradient element riêng lẻ
- **Gradient clipping by norm:** Clip toàn bộ gradient vector (được sử dụng trong project)

### 2.6.4. Label Smoothing

Szegedy et al. [2016] đề xuất **label smoothing** để prevent overconfident predictions:

$$y_{smooth}^{(i)} = \begin{cases}
1 - \epsilon & \text{if } i = y_{true} \\
\frac{\epsilon}{K-1} & \text{otherwise}
\end{cases}$$

Trong đó $K$ là number of classes và $\epsilon$ là smoothing factor (typically 0.1).

**Ưu điểm:**
- Prevent overconfidence [Müller et al., 2019]
- Improve calibration
- Better generalization

## 2.7. Data Augmentation: Back Translation

### 2.7.1. Text Data Augmentation Challenges

Khác với computer vision, data augmentation cho text khó khăn hơn vì [Wei & Zou, 2019]:
- Small changes có thể thay đổi hoàn toàn meaning
- Grammar và syntax constraints
- Need to preserve semantic meaning

### 2.7.2. Back Translation

Sennrich et al. [2016] giới thiệu **back translation** cho machine translation. Edunov et al. [2018] chứng minh hiệu quả cho data augmentation:

```
Original Text (English) 
    ↓ Translation Model
Intermediate Language (German/French)
    ↓ Back-Translation Model
Augmented Text (English)
```

**Key properties:**
- Preserve semantic meaning
- Generate paraphrases
- Maintain label consistency (sentiment polarity không đổi)

**Ví dụ:**
- Original: "This movie is absolutely amazing!"
- German: "Dieser Film ist absolut erstaunlich!"
- Back: "This film is absolutely astonishing!"

### 2.7.3. Ưu Điểm của Back Translation

Xia et al. [2019] phân tích advantages của back translation:

1. **Label preservation:** Sentiment polarity được giữ nguyên
2. **Diversity:** Tạo ra diverse paraphrases
3. **Naturalness:** Generated text natural hơn random operations
4. **Scalability:** Có thể scale với nhiều intermediate languages

## 2.8. Optimization và Training

### 2.8.1. Adam Optimizer

Kingma & Ba [2014] đề xuất **Adam** (Adaptive Moment Estimation), kết hợp momentum và RMSProp:

**First moment estimate (mean):**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment estimate (variance):**
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Bias correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update:**
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Default hyperparameters [Kingma & Ba, 2014]:**
- $\alpha = 10^{-3}$ (learning rate)
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

**Ưu điểm:** Adaptive learning rates cho mỗi parameter, robust với hyperparameter choice.

### 2.8.2. Learning Rate Scheduling

**ReduceLROnPlateau** [PyTorch Documentation] giảm learning rate khi validation metric plateau:

$$\alpha_{new} = \alpha_{old} \times \text{factor}$$

khi metric không improve sau `patience` epochs.

**Ưu điểm:** Automatic adaptation, không cần schedule phức tạp.

### 2.8.3. Early Stopping

Prechelt [1998] chứng minh **early stopping** là effective regularization:

- Monitor validation metric
- Stop training khi không improve sau `patience` epochs
- Restore best checkpoint

**Effect:** Prevent overfitting bằng cách dừng training trước khi model overfit.

---

# CHƯƠNG 3: DỮ LIỆU VÀ TIỀN XỬ LÝ

## 3.1. Dataset

### 3.1.1. Mô Tả Dataset
- **Tổng số mẫu:** 50,000 samples
- **Phân phối nhãn:**
  - Negative: 25,000 samples
  - Positive: 25,000 samples
- **Nguồn dữ liệu:** `dataset.csv` từ IMDB dataset trên kaggle

### 2.2. Chia Train/Test Set
Sử dụng **stratified train-test split** [Kohavi, 1995] để đảm bảo phân phối nhãn trong train và test sets giống nhau:

**Configuration:**
- **Train Set:** 80% (40000 samples)
  - Negative: 25,000 samples
  - Positive: 25,000 samples
- **Test Set:** 20% (10,000 samples)
  - Negative: 5,000 samples  
  - Positive: 5,000 samples
- **Method:** `train_test_split` với `stratify=y`
- **Random seed:** 42 (reproducibility)

### 3.2.2. Nguyên Tắc Quan Trọng

**Data leakage prevention [Kaufman et al., 2012]:**
1. Test set được chia **TRƯỚC KHI** augmentation
2. Test set **KHÔNG** được augment
3. Vocabulary chỉ build từ train set
4. Normalization statistics (mean, std) chỉ tính từ train
5. Không có overlap giữa train và test

**Lý do:** Đảm bảo test set reflect true unseen data distribution, tránh overestimate performance.

---

## 3. DATA AUGMENTATION

### 3.1. Phương Pháp: Back Translation
**Nguyên lý:** Dịch văn bản sang ngôn ngữ trung gian, sau đó dịch ngược lại tiếng Anh để tạo ra các phiên bản paraphrase giữ nguyên ý nghĩa.

**Cấu hình:**
- **Ngôn ngữ trung gian:** German (de), French (fr)
- **Tỷ lệ augmentation:** 50% (mỗi class tăng thêm 50% dữ liệu)
- **Độ dài tối đa cho back translation:** 500 ký tự
- **Delay giữa requests:** 0.5 giây (tránh rate limit)

### 3.2. Kết Quả Augmentation

**Train Set sau augmentation:**
- **Original train:** 40,000 samples
- **Augmented:** 19,995 samples (Back Translation)
- **Total train:** 59,995 samples

**Phân phối sau augmentation:**
- Negative: 30,000 samples
- Positive: 29,995 samples

**Test Set:**
- **Giữ nguyên:** 10,000 samples (không augment)

### 3.3. Lợi Ích của Back Translation
- Tăng cường đa dạng dữ liệu train
- Giữ nguyên ý nghĩa ngữ nghĩa
- Cải thiện khả năng generalization
- Giảm overfitting
- Không làm thay đổi nhãn cảm xúc

---

## 4. DATA ENCODING

### 4.1. Tiền Xử Lý Text
**Các bước:**
1. Chuyển về chữ thường (lowercase)
2. Loại bỏ HTML tags
3. Loại bỏ URLs và emails
4. Loại bỏ ký tự đặc biệt (chỉ giữ a-z, 0-9)
5. Loại bỏ số đứng riêng
6. Loại bỏ khoảng trắng thừa
7. Tokenization (tách từ)

### 4.2. Xây Dựng Vocabulary

**Cấu hình:**
- **Nguồn:** Chỉ xây dựng từ **train set** (không bao gồm test)
- **Tần suất tối thiểu:** 2 (từ phải xuất hiện ít nhất 2 lần)
- **Kích thước tối đa:** 50,000 từ
- **Special tokens:**
  - `<PAD>` (index 0): Padding
  - `<UNK>` (index 1): Unknown words

**Thống kê Vocabulary:**
- **Vocab size cuối cùng:** 50,002 từ (bao gồm special tokens)
- **Tổng số từ unique trong train:** >50,000 từ
- **Sau khi lọc (freq ≥ 2):** 50,000 từ

### 4.3. Thống Kê Độ Dài Sequences

**Train Set:**
- Min: 1 token
- Max: 2,485 tokens
- Mean: 183.9 tokens
- Median: 128 tokens
- P95: 516 tokens

**Test Set:**
- Min: 6 tokens
- Max: 2,151 tokens
- Mean: 233.2 tokens
- Median: 175 tokens
- P95: 597 tokens

**Độ dài tối đa được chọn:** 256 tokens (để cân bằng giữa coverage và hiệu quả)

### 4.4. Tỷ Lệ Unknown Words trong Test
- **UNK rate:** 0.86%
- Tỷ lệ rất thấp → Vocabulary từ train tốt và đại diện
- Model có thể xử lý tốt các từ mới trong test

---

## 5. KIẾN TRÚC MÔ HÌNH BiLSTM

### 5.1. Kiến Trúc Tổng Thể

```
Input Text
    ↓
Embedding Layer (256-dim)
    ↓
Bidirectional LSTM (256-dim)
    ↓
Multi-Pooling (Max + Mean + Last Hidden States)
    ↓
Dropout (0.35)
    ↓
Fully Connected Layer
    ↓
Output (2 classes: negative/positive)
```

### 5.2. Chi Tiết Layers

#### a) Embedding Layer
- **Input:** Vocabulary size = 50,002
- **Output:** Embedding dimension = 256
- **Padding index:** 0 (không học gradient cho padding tokens)
- **Tổng số parameters:** 50,002 × 256 = 12,800,512 params

#### b) Bidirectional LSTM
- **Hidden dimension:** 256
- **Direction:** Bidirectional (forward + backward)
- **Output dimension:** 256 × 2 = 512 (do bidirectional)
- **Batch first:** True
- **Tổng số parameters:** ~1,050,000 params

#### c) Multi-Pooling Strategy
Kết hợp 3 chiến lược pooling để capture thông tin toàn diện:

1. **Max Pooling:** 
   - Lấy giá trị max của mỗi feature qua toàn bộ sequence
   - Capture các đặc trưng quan trọng nhất
   - Output: 512-dim

2. **Mean Pooling:**
   - Lấy trung bình của mỗi feature qua toàn bộ sequence
   - Capture thông tin tổng quát
   - Output: 512-dim

3. **Last Hidden States:**
   - Kết hợp hidden states cuối cùng của forward và backward LSTM
   - Capture context cuối cùng
   - Output: 512-dim

**Concatenation:** 512 + 512 + 512 = 1,536-dim

#### d) Dropout Layer
- **Dropout rate:** 0.35
- **Mục đích:** Regularization, giảm overfitting

#### e) Fully Connected Layer
- **Input:** 1,536-dim
- **Output:** 2 classes (negative/positive)
- **Tổng số parameters:** 1,536 × 2 + 2 = 3,074 params

### 5.3. Tổng Số Parameters
**Total trainable parameters:** ~13,853,586 parameters (~13.9M params)

---

## 6. HUẤN LUYỆN MÔ HÌNH

### 6.1. Cấu Hình Huấn Luyện

**Hyperparameters:**
- **Batch size:** 128
- **Epochs:** 40 (với early stopping)
- **Learning rate:** 5e-4 (0.0005)
- **Optimizer:** Adam
- **Weight decay:** 1e-4 (L2 regularization)
- **Gradient clipping:** 0.5 (tránh gradient explosion)
- **Label smoothing:** 0.1
- **Dropout:** 0.35
- **Validation split:** 15% từ train set
- **Early stopping patience:** 8 epochs
- **Device:** GPU (CUDA)

### 6.2. Chia Train/Validation

Từ **train set đã augment** (59,995 samples), chia thành:
- **Train subset:** 50,995 samples (85%)
- **Validation subset:** 9,000 samples (15%)

**Phương pháp:** Stratified split để đảm bảo phân phối nhãn cân bằng

### 6.3. Loss Function và Optimizer

**Loss Function:** CrossEntropyLoss với các cải tiến:
- **Class weights:** Balanced (xử lý imbalanced data)
  - Negative: weight ≈ 1.0
  - Positive: weight ≈ 1.0
- **Label smoothing:** 0.1 (giảm overconfidence)

**Optimizer:** Adam
- Learning rate: 5e-4
- Weight decay: 1e-4 (L2 regularization)

**Learning Rate Scheduler:** ReduceLROnPlateau
- Mode: Maximize validation F1
- Factor: 0.5 (giảm LR xuống 50%)
- Patience: 3 epochs
- Min LR: 1e-6

### 6.4. Regularization Techniques

1. **Dropout:** 0.35 (giảm overfitting)
2. **Weight Decay:** 1e-4 (L2 regularization)
3. **Gradient Clipping:** 0.5 (stability)
4. **Label Smoothing:** 0.1 (giảm overconfidence)
5. **Early Stopping:** Patience = 8 epochs
6. **Learning Rate Scheduling:** Giảm LR khi val F1 không cải thiện

### 6.5. Quá Trình Huấn Luyện

**Epochs trained:** 34 epochs (stopped early)

**Training curve highlights:**
- Validation F1 tăng dần từ epoch 1
- Best validation F1 đạt được ở epoch 26-34
- Learning rate giảm dần khi validation F1 plateau
- Early stopping kích hoạt sau 8 epochs không cải thiện

**Learning rate schedule:**
- Initial: 5e-4
- Giảm dần theo ReduceLROnPlateau
- Final: ~1e-5 đến 5e-6

![Training History](outputs_bilstm/training_history.png)
*Hình 5.1: Đường cong training và validation loss/F1-score qua các epochs*

---

## 7. KẾT QUẢ ĐÁNH GIÁ

### 7.1. Performance trên Validation Set

**Best Validation Macro-F1:** 0.8977 (89.77%)

**Classification Report (Validation):**
```
              precision    recall  f1-score   support

    negative       0.90      0.89      0.90      4500
    positive       0.89      0.90      0.90      4500

    accuracy                           0.90      9000
   macro avg       0.90      0.90      0.90      9000
weighted avg       0.90      0.90      0.90      9000
```

### 7.2. Performance trên Test Set (QUAN TRỌNG)

**Test Macro-F1:** 0.8833 (88.33%)  
**Test Accuracy:** 88.33%  
**Test Loss:** 0.3855

**Classification Report (Test):**
```
              precision    recall  f1-score   support

    negative       0.89      0.88      0.88      5000
    positive       0.88      0.89      0.88      5000

    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000
```

**Confusion Matrix (Test Set):**
```
                 Predicted
              Negative  Positive
Actual
Negative       4385      615
Positive        552     4448
```

**Phân tích:**
- **True Negatives:** 4,385 (87.7%)
- **False Positives:** 615 (12.3%)
- **False Negatives:** 552 (11.0%)
- **True Positives:** 4,448 (89.0%)

![Confusion Matrix](outputs_bilstm/test_confusion_matrix.png)
*Hình 6.1: Ma trận nhầm lẫn (Confusion Matrix) trên test set*

### 7.3. Đánh Giá Tổng Thể

**Ưu điểm:**
1. **Balanced performance:** F1 score cân bằng giữa negative và positive (~0.88)
2. **Generalization tốt:** Val F1 (89.77%) và Test F1 (88.33%) chênh lệch nhỏ (~1.4%)
3. **Không overfitting:** Test performance tốt chứng tỏ model generalize tốt
4. **Precision và Recall cân bằng:** Cả hai đều ở mức ~0.88-0.89

**Nhược điểm:**
1. **False Positives:** 615 negative reviews bị phân loại nhầm thành positive (12.3%)
2. **False Negatives:** 552 positive reviews bị phân loại nhầm thành negative (11%)
3. **Có thể cải thiện thêm:** F1 ~88.33% còn có thể tăng lên

---

## 8. LƯU Ý QUAN TRỌNG

### 8.1. Về Dữ Liệu
- **Train set đã augment** (Back Translation) để tăng cường dữ liệu
- **Test set GIỮ NGUYÊN** (không augment) để đảm bảo đánh giá khách quan
- **Vocabulary chỉ xây dựng từ train set** (không leak thông tin từ test)
- **KHÔNG có data leakage** giữa train và test
- **Stratified split** đảm bảo phân phối nhãn cân bằng

### 8.2. Về Model
- **Multi-pooling strategy** giúp capture thông tin toàn diện
- **Bidirectional LSTM** xử lý context hai chiều
- **Regularization đầy đủ** (dropout, weight decay, gradient clipping, label smoothing)
- **Learning rate scheduling** giúp model hội tụ tốt hơn
- **Early stopping** tránh overfitting

### 8.3. Về Evaluation
- **Test F1 (88.33%)** phản ánh khả năng thực tế của model
- **Chênh lệch nhỏ giữa val và test** (~1.4%) chứng tỏ model ổn định
- **Balanced performance** trên cả negative và positive class  

---

## 9. KẾT LUẬN

### 9.1. Tóm Tắt
Mô hình **BiLSTM với multi-pooling** đã được xây dựng và huấn luyện thành công cho bài toán phân loại cảm xúc, đạt được:
- **Test Macro-F1:** 88.33%
- **Test Accuracy:** 88.33%
- **Generalization tốt** (val-test gap chỉ 1.4%)
- **Balanced performance** trên cả hai classes

### 9.2. Điểm Mạnh
1. **Quy trình chặt chẽ:** Từ data augmentation → encoding → training → evaluation
2. **Data augmentation hiệu quả:** Back Translation tăng 50% dữ liệu train
3. **Kiến trúc hợp lý:** BiLSTM + multi-pooling capture thông tin toàn diện
4. **Regularization đầy đủ:** Nhiều techniques để tránh overfitting
5. **Evaluation khách quan:** Test set không augment, không leak

### 9.3. Hướng Cải Thiện
1. **Tăng model capacity:** Thử tăng embed_dim, hidden_dim lên 512
2. **Attention mechanism:** Thêm attention layer để focus vào từ quan trọng
3. **Ensemble:** Kết hợp nhiều models (BiLSTM + CNN + Transformer)
4. **Hyperparameter tuning:** Grid search hoặc Bayesian optimization
5. **Pre-trained embeddings:** Sử dụng GloVe, Word2Vec, FastText
6. **Data augmentation thêm:** Synonym replacement, random insertion/deletion
7. **Class balancing:** Thử các techniques khác như SMOTE, focal loss

---

## 10. FILES VÀ OUTPUTS

### 10.1. Input Files
- `data/dataset.csv` - Dataset gốc
- `split_augmented_data/train_augmented.csv` - Train set đã augment
- `split_augmented_data/test_original.csv` - Test set nguyên bản
- `encoded_split_data/train_encoded_texts.npy` - Train texts đã encode
- `encoded_split_data/train_encoded_labels.npy` - Train labels đã encode
- `encoded_split_data/test_encoded_texts.npy` - Test texts đã encode
- `encoded_split_data/test_encoded_labels.npy` - Test labels đã encode
- `encoded_split_data/word2idx.json` - Word to index mapping
- `encoded_split_data/idx2word.json` - Index to word mapping
- `encoded_split_data/label2idx.json` - Label to index mapping
- `encoded_split_data/idx2label.json` - Index to label mapping
- `encoded_split_data/metadata.json` - Dataset metadata

### 10.2. Output Files
- `out_put_bilstm/best_model.pt` - Best model checkpoint
- `out_put_bilstm/meta.json` - Model metadata và performance metrics
- `out_put_bilstm/meta.npz` - Model metadata (binary format)

### 10.3. Notebooks
- `1_split_and_augment.ipynb` - Data splitting và augmentation
- `2_encode_split_data.ipynb` - Data encoding
- `3-train-with-split.ipynb` - Model training và evaluation

---

## 11. CƠ SỞ LÝ THUYẾT

### 11.1. Long Short-Term Memory (LSTM)

#### 11.1.1. Vấn Đề Vanishing Gradient
Recurrent Neural Networks (RNNs) truyền thống gặp vấn đề **vanishing gradient** khi xử lý sequences dài [Bengio et al., 1994]. Gradient giảm dần theo thời gian, khiến network không học được long-term dependencies.

#### 11.1.2. Kiến Trúc LSTM
LSTM được đề xuất bởi Hochreiter & Schmidhuber (1997) để giải quyết vấn đề này thông qua **cell state** và các **gates**:

**1. Forget Gate (ft):**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
Quyết định thông tin nào từ cell state cũ cần loại bỏ.

**2. Input Gate (it) và Candidate Cell State (C̃t):**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
Quyết định thông tin mới nào cần thêm vào cell state.

**3. Cell State Update:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
Cập nhật cell state bằng cách kết hợp thông tin cũ và mới.

**4. Output Gate (ot) và Hidden State:**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$
Quyết định output dựa trên cell state.

**Ưu điểm của LSTM:**
- Giải quyết vanishing gradient problem
- Học được long-term dependencies (hundreds of timesteps)
- Selective memory thông qua gate mechanism

### 11.2. Bidirectional LSTM (BiLSTM)

#### 11.2.1. Motivation
LSTM truyền thống chỉ xử lý sequence theo một hướng (left-to-right), dẫn đến việc mất mát context từ phía sau. Schuster & Paliwal (1997) đề xuất **Bidirectional RNN** để giải quyết vấn đề này.

#### 11.2.2. Cơ Chế Hoạt Động
BiLSTM xử lý sequence theo **hai hướng ngược nhau**:

**Forward LSTM:** 
$$\overrightarrow{h}_t = \text{LSTM}_{forward}(x_t, \overrightarrow{h}_{t-1})$$
Xử lý từ trái sang phải, capture context trước đó.

**Backward LSTM:**
$$\overleftarrow{h}_t = \text{LSTM}_{backward}(x_t, \overleftarrow{h}_{t+1})$$
Xử lý từ phải sang trái, capture context phía sau.

**Output Concatenation:**
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$
Kết hợp outputs từ cả hai hướng.

**Ưu điểm trong Sentiment Analysis:**
- Capture full context (past + future)
- Hiểu được negation patterns ("not good", "not bad")
- Tốt hơn cho tasks cần hiểu toàn bộ câu

### 11.3. Word Embeddings

#### 11.3.1. Distributed Representations
Mikolov et al. (2013) chỉ ra rằng **distributed word representations** học được semantic và syntactic patterns tốt hơn one-hot encoding.

#### 11.3.2. Embedding Layer
Embedding layer map discrete tokens sang continuous vector space:
$$e_i = E[w_i]$$
Trong đó:
- $w_i$: Word index
- $E \in \mathbb{R}^{V \times d}$: Embedding matrix
- $V$: Vocabulary size
- $d$: Embedding dimension
- $e_i \in \mathbb{R}^d$: Word embedding vector

**Lợi ích:**
- Capture semantic similarity (words với nghĩa giống nhau có embeddings gần nhau)
- Dimensionality reduction (từ $V$-dim sparse sang $d$-dim dense)
- Transfer learning potential (có thể dùng pre-trained embeddings)

### 11.4. Pooling Strategies

#### 11.4.1. Max Pooling
$$h_{max} = \max_{t=1}^{T} h_t$$
Capture most salient features, phù hợp với sentiment analysis vì từ quan trọng nhất thường quyết định cảm xúc [Collobert et al., 2011].

#### 11.4.2. Mean Pooling (Average Pooling)
$$h_{mean} = \frac{1}{T} \sum_{t=1}^{T} h_t$$
Capture overall sentiment distribution, giảm effect của outliers.

#### 11.4.3. Last Hidden State
$$h_{last} = [\overrightarrow{h}_T; \overleftarrow{h}_1]$$
BiLSTM's last states capture final context sau khi xử lý toàn bộ sequence.

#### 11.4.4. Multi-Pooling
Kết hợp nhiều pooling strategies [Wang et al., 2016]:
$$h_{combined} = [h_{max}; h_{mean}; h_{last}]$$
Provides richer representation bằng cách kết hợp complementary information.

### 11.5. Regularization Techniques

#### 11.5.1. Dropout
Srivastava et al. (2014) đề xuất **dropout** để prevent overfitting:
$$h_{drop} = h \odot m, \quad m \sim \text{Bernoulli}(1-p)$$
Trong đó $p$ là dropout rate (0.35 trong project này).

**Cơ chế:**
- Training: Randomly "drop" neurons với probability $p$
- Inference: Scale outputs by $(1-p)$ hoặc dùng inverted dropout

#### 11.5.2. Weight Decay (L2 Regularization)
Thêm penalty term vào loss function:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_{i} w_i^2$$
Trong đó $\lambda = 10^{-4}$ (weight decay coefficient).

**Mục đích:** Prevent large weights, improve generalization.

#### 11.5.3. Gradient Clipping
Pascanu et al. (2013) đề xuất gradient clipping để prevent **exploding gradients** trong RNNs:
$$g \leftarrow \begin{cases}
\frac{\text{threshold}}{||g||} \cdot g & \text{if } ||g|| > \text{threshold} \\
g & \text{otherwise}
\end{cases}$$
Trong project này: threshold = 0.5

#### 11.5.4. Label Smoothing
Szegedy et al. (2016) đề xuất **label smoothing** để prevent overconfident predictions:
$$y_{smooth} = (1 - \epsilon) \cdot y_{true} + \frac{\epsilon}{K}$$
Trong đó:
- $\epsilon = 0.1$: Smoothing factor
- $K = 2$: Number of classes

**Lợi ích:** Reduce overfitting, improve calibration.

### 11.6. Data Augmentation: Back Translation

#### 11.6.1. Motivation
Sennrich et al. (2016) giới thiệu back translation cho machine translation. Edunov et al. (2018) chỉ ra rằng back translation hiệu quả cho text augmentation:
- Preserve semantic meaning
- Introduce paraphrases
- Increase training data diversity

#### 11.6.2. Process
```
Original (English) → Translation (German/French) → Back-Translation (English)
```

**Ví dụ:**
- Original: "This movie is absolutely amazing!"
- German: "Dieser Film ist absolut erstaunlich!"
- Back: "This film is absolutely astonishing!"

**Lợi ích cho Sentiment Analysis:**
- Maintain sentiment polarity
- Generate diverse expressions
- Improve model robustness

### 11.7. Optimization và Training

#### 11.7.1. Adam Optimizer
Kingma & Ba (2014) đề xuất **Adam** (Adaptive Moment Estimation):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Hyperparameters:**
- $\alpha = 5 \times 10^{-4}$: Learning rate
- $\beta_1 = 0.9$: Exponential decay for first moment
- $\beta_2 = 0.999$: Exponential decay for second moment
- $\epsilon = 10^{-8}$: Numerical stability

#### 11.7.2. Learning Rate Scheduling
**ReduceLROnPlateau** giảm learning rate khi validation metric plateau:
$$\alpha_{new} = \alpha_{old} \times \text{factor}$$

**Cấu hình:**
- Factor: 0.5
- Patience: 3 epochs
- Min LR: $10^{-6}$

#### 11.7.3. Early Stopping
Prechelt (1998) chỉ ra rằng early stopping là effective regularization:
- Monitor validation F1 score
- Stop training nếu không cải thiện sau 8 epochs
- Restore best weights

---

## 12. TÀI LIỆU THAM KHẢO

### 12.1. Core Papers (LSTM & BiLSTM)

1. **Hochreiter, S., & Schmidhuber, J.** (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
   - Đề xuất kiến trúc LSTM để giải quyết vanishing gradient problem
   - Foundation paper cho LSTM networks

2. **Schuster, M., & Paliwal, K. K.** (1997). *Bidirectional Recurrent Neural Networks*. IEEE Transactions on Signal Processing, 45(11), 2673-2681.
   - Giới thiệu Bidirectional RNN architecture
   - Xử lý sequence theo cả hai hướng

3. **Bengio, Y., Simard, P., & Frasconi, P.** (1994). *Learning Long-Term Dependencies with Gradient Descent is Difficult*. IEEE Transactions on Neural Networks, 5(2), 157-166.
   - Phân tích vanishing/exploding gradient problem trong RNNs

4. **Gers, F. A., Schmidhuber, J., & Cummins, F.** (2000). *Learning to Forget: Continual Prediction with LSTM*. Neural Computation, 12(10), 2451-2471.
   - Cải tiến LSTM với forget gate

### 12.2. Sentiment Analysis

5. **Socher, R., et al.** (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. EMNLP.
   - Stanford Sentiment Treebank dataset
   - Recursive neural networks cho sentiment analysis

6. **Kim, Y.** (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP.
   - CNN cho text classification
   - Baseline cho nhiều sentiment analysis tasks

7. **Tang, D., Qin, B., & Liu, T.** (2015). *Document Modeling with Gated Recurrent Neural Network for Sentiment Classification*. EMNLP.
   - GRU và LSTM cho document-level sentiment

### 12.3. Word Embeddings

8. **Mikolov, T., et al.** (2013). *Efficient Estimation of Word Representations in Vector Space*. ICLR.
   - Giới thiệu Word2Vec (CBOW và Skip-gram)
   - Distributed word representations

9. **Pennington, J., Socher, R., & Manning, C. D.** (2014). *GloVe: Global Vectors for Word Representation*. EMNLP.
   - GloVe embeddings
   - Co-occurrence statistics

10. **Bojanowski, P., et al.** (2017). *Enriching Word Vectors with Subword Information*. TACL.
    - FastText embeddings
    - Subword information

### 12.4. Data Augmentation

11. **Edunov, S., et al.** (2018). *Understanding Back-Translation at Scale*. EMNLP.
    - Back translation cho data augmentation
    - Analysis of different translation models

12. **Sennrich, R., Haddow, B., & Birch, A.** (2016). *Improving Neural Machine Translation Models with Monolingual Data*. ACL.
    - Back translation cho machine translation
    - Pioneering work

13. **Wei, J., & Zou, K.** (2019). *EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks*. EMNLP.
    - Synonym replacement, random insertion/deletion
    - Simple but effective augmentation techniques

14. **Kobayashi, S.** (2018). *Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations*. NAACL.
    - Context-aware data augmentation

### 12.5. Regularization

15. **Srivastava, N., et al.** (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 15(1), 1929-1958.
    - Dropout regularization technique
    - Comprehensive analysis

16. **Szegedy, C., et al.** (2016). *Rethinking the Inception Architecture for Computer Vision*. CVPR.
    - Label smoothing regularization
    - Auxiliary classifiers

17. **Pascanu, R., Mikolov, T., & Bengio, Y.** (2013). *On the Difficulty of Training Recurrent Neural Networks*. ICML.
    - Gradient clipping for RNNs
    - Analysis of gradient flow

18. **Prechelt, L.** (1998). *Early Stopping - But When?* In Neural Networks: Tricks of the Trade. Springer.
    - Early stopping as regularization
    - Best practices

### 12.6. Optimization

19. **Kingma, D. P., & Ba, J.** (2014). *Adam: A Method for Stochastic Optimization*. ICLR.
    - Adam optimizer
    - Adaptive learning rates

20. **Loshchilov, I., & Hutter, F.** (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR.
    - Learning rate scheduling
    - Cosine annealing

### 12.7. Pooling Strategies

21. **Collobert, R., et al.** (2011). *Natural Language Processing (Almost) from Scratch*. JMLR, 12, 2493-2537.
    - Max pooling cho NLP
    - Convolutional architectures

22. **Wang, Y., et al.** (2016). *Attention-based LSTM for Aspect-level Sentiment Classification*. EMNLP.
    - Attention mechanisms
    - Multi-pooling strategies

### 12.8. Deep Learning Frameworks & Tools

23. **Paszke, A., et al.** (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
    - PyTorch framework

24. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
    - Scikit-learn library
    - Machine learning tools

### 12.9. Evaluation Metrics

25. **Sokolova, M., & Lapalme, G.** (2009). *A Systematic Analysis of Performance Measures for Classification Tasks*. Information Processing & Management, 45(4), 427-437.
    - Precision, Recall, F1-score analysis
    - Comprehensive evaluation metrics

26. **Davis, J., & Goadrich, M.** (2006). *The Relationship Between Precision-Recall and ROC Curves*. ICML.
    - ROC curves và PR curves
    - Performance evaluation

### 12.10. Best Practices & Surveys

27. **Goldberg, Y.** (2017). *Neural Network Methods for Natural Language Processing*. Morgan & Claypool Publishers.
    - Comprehensive NLP textbook
    - Deep learning cho NLP

28. **Zhang, L., Wang, S., & Liu, B.** (2018). *Deep Learning for Sentiment Analysis: A Survey*. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8(4), e1253.
    - Survey về deep learning cho sentiment analysis
    - State-of-the-art methods

29. **Young, T., et al.** (2018). *Recent Trends in Deep Learning Based Natural Language Processing*. IEEE Computational Intelligence Magazine, 13(3), 55-75.
    - Recent trends trong deep learning NLP
    - Comprehensive overview

### 12.11. Additional Resources

**Books:**
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). Pearson.

**Online Resources:**
- PyTorch Documentation: https://pytorch.org/docs/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Papers With Code (Sentiment Analysis): https://paperswithcode.com/task/sentiment-analysis

**Datasets:**
- IMDb Movie Reviews: Maas et al. (2011)
- Stanford Sentiment Treebank (SST): Socher et al. (2013)
- Yelp Reviews: Zhang et al. (2015)

---

**Người thực hiện:** [Tên của bạn]  
**Ngày hoàn thành:** 20/12/2025  
**Phiên bản:** 1.0  

---

*Báo cáo này tóm tắt toàn bộ quy trình từ xử lý dữ liệu, augmentation, encoding, training đến evaluation cho mô hình BiLSTM trong bài toán sentiment analysis.*
