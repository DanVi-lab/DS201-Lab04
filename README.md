# LAB 4 – Encoder–Decoder và Attention (Anh → Việt)

Dự án này xây dựng và so sánh ba mô hình sequence-to-sequence dịch máy từ tiếng Anh sang tiếng Việt trên bộ dữ liệu **PhoMT**, sử dụng PyTorch trên GPU.

---

## Mục tiêu

| Bài | Mô hình | Mô tả |
|-----|---------|-------|
| **Bài 1** | Encoder–Decoder LSTM (không Attention) | Baseline: encoder nén toàn bộ chuỗi nguồn vào một vector ngữ cảnh cố định |
| **Bài 2** | Encoder–Decoder + **Bahdanau Attention** | Cơ chế attention cộng-tính (additive) học trọng số căn chỉnh |
| **Bài 3** | Encoder–Decoder + **Luong General Attention** | Cơ chế attention tích vô hướng (multiplicative/general) |

---

## Cấu hình thí nghiệm

| Tham số | Giá trị |
|---------|---------|
| Dữ liệu | small-PhoMT (train: 18 607 / dev: 1 901 / test: 1 786 câu, lọc ≤ 50 từ) |
| Embedding size | 256 |
| Hidden size | 256 |
| Số lớp LSTM | 3 (Encoder & Decoder) |
| Dropout | 0.3 |
| Optimizer | Adam, lr = 0.001 |
| Batch size | 32 |
| Số epoch tối đa | 10 |
| Teacher forcing ratio | 0.5 |
| Vocab limit | 10 000 (tiếng Anh: 8 943 từ / tiếng Việt: 4 003 từ) |
| Giải mã | Greedy decoding |
| Chọn checkpoint | ROUGE-L F1 cao nhất trên **dev** |

---

## Kết quả

### So sánh trên Dev & Test

| Mô hình | ROUGE-L Dev | ROUGE-L Test | BLEU Test | Tham số | Giây/epoch |
|---------|:-----------:|:------------:|:---------:|--------:|:----------:|
| Không Attention | 0.2299 | 0.2354 | 2.69 | 7 500 963 | 65.18 |
| Bahdanau Attention | 0.2918 | 0.3046 | 5.40 | 8 025 763 | 105.04 |
| **Luong General** | **0.2985** | **0.3093** | **5.61** | 7 959 971 | 98.34 |

> **Checkpoint tốt nhất** của cả ba mô hình đều là epoch 10 (mô hình vẫn tiếp tục cải thiện).

### Nhận xét nhanh

- Cả hai cơ chế Attention (Bahdanau và Luong) đều **vượt trội rõ rệt** so với baseline không có Attention trên cả ROUGE-L và BLEU.
- **Luong General Attention** đạt ROUGE-L test cao nhất (0.3093), nhỉnh hơn Bahdanau (0.3046) nhưng sử dụng **ít tham số hơn** và **nhanh hơn** một chút (~7% giây/epoch).
- Mô hình không Attention nhanh nhất (~65 s/epoch) nhưng chất lượng dịch thấp hơn đáng kể.

---

## Hướng dẫn chạy

### Trên Kaggle

1. Upload notebook `lab04.ipynb` lên [Kaggle Notebooks](https://www.kaggle.com/code).
2. Thêm dataset `danvi12/phomt-dataset` (small-PhoMT) vào notebook.
3. Bật GPU: **Settings → Accelerator → GPU T4 x2** (hoặc GPU P100).
4. Chạy toàn bộ notebook từ đầu đến cuối.
5. Notebook xuất file `artifacts/` chứa checkpoint `.pt`, bảng metrics CSV, `learning_curves.png` và heatmap attention.

### Trên Google Colab

1. Mở `lab04.ipynb` trên Google Colab.
2. Vào **Runtime → Change runtime type → GPU**.
3. Tải dữ liệu small-PhoMT từ [Google Drive của đề](https://drive.google.com/drive/folders/186OAOuSEYEDVcry7WP5UBdqECXo26QAb) (cần quyền truy cập).
4. Chỉnh đường dẫn `DATA` trong notebook cho phù hợp rồi chạy tuần tự.

### Cài đặt thư viện (nếu chạy local)

```bash
pip install -r requirements.txt
```

---

## Artifact đầu ra

Sau khi chạy xong, notebook tạo ra:

| File | Mô tả |
|------|-------|
| `none.pt` | Checkpoint tốt nhất – mô hình không Attention |
| `bahdanau.pt` | Checkpoint tốt nhất – Bahdanau Attention |
| `luong.pt` | Checkpoint tốt nhất – Luong General Attention |
| `learning_curves.png` | Đồ thị Train Loss & Dev ROUGE-L theo epoch |
| `comparison_dev.csv` | Bảng so sánh trên tập Dev |
| `comparison_test.csv` | Bảng so sánh cuối cùng trên tập Test |
| `translations.csv` | Ba câu dịch mẫu của cả ba mô hình |
| `attention_bahdanau_*.png` | Attention heatmap – Bahdanau (3 câu) |
| `attention_luong_*.png` | Attention heatmap – Luong (3 câu) |

> **Không commit** các file checkpoint `.pt` lớn lên GitHub. Chỉ commit notebook có output, hình ảnh và CSV nếu cần.

---

## Kiến trúc mô hình

```
Encoder (3-layer LSTM)
  └─ Embedding (8943 × 256)
  └─ Dropout (0.3)
  └─ LSTM (256 → 256, 3 layers, dropout=0.3)
  └─ Output: memory [B, T, 256], hidden state

Attention (Bahdanau hoặc Luong)
  └─ Bahdanau: score = v · tanh(Wq·query + Wk·key)
  └─ Luong General: score = memory · W · query

Decoder (3-layer LSTM)
  └─ Embedding (4003 × 256)
  └─ LSTM input = emb (+ context nếu có Attention)
  └─ Fuse layer: tanh(W · [query; context]) nếu có Attention
  └─ FC: 256 → 4003 (vocab_tgt)
```

---

## Tài liệu tham khảo

- Bahdanau et al. (2015) – *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong et al. (2015) – *Effective Approaches to Attention-based Neural Machine Translation*
