<div align="center">

# 🏗️ TWO-TOWER RECOMMENDATION SYSTEM

### Hệ thống Gợi ý Sản phẩm Tháp Đôi tích hợp LLM

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792EE5?style=for-the-badge&logo=lightning&logoColor=white)](https://lightning.ai)
[![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com)
[![Apache Iceberg](https://img.shields.io/badge/Apache_Iceberg-Lakehouse-blue?style=for-the-badge)](https://iceberg.apache.org)
[![Feast](https://img.shields.io/badge/Feast-Feature_Store-orange?style=for-the-badge)](https://feast.dev)

---

*Hệ thống gợi ý sản phẩm quy mô lớn trên tập dữ liệu **Amazon Reviews**, sử dụng kiến trúc **Two-Tower Neural Network** kết hợp **LLM Reranking**, xây dựng trên nền tảng **Data Lakehouse** (Medallion Architecture) và tối ưu vận hành theo nguyên tắc **FinOps**.*

</div>

---

## 📑 Mục lục

- [Tổng quan](#-tổng-quan)
- [1. Sơ đồ Kiến trúc Tổng thể](#1--sơ-đồ-kiến-trúc-tổng-thể-high-level-system-architecture)
- [2. Sơ đồ Luồng Dữ liệu & Quản lý Đặc trưng](#2--sơ-đồ-luồng-dữ-liệu--quản-lý-đặc-trưng-data--feature-pipeline)
- [3. Sơ đồ Kiến trúc Liên kết Tháp Đôi](#3--sơ-đồ-kiến-trúc-liên-kết-tháp-đôi-two-tower-neural-network)
- [4. Sơ đồ Quá trình Suy luận Thời gian thực](#4--sơ-đồ-quá-trình-suy-luận-thời-gian-thực-real-time-inference)
- [5. Sơ đồ Phân bổ Hạ tầng & FinOps](#5--sơ-đồ-phân-bổ-hạ-tầng--finops-cloud-infra--finops-topology)
- [Cấu trúc Mã nguồn](#-cấu-trúc-mã-nguồn)
- [Hướng dẫn Cài đặt & Sử dụng](#-hướng-dẫn-cài-đặt--sử-dụng)
- [Trạng thái Dự án](#-trạng-thái-dự-án)

---

## 🔭 Tổng quan

Dự án xây dựng một **hệ thống gợi ý (Recommendation System)** end-to-end với ba tầng phễu chính:

| Tầng | Vai trò | Công nghệ | Độ trễ mục tiêu |
|------|---------|------------|:----------------:|
| **Retrieval** | Truy xuất Top-K ứng viên từ hàng triệu sản phẩm | Two-Tower + Qdrant (ANN) | < 20ms |
| **Ranking** | Xếp hạng lại bằng LLM với ngữ cảnh cá nhân hóa | Gemini / OpenAI API | < 500ms |
| **Serving** | Trả kết quả cuối cùng cho người dùng | Cloud Run / FastAPI | < 50ms |

**Điểm nổi bật kỹ thuật:**
- 🧊 **Data Lakehouse** kiến trúc Medallion (Bronze → Silver → Gold) trên Apache Iceberg + PySpark
- 🏗️ **Two-Tower Architecture** với Item Tower đóng băng (BAAI/bge-small-en-v1.5) và User Tower huấn luyện bằng MLP
- 🧠 **Feature Store** (Feast) tách biệt Online/Offline Store chống Data Leakage
- 💰 **FinOps-first**: Tách GPU (Lightning.ai) / CPU (GCP), Semantic Caching cho LLM

---

## 1. 🏛️ Sơ đồ Kiến trúc Tổng thể (High-Level System Architecture)

> Sơ đồ mô tả toàn cảnh hệ thống từ nguồn dữ liệu thô đến phục vụ người dùng cuối, chia thành các tầng xử lý rõ ràng.

```mermaid
graph TB
    subgraph DATA_SOURCES["🗄️ NGUỒN DỮ LIỆU"]
        direction LR
        DS1["📦 Đánh giá Amazon<br/>Dữ liệu Thô"]
        DS2["📦 Siêu dữ liệu Sản phẩm<br/>Tiêu đề, Danh mục, Giá"]
    end

    subgraph DATA_LAKEHOUSE["🧊 DATA LAKEHOUSE — Kiến trúc Medallion"]
        direction LR
        BRONZE["🥉 Tầng Đồng (Bronze)<br/>━━━━━━━━━━━━━<br/>Nạp dữ liệu thô<br/>Apache Iceberg<br/>PySpark"]
        SILVER["🥈 Tầng Bạc (Silver)<br/>━━━━━━━━━━━━━<br/>Làm sạch & Xác thực<br/>Khử trùng lặp<br/>Ép kiểu dữ liệu"]
        GOLD["🥇 Tầng Vàng (Gold)<br/>━━━━━━━━━━━━━<br/>Sẵn sàng tạo đặc trưng<br/>Thống kê Người dùng<br/>Ngữ cảnh Sản phẩm"]
        BRONZE -->|"Làm sạch<br/>& Xác thực"| SILVER
        SILVER -->|"Tổng hợp<br/>& Làm giàu"| GOLD
    end

    subgraph FEATURE_STORE["🏪 FEAST — Cửa hàng Đặc trưng (Feature Store)"]
        direction TB
        OFFLINE["📁 Lưu trữ Ngoại tuyến<br/>━━━━━━━━━━━━━<br/>GCS Parquet<br/>Dữ liệu Huấn luyện"]
        ONLINE["⚡ Lưu trữ Trực tuyến<br/>━━━━━━━━━━━━━<br/>GCP Datastore<br/>Phục vụ Độ trễ thấp"]
        REGISTRY["📋 Sổ đăng ký<br/>Lưu trên GCS"]
        OFFLINE -.->|"feast materialize"| ONLINE
    end

    subgraph ML_TRAINING["🧠 NỀN TẢNG HUẤN LUYỆN ML — Lightning.ai GPU L4"]
        direction TB
        ITEM_TOWER["🗼 Tháp Sản phẩm<br/>━━━━━━━━━━━━━<br/>BAAI/bge-small-en-v1.5<br/>Vector 384 chiều<br/>🔒 ĐÓNG BĂNG"]
        USER_TOWER["🗼 Tháp Người dùng<br/>━━━━━━━━━━━━━<br/>Mạng MLP<br/>PyTorch Lightning<br/>🔥 CÓ THỂ HUẤN LUYỆN"]
        CHECKPOINT["💾 Điểm lưu<br/>Mô hình (Checkpoint)"]
        USER_TOWER -->|"save_top_k=1"| CHECKPOINT
    end

    subgraph INFERENCE["🔮 SUY LUẬN & PHỤC VỤ (INFERENCE & SERVING)"]
        direction TB
        QDRANT["🔍 Qdrant<br/>Cơ sở dữ liệu Vector<br/>━━━━━━━━━━━━━<br/>Tìm kiếm ANN<br/>Chỉ dùng CPU"]
        LLM["🤖 API LLM<br/>━━━━━━━━━━━━━<br/>Gemini / OpenAI<br/>Xếp hạng lại<br/>AI Giải thích được"]
        CACHE["💨 Bộ nhớ đệm<br/>Ngữ nghĩa"]
        API["🌐 API Phục vụ<br/>━━━━━━━━━━━━━<br/>FastAPI<br/>Cloud Run"]
    end

    subgraph END_USER["👤 NGƯỜI DÙNG CUỐI"]
        USER["🧑‍💻 Ứng dụng Máy khách"]
    end

    DS1 & DS2 --> BRONZE
    GOLD -->|"item_text_context"| ITEM_TOWER
    GOLD -->|"user_features"| OFFLINE
    ITEM_TOWER -->|"Vector nhúng 384 chiều<br/>Parquet → GCS"| QDRANT
    OFFLINE -->|"Đặc trưng Ngoại tuyến"| USER_TOWER
    CHECKPOINT -->|"Trọng số đã huấn luyện"| API
    ONLINE -->|"Đặc trưng Người dùng<br/>Thời gian thực"| API
    API -->|"Vector Người dùng"| QDRANT
    QDRANT -->|"Top-100<br/>Ứng viên"| LLM
    LLM <-->|"Tối ưu chi phí (FinOps)"| CACHE
    LLM -->|"Top-10<br/>Đã xếp hạng lại"| API
    API -->|"Gợi ý<br/>+ Lời giải thích"| USER

    classDef bronze fill:#cd7f32,stroke:#8b5a2b,color:#fff
    classDef silver fill:#c0c0c0,stroke:#808080,color:#333
    classDef gold fill:#ffd700,stroke:#daa520,color:#333
    classDef feast fill:#ff6b35,stroke:#c44900,color:#fff
    classDef training fill:#7c3aed,stroke:#5b21b6,color:#fff
    classDef inference fill:#0ea5e9,stroke:#0284c7,color:#fff

    class BRONZE bronze
    class SILVER silver
    class GOLD gold
    class OFFLINE,ONLINE,REGISTRY feast
    class ITEM_TOWER,USER_TOWER,CHECKPOINT training
    class QDRANT,LLM,CACHE,API inference
```

**Giải thích các tầng:**

| Ký hiệu | Tầng | Chức năng |
|:--------:|-------|-----------|
| 🗄️ | **Nguồn dữ liệu** | Tập dữ liệu Amazon Reviews thô, bao gồm đánh giá và metadata sản phẩm |
| 🧊 | **Data Lakehouse** | Xử lý dữ liệu theo kiến trúc Medallion (Bronze → Silver → Gold) bằng Apache Iceberg + PySpark |
| 🏪 | **Feature Store** | Feast quản lý đặc trưng, tách biệt Offline Store (Training) và Online Store (Serving) |
| 🧠 | **ML Training** | Huấn luyện mô hình Two-Tower trên GPU L4 (Lightning.ai) |
| 🔮 | **Inference & Serving** | Vector search (Qdrant) + LLM reranking + API serving |

---

## 2. 📊 Sơ đồ Luồng Dữ liệu & Quản lý Đặc trưng (Data & Feature Pipeline)

> Sơ đồ chi tiết quá trình biến đổi dữ liệu thô thành đặc trưng sẵn sàng cho huấn luyện và phục vụ, tuân thủ nguyên tắc chống rò rỉ dữ liệu (Data Leakage Prevention).

```mermaid
flowchart LR
    subgraph RAW["📥 DỮ LIỆU THÔ"]
        R1["Đánh giá Amazon<br/>JSON/CSV"]
        R2["Siêu dữ liệu Sản phẩm<br/>JSON"]
    end

    subgraph BRONZE_LAYER["🥉 TẦNG ĐỒNG — Nạp dữ liệu thô"]
        B1["Trình đọc PySpark<br/>━━━━━━━━━━━━━━━<br/>• Suy luận Lược đồ<br/>• Phân vùng theo Danh mục<br/>• Định dạng Apache Iceberg"]
    end

    subgraph SILVER_LAYER["🥈 TẦNG BẠC — Làm sạch & Xác thực"]
        S1["Chất lượng Dữ liệu<br/>━━━━━━━━━━━━━━━<br/>• Xử lý giá trị Null<br/>• Khử trùng lặp<br/>• Ép kiểu dữ liệu<br/>• Phát hiện Ngoại lệ"]
        S2["Phân giải Thực thể<br/>━━━━━━━━━━━━━━━<br/>• Chuẩn hóa user_id<br/>• Ánh xạ item_id<br/>• Căn chỉnh Thời gian"]
    end

    subgraph GOLD_LAYER["🥇 TẦNG VÀNG — Kỹ nghệ Đặc trưng"]
        direction TB
        G1["👤 Đặc trưng Người dùng<br/>━━━━━━━━━━━━━━━<br/>• total_reviews: Int64<br/>• avg_rating_given: Float32<br/>• stddev_rating_given: Float32"]
        G2["📦 Đặc trưng Sản phẩm<br/>━━━━━━━━━━━━━━━<br/>• item_text_context: String<br/>  ↳ tiêu đề + danh mục +<br/>    tóm tắt đánh giá tb"]
        G3["🎯 Dữ liệu Huấn luyện<br/>━━━━━━━━━━━━━━━<br/>• user_id, item_id<br/>• nhãn (label): Nhị phân (0/1)<br/>• Lấy mẫu Âm"]
    end

    subgraph FEAST_STORE["🏪 CỬA HÀNG ĐẶC TRƯNG FEAST"]
        direction TB
        FS_DEF["📋 Định nghĩa Đặc trưng<br/>━━━━━━━━━━━━━━━<br/>features.py<br/>━━━━━━━━━━━━━━━<br/>Thực thể: user_id (STRING)<br/>View: user_statistical_features<br/>TTL: 3650 ngày"]
        FS_OFFLINE["📁 Lưu trữ Ngoại tuyến<br/>━━━━━━━━━━━━━━━<br/>File Parquet trên GCS<br/>✅ Cho HUẤN LUYỆN<br/>🔒 Kết nối đúng thời điểm"]
        FS_ONLINE["⚡ Lưu trữ Trực tuyến<br/>━━━━━━━━━━━━━━━<br/>GCP Datastore<br/>✅ Cho PHỤC VỤ<br/>⚡ Tra cứu Độ trễ thấp"]
        FS_REG["📡 Sổ đăng ký<br/>Lưu trên GCS"]
        FS_DEF --> FS_OFFLINE & FS_ONLINE
        FS_OFFLINE -->|"feast materialize<br/>━━━━━━━━━━━━━━━<br/>Đồng bộ theo Lô"| FS_ONLINE
    end

    subgraph EMBEDDING_PIPELINE["⚡ ĐƯỜNG ỐNG NHÚNG (EMBEDDING PIPELINE)"]
        direction TB
        EMB_MODEL["🤖 BAAI/bge-small-en-v1.5<br/>━━━━━━━━━━━━━━━<br/>SentenceTransformer<br/>FP16 • GPU L4"]
        EMB_CHUNK["📦 Xử lý theo Khối (Chunk)<br/>━━━━━━━━━━━━━━━<br/>chunk_size: 100,000<br/>batch_size: 512<br/>Đọc trễ bằng PyArrow"]
        EMB_UPLOAD["☁️ Tải lên Bất đồng bộ<br/>━━━━━━━━━━━━━━━<br/>ThreadPool (4 luồng)<br/>Gửi-và-Quên<br/>Nén Snappy"]
        EMB_OUTPUT["💎 Vector Sản phẩm<br/>━━━━━━━━━━━━━━━<br/>Float32 384 chiều<br/>Parquet trên GCS<br/>Tổng ~14GB"]
        EMB_MODEL --> EMB_CHUNK --> EMB_UPLOAD --> EMB_OUTPUT
    end

    subgraph TRAINING_DATA["🎯 ĐẦU RA HUẤN LUYỆN"]
        TD["training_data/<br/>part-00000.parquet<br/>━━━━━━━━━━━━━━━<br/>user_features ⊕<br/>item_embeddings ⊕<br/>nhãn nhị phân"]
    end

    R1 & R2 --> B1
    B1 --> S1 --> S2
    S2 --> G1 & G2 & G3
    G1 --> FS_DEF
    G2 --> EMB_MODEL
    G3 --> TD
    FS_OFFLINE -.->|"🔒 CHỈ HUẤN LUYỆN<br/>Chống rò rỉ dữ liệu"| TD
    EMB_OUTPUT -.->|"Vector Sản phẩm"| TD

    style RAW fill:#1e293b,stroke:#475569,color:#e2e8f0
    style BRONZE_LAYER fill:#92400e,stroke:#78350f,color:#fef3c7
    style SILVER_LAYER fill:#6b7280,stroke:#4b5563,color:#f3f4f6
    style GOLD_LAYER fill:#854d0e,stroke:#713f12,color:#fef9c3
    style FEAST_STORE fill:#9a3412,stroke:#7c2d12,color:#fed7aa
    style EMBEDDING_PIPELINE fill:#581c87,stroke:#3b0764,color:#e9d5ff
    style TRAINING_DATA fill:#065f46,stroke:#064e3b,color:#a7f3d0
```

**Nguyên tắc chống Data Leakage:**

```
┌─────────────────────────────────────────────────────────────┐
│  🔒 TRAINING PHASE          │  ⚡ SERVING PHASE             │
│  ─────────────────           │  ──────────────────           │
│  ✅ Offline Store (Parquet)  │  ✅ Online Store (Datastore)  │
│  ✅ Point-in-time Join       │  ✅ Real-time Lookup          │
│  ❌ KHÔNG dùng Online Store  │  ❌ KHÔNG dùng Offline Store  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 🧠 Sơ đồ Kiến trúc Liên kết Tháp Đôi (Two-Tower Neural Network)

> Sơ đồ mô tả chi tiết kiến trúc mạng nơ-ron Tháp Đôi, bao gồm cấu trúc từng tầng, kích thước tensor, và cơ chế tương tác giữa hai tháp.

```mermaid
graph TB
    subgraph INPUT_LAYER["📥 TẦNG ĐẦU VÀO"]
        direction LR
        UI["👤 Đầu vào Người dùng<br/>━━━━━━━━━━━━━━━━<br/>• total_reviews<br/>• avg_rating_given<br/>• stddev_rating_given<br/>━━━━━━━━━━━━━━━━<br/>Tensor: [batch, 3]<br/>dtype: float32"]
        II["📦 Đầu vào Sản phẩm<br/>━━━━━━━━━━━━━━━━<br/>item_text_context<br/>━━━━━━━━━━━━━━━━<br/>Kiểu: Chuỗi (String)<br/>Trung bình ~200 tokens"]
    end

    subgraph USER_TOWER_DETAIL["🗼 THÁP NGƯỜI DÙNG — MLP Có thể Huấn luyện"]
        direction TB
        U_FC1["🔵 Tầng Tuyến tính 1<br/>━━━━━━━━━━━━━━━━<br/>nn.Linear(3, 128)<br/>Tham số: 512"]
        U_BN1["📊 BatchNorm1d(128)<br/>━━━━━━━━━━━━━━━━<br/>Chuẩn hóa Hoạt hóa"]
        U_ACT1["⚡ Hàm kích hoạt ReLU<br/>━━━━━━━━━━━━━━━━<br/>Tính phi tuyến"]
        U_DROP["💧 Dropout(0.2)<br/>━━━━━━━━━━━━━━━━<br/>Chống quá khớp (Regularization)"]
        U_FC2["🔵 Tầng Tuyến tính 2<br/>━━━━━━━━━━━━━━━━<br/>nn.Linear(128, 384)<br/>Tham số: 49,536"]
        U_OUT["🎯 Vector Người dùng<br/>━━━━━━━━━━━━━━━━<br/>Tensor: [batch, 384]"]

        U_FC1 --> U_BN1 --> U_ACT1 --> U_DROP --> U_FC2 --> U_OUT
    end

    subgraph ITEM_TOWER_DETAIL["🗼 THÁP SẢN PHẨM — Bộ mã hóa Đóng băng"]
        direction TB
        I_TOK["✂️ Bộ tách từ (Tokenizer)<br/>━━━━━━━━━━━━━━━━<br/>WordPiece<br/>Độ dài Tối đa: 512"]
        I_BERT["🧊 BAAI/bge-small-en-v1.5<br/>━━━━━━━━━━━━━━━━<br/>Bộ mã hóa Transformer<br/>6 Tầng, Kích thước 384<br/>🔒 ĐÓNG BĂNG MỌI TRỌNG SỐ"]
        I_POOL["🔄 Gom nhóm Trung bình<br/>━━━━━━━━━━━━━━━━<br/>+ Chuẩn hóa L2"]
        I_OUT["🎯 Vector Sản phẩm<br/>━━━━━━━━━━━━━━━━<br/>Tensor: [batch, 384]<br/>Chuẩn hóa ∈ hình cầu đơn vị"]

        I_TOK --> I_BERT --> I_POOL --> I_OUT
    end

    subgraph INTERACTION["🤝 TẦNG TƯƠNG TÁC"]
        direction TB
        DOT["⊙ Tích Vô hướng (Dot Product)<br/>━━━━━━━━━━━━━━━━<br/>logits = Σ(user ⊙ item)<br/>Tensor: [batch]"]
        LOSS["📉 BCEWithLogitsLoss<br/>━━━━━━━━━━━━━━━━<br/>Entropy Chéo Nhị phân<br/>kèm Sigmoid"]
        LABEL["🏷️ Nhãn (Labels)<br/>━━━━━━━━━━━━━━━━<br/>1 = Tương tác Tích cực<br/>0 = Mẫu Âm"]
    end

    subgraph OPTIMIZER["⚙️ CẤU HÌNH HUẤN LUYỆN"]
        direction LR
        OPT["🔧 Bộ tối ưu hóa AdamW<br/>━━━━━━━━━━━━━━━━<br/>lr: 1e-3<br/>weight_decay: 1e-4"]
        PREC["⚡ Độ chính xác Hỗn hợp<br/>━━━━━━━━━━━━━━━━<br/>16-bit (FP16)<br/>Tối ưu cho GPU L4"]
        BATCH["📦 DataLoader<br/>━━━━━━━━━━━━━━━━<br/>batch_size: 4096<br/>workers: 4<br/>pin_memory: True"]
    end

    UI --> U_FC1
    II --> I_TOK
    U_OUT & I_OUT --> DOT
    DOT --> LOSS
    LABEL --> LOSS
    LOSS -.->|"Lan truyền ngược<br/>∇ Hàm mất mát"| U_FC1

    style USER_TOWER_DETAIL fill:#4c1d95,stroke:#6d28d9,color:#ede9fe
    style ITEM_TOWER_DETAIL fill:#164e63,stroke:#155e75,color:#cffafe
    style INTERACTION fill:#7c2d12,stroke:#9a3412,color:#fed7aa
    style INPUT_LAYER fill:#1e293b,stroke:#334155,color:#e2e8f0
    style OPTIMIZER fill:#14532d,stroke:#166534,color:#bbf7d0
```

**Tóm tắt Kiến trúc:**

```
USER FEATURES ──► [3] ──► Linear(128) ──► BN ──► ReLU ──► Dropout ──► Linear(384) ──► User Vector ─┐
                                                                                                     ├──► Dot Product ──► σ(logit) ──► Loss
ITEM TEXT ──────► Tokenize ──► BGE-small (Frozen, 384-dim) ──► L2 Norm ──► Item Vector ──────────────┘
```

| Thành phần | Tham số | Chi tiết |
|------------|---------|----------|
| **User Tower** | ~50,048 params | 2-layer MLP, Trainable |
| **Item Tower** | ~33M params | Pre-trained BERT, Frozen |
| **Output Dim** | 384 | Cả hai tháp cùng chiều không gian |
| **Loss** | BCEWithLogitsLoss | Phân loại nhị phân (tương tác / không tương tác) |
| **Optimizer** | AdamW | lr=1e-3, weight_decay=1e-4 |

---

## 4. 🔄 Sơ đồ Quá trình Suy luận Thời gian thực (Real-time Inference)

> Sơ đồ tuần tự mô tả luồng xử lý từ khi người dùng gửi yêu cầu cho đến khi nhận được danh sách gợi ý cuối cùng, bao gồm các bước truy xuất đặc trưng, sinh vector, tìm kiếm láng giềng gần nhất, và reranking bằng LLM.

```mermaid
sequenceDiagram
    autonumber

    actor User as 👤 Người dùng
    participant API as 🌐 API Phục vụ<br/>(FastAPI / Cloud Run)
    participant Feast as 🏪 Feast<br/>Lưu trữ Trực tuyến
    participant UT as 🗼 Tháp Người dùng<br/>(Mô hình MLP)
    participant Qdrant as 🔍 Qdrant<br/>Vector DB
    participant Cache as 💨 Bộ nhớ đệm<br/>Ngữ nghĩa
    participant LLM as 🤖 API LLM<br/>(Gemini)

    Note over User,LLM: ━━━━━━━ GIAI ĐOẠN 1: TRUY XUẤT (Mục tiêu < 20ms) ━━━━━━━

    User->>+API: GET /recommend?user_id=U123
    
    API->>+Feast: get_online_features(user_id="U123")
    Note right of Feast: Truy xuất từ<br/>GCP Datastore<br/>Độ trễ ~5ms
    Feast-->>-API: {total_reviews, avg_rating, stddev_rating}

    API->>+UT: forward(user_features=[12, 4.2, 0.8])
    Note right of UT: Suy luận MLP<br/>[3] → [128] → [384]<br/>~1ms trên CPU
    UT-->>-API: user_vector: Float32[384]

    API->>+Qdrant: search(vector=user_vector, top_k=100)
    Note right of Qdrant: Láng giềng Gần nhất<br/>Xấp xỉ (ANN)<br/>~10ms Chỉ dùng CPU
    Qdrant-->>-API: top_100_items: [{id, score, metadata}]

    Note over User,LLM: ━━━━━━━ GIAI ĐOẠN 2: XẾP HẠNG LẠI BẰNG LLM (Mục tiêu < 500ms) ━━━━━━━

    API->>+Cache: lookup(user_context + top_100_hash)
    
    alt Cache HIT ✅
        Cache-->>API: cached_response: Top-10 + Explanations
        Note over Cache: 💰 FinOps: Tiết kiệm<br/>1 lệnh gọi API (~$0.01)
    else Cache MISS ❌
        Cache-->>-API: null
        
        API->>+LLM: rerank(user_profile, top_100_items)
        Note right of LLM: Kỹ nghệ Prompt:<br/>• Sở thích Người dùng<br/>• Mô tả Sản phẩm<br/>• Ràng buộc Đa dạng<br/>~300ms
        LLM-->>-API: reranked_top_10 + explanations
        
        API->>Cache: store(key, reranked_response, TTL=1h)
    end

    Note over User,LLM: ━━━━━━━ GIAI ĐOẠN 3: PHỤC VỤ (SERVING) ━━━━━━━

    API-->>-User: 📋 Phản hồi JSON
    
    Note over User: {<br/>  "recommendations": [...top_10...],<br/>  "explanations": [...ai_generated...],<br/>  "latency_ms": 342<br/>}
```

**Breakdown độ trễ từng bước:**

```
┌──────────────────────────────────────────────────────────────┐
│  RETRIEVAL PHASE                              Total: ~16ms   │
│  ├── Feast Online Lookup ················· ~5ms              │
│  ├── User Tower MLP Inference ············ ~1ms              │
│  └── Qdrant ANN Search (Top-100) ········· ~10ms             │
│                                                              │
│  RERANKING PHASE                              Total: ~300ms  │
│  ├── Semantic Cache Lookup ··············· ~2ms              │
│  ├── LLM API Call (if cache miss) ········ ~300ms            │
│  └── Cache Store ························· ~1ms              │
│                                                              │
│  END-TO-END (with cache hit) ··············· ~20ms           │
│  END-TO-END (with cache miss) ·············· ~320ms          │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. ☁️ Sơ đồ Phân bổ Hạ tầng & FinOps (Cloud Infra & FinOps Topology)

> Sơ đồ mô tả cách phân bổ tài nguyên hạ tầng đám mây theo nguyên tắc FinOps — tách biệt môi trường GPU (Training) và CPU (Serving), tối ưu chi phí trên Google Cloud $300 credit.

```mermaid
graph TB
    subgraph FINOPS_PRINCIPLES["💰 NGUYÊN TẮC FINOPS"]
        direction LR
        P1["🎯 Tận dụng tối đa<br/>hệ sinh thái MIỄN PHÍ"]
        P2["📏 Cấp phát đúng cỡ<br/>mọi tài nguyên"]
        P3["🚫 Không nâng<br/>GPU Quota trên GCP"]
    end

    subgraph LIGHTNING_AI["⚡ LIGHTNING.AI — Tác vụ GPU"]
        direction TB
        
        subgraph GPU_COMPUTE["🎮 Máy ảo GPU L4"]
            TRAIN_JOB["🏋️ Tác vụ Huấn luyện<br/>━━━━━━━━━━━━━━━<br/>Tháp Người dùng MLP<br/>PyTorch Lightning<br/>Độ chính xác Hỗn hợp FP16<br/>batch_size: 4096<br/>max_epochs: 10"]
            EMB_JOB["🔄 Nhúng theo Lô<br/>━━━━━━━━━━━━━━━<br/>Tháp Sản phẩm (Đóng băng)<br/>BGE-small-en-v1.5<br/>Khối: 100K Sản phẩm<br/>Đầu ra ~14GB"]
        end

        LA_COST["💳 Chi phí: Đã bao gồm<br/>trong gói Lightning.ai"]
    end

    subgraph GCP_CLOUD["☁️ NỀN TẢNG GOOGLE CLOUD — Tác vụ Chỉ dùng CPU"]
        direction TB
        
        subgraph GCS_STORAGE["🗄️ Google Cloud Storage"]
            GCS1["📦 Data Lakehouse<br/>━━━━━━━━━━━━━━━<br/>Đồng/Bạc/Vàng<br/>Apache Iceberg"]
            GCS2["📦 Vector Sản phẩm<br/>━━━━━━━━━━━━━━━<br/>Parquet 384 chiều<br/>~14GB"]
            GCS3["📦 Sổ đăng ký Feast<br/>━━━━━━━━━━━━━━━<br/>Định nghĩa Đặc trưng"]
            GCS4["📦 Tạo tác Mô hình<br/>━━━━━━━━━━━━━━━<br/>Điểm lưu (.ckpt)"]
        end

        subgraph GCP_COMPUTE["🖥️ Máy tính CPU"]
            CLOUD_RUN["🌐 Cloud Run<br/>━━━━━━━━━━━━━━━<br/>API Phục vụ<br/>FastAPI Container<br/>Tự động mở rộng 0→N<br/>💰 Trả theo Yêu cầu"]
            QDRANT_SVC["🔍 Dịch vụ Qdrant<br/>━━━━━━━━━━━━━━━<br/>Cơ sở dữ liệu Vector<br/>Chế độ Chỉ dùng CPU<br/>Chỉ mục ANN"]
        end

        subgraph GCP_DB["🗃️ Cơ sở dữ liệu Quản lý"]
            DATASTORE["⚡ Cloud Datastore<br/>━━━━━━━━━━━━━━━<br/>Lưu trữ Trực tuyến Feast<br/>Tra cứu Khóa-Giá trị<br/>💰 Nằm trong Dải Miễn phí"]
        end

        GCP_COST["💳 Ngân sách: $300 Credit<br/>━━━━━━━━━━━━━━━<br/>⚠️ Nghiêm cấm kích hoạt<br/>thanh toán ngoài credit"]
    end

    subgraph EXTERNAL_API["🌍 CÁC API BÊN NGOÀI"]
        GEMINI["🤖 API Gemini<br/>━━━━━━━━━━━━━━━<br/>LLM Xếp hạng lại<br/>+ Bộ nhớ đệm Ngữ nghĩa<br/>💰 Trả theo Token"]
    end

    %% Connections
    TRAIN_JOB -->|"Tải lên Điểm lưu"| GCS4
    EMB_JOB -->|"Tải lên Vector nhúng"| GCS2
    GCS2 -->|"Đường ống Nạp dữ liệu<br/>Chỉ dùng CPU"| QDRANT_SVC
    GCS4 -->|"Tải Trọng số"| CLOUD_RUN
    DATASTORE <-->|"Phục vụ Đặc trưng"| CLOUD_RUN
    CLOUD_RUN <-->|"Truy vấn ANN"| QDRANT_SVC
    CLOUD_RUN <-->|"Yêu cầu Xếp hạng lại"| GEMINI

    style FINOPS_PRINCIPLES fill:#fbbf24,stroke:#d97706,color:#1c1917
    style LIGHTNING_AI fill:#7c3aed,stroke:#6d28d9,color:#ede9fe
    style GCP_CLOUD fill:#2563eb,stroke:#1d4ed8,color:#dbeafe
    style GPU_COMPUTE fill:#5b21b6,stroke:#4c1d95,color:#ddd6fe
    style GCS_STORAGE fill:#1e40af,stroke:#1e3a8a,color:#bfdbfe
    style GCP_COMPUTE fill:#1d4ed8,stroke:#1e3a8a,color:#bfdbfe
    style GCP_DB fill:#0e7490,stroke:#155e75,color:#cffafe
    style EXTERNAL_API fill:#dc2626,stroke:#b91c1c,color:#fef2f2
```

**Bảng Phân bổ Chi phí:**

| Thành phần | Nền tảng | Loại | Chi phí ước tính | Ghi chú |
|:----------:|:--------:|:----:|:-----------------:|---------|
| Training (GPU L4) | Lightning.ai | GPU | Included | Tách riêng khỏi GCP budget |
| Batch Embedding | Lightning.ai | GPU | Included | Chạy 1 lần, ~14GB output |
| Cloud Storage (GCS) | GCP | Storage | ~$5/tháng | Lakehouse + Embeddings + Registry |
| Cloud Datastore | GCP | Database | Free tier | Feast Online Store |
| Qdrant | GCP (VM/Container) | CPU | ~$15/tháng | CPU-only, nhỏ gọn |
| Cloud Run | GCP | Serverless | Pay-per-use | Scale-to-zero khi rảnh |
| Gemini API | Google AI | API | Pay-per-token | Giảm ~60% nhờ Semantic Cache |

**Chiến lược tiết kiệm chi phí:**

```
┌─────────────────────────────────────────────────────────────┐
│  💡 FINOPS BEST PRACTICES                                   │
│                                                             │
│  1. GPU ISOLATION: Mọi tác vụ GPU chạy trên Lightning.ai   │
│     → GCP credit chỉ dùng cho CPU/Storage/DB               │
│                                                             │
│  2. SEMANTIC CACHING: Cache kết quả LLM reranking          │
│     → Giảm ~60% lượng API call tới Gemini                  │
│                                                             │
│  3. SCALE-TO-ZERO: Cloud Run tự động tắt khi không có      │
│     request → $0 khi hệ thống rảnh                         │
│                                                             │
│  4. CPU-ONLY QDRANT: Không cần GPU cho Vector Search       │
│     → Tiết kiệm >$200/tháng so với GPU-based               │
│                                                             │
│  5. NO GPU QUOTA: Không nâng quota GPU trên GCP             │
│     → Ngăn chặn kích hoạt billing ngoài $300 credit        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc Mã nguồn

```
mining_massive/
│
├── 📄 plan.txt                    # Tài liệu kỹ thuật & tiến độ dự án
├── 📄 requirements.txt            # Dependencies (PyTorch, Lightning, PyArrow, ...)
├── 📄 README.md                   # Tài liệu bạn đang đọc
│
├── 🧊 DATA LAKEHOUSE (Notebooks)
│   ├── 📓 bronze.ipynb            # Bronze Layer: Raw ingestion từ Amazon Reviews
│   ├── 📓 silver_singler.ipynb    # Silver Layer: Cleansing & Validation
│   ├── 📓 silver_gold.ipynb       # Silver → Gold: Aggregation & Feature Engineering
│   ├── 📓 user_features.ipynb     # Gold Layer: Tính toán User Statistical Features
│   └── 📓 event_timestamp.ipynb   # Xử lý Event Timestamp cho Feast compatibility
│
├── 🏪 FEATURE STORE
│   └── 📄 features.py             # Feast Feature Definitions (Entity, FeatureView)
│
├── ⚡ EMBEDDING PIPELINE
│   └── 📄 embedding.py            # Item Tower: BGE embedding + GCS upload (chunked)
│
├── 🧠 ML TRAINING
│   ├── 📄 model.py                # UserTowerLightning: MLP architecture (PyTorch Lightning)
│   ├── 📄 dataset.py              # UserTowerDataset: PyTorch Dataset wrapper
│   ├── 📄 train.py                # Training script: DataLoader, Trainer, Checkpoint
│   ├── 📄 download_data.py        # GCS → Local: Multi-threaded data download
│   └── 📓 generate_training_data.ipynb  # Sinh dữ liệu huấn luyện (Positive + Negative)
│
└── 🔮 INFERENCE (Coming Soon)
    ├── 📄 predict.py              # [Planned] User embedding inference
    └── 📄 eval.py                 # [Planned] Offline evaluation (Hit Ratio, NDCG)
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu Hệ thống

- Python 3.10+
- CUDA-compatible GPU (khuyến nghị NVIDIA L4 hoặc tương đương)
- Google Cloud SDK (đã xác thực)
- ~20GB RAM cho quá trình embedding

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
pip install sentence-transformers feast google-cloud-storage gcsfs
```

### 2. Tải Dữ liệu từ GCS

```bash
# Tải Item Embeddings (~14GB) và Training Data từ GCS về local
python download_data.py
```

### 3. Chạy Embedding Pipeline (nếu cần tái tạo)

```bash
# Chạy trên GPU - sinh 384-dim vectors cho toàn bộ items
python embedding.py
```

### 4. Huấn luyện User Tower

```bash
# Huấn luyện MLP trên GPU L4 với Mixed Precision
python train.py
# Checkpoint tốt nhất được lưu tại: model_checkpoints/
```

### 5. Feast Feature Store

```bash
# Áp dụng Feature Definitions
feast apply

# Materialize features từ Offline → Online Store
feast materialize <START_DATE> <END_DATE>
```

---

## 📋 Trạng thái Dự án

| Hạng mục | Trạng thái | Ghi chú |
|----------|:----------:|---------|
| Data Lakehouse (Medallion) | ✅ Hoàn thành | Bronze → Silver → Gold trên Apache Iceberg |
| Item Tower Embedding | ✅ Hoàn thành | ~14GB vectors trên GCS |
| Feast Feature Store | ✅ Hoàn thành | Online Store (Datastore) hoạt động ổn định |
| User Tower Training | 🔄 Đang chạy | MLP training trên GPU L4 |
| Inference Pipeline | 📋 Lên kế hoạch | predict.py + Qdrant integration |
| Offline Evaluation | 📋 Lên kế hoạch | Hit Ratio, NDCG metrics |
| LLM Reranking | 📋 Lên kế hoạch | Gemini API + Semantic Cache |
| Serving API | 📋 Lên kế hoạch | FastAPI trên Cloud Run |

---

<div align="center">

### 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?style=flat-square&logo=lightning&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat-square&logo=googlecloud&logoColor=white)

**Phạm Minh Quang** · 2A202600263 · Mining Massive Datasets

</div>
