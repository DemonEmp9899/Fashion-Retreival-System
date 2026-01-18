# Fashion Retrieval System (Multi-Attribute, Zero-Shot)

This project implements a **multi-attribute, zero-shot fashion image retrieval system** that retrieves relevant fashion images from natural language queries such as:

- *“A person wearing a blue shirt sitting in a park”*
- *“Professional business attire inside an office”*
- *“Someone walking on a ramp walk”*

The system is designed to handle **compositional fashion queries**, scale to large datasets, and generalize to unseen descriptions without supervised training.

---

## 🧠 How This Project Works

The system follows a **two-stage design**:  
1. **Offline Indexing**  
2. **Online Retrieval**

Instead of representing images with a single embedding, fashion understanding is decomposed into **independent attributes**, making retrieval more precise and interpretable.

---

### 🔹 1. Indexing Pipeline (Offline)

The indexing step processes all images **once** and stores their features in a vector database.

For each image, the indexer extracts:

- **Semantic Embedding**  
  Global visual understanding using CLIP (style, posture, overall context)

- **Dominant Colors**  
  Extracted using K-means clustering on image pixels

- **Garment Types**  
  Detected using zero-shot CLIP prompts (e.g., jacket, shirt, dress)

- **Scene / Environment**  
  Classified using zero-shot CLIP (office, park, street, indoor)

Each attribute is stored in a **separate ChromaDB collection**, along with metadata such as image path and detected attributes.

📁 Implemented in: `indexer/`

---

### 🔹 2. Retrieval Pipeline (Online)

At query time, the system processes a natural language query and retrieves the most relevant images.

**Example Query:**  
> *“Someone wearing a blue shirt sitting in a park”*

The retriever performs the following steps:

1. **Query Parsing**  
   Extracts attributes such as:
   - Colors → `blue`
   - Garments → `shirt`
   - Scene → `park`
   - Semantic intent → full query meaning

2. **Independent Attribute Search**
   - Semantic search over CLIP embeddings
   - Color search over color embeddings
   - Garment search over garment embeddings
   - Scene search (if present)

3. **Score Fusion & Re-Ranking**
   - Similarity scores are normalized
   - Attribute-specific scores are combined using weighted fusion
   - Images matching more attributes are ranked higher

4. **Graceful Fallback**
   - If no attributes are detected, the system falls back to pure semantic retrieval

📁 Implemented in: `retriever/`

---

### 🔹 Why This Design Works Well for Fashion

Fashion queries are inherently **multi-attribute**.  
A single embedding often fails to distinguish compositions like:

- *“Blue shirt + black pants”* vs *“Black shirt + blue pants”*

By separating attributes, the system:
- Avoids common CLIP failure cases
- Improves fine-grained precision
- Remains interpretable and debuggable
- Supports zero-shot generalization

---

### 🔹 Zero-Shot & Scalability Properties

- No supervised labels required
- Handles unseen garments, styles, and descriptions
- Modular design (indexer ≠ retriever)
- Scales to **1M+ images** using HNSW indexing and distributed vector databases

---

# Complete Setup Guide – Fashion Retrieval System

This guide walks through setting up and running the system end-to-end.

---

## 📋 Prerequisites

- Python **3.8+**
- ~3200 `.jpg` fashion images
- OpenRouter account (free tier available)
- **GPU recommended** (CPU supported but slower)

---

## 🚀Complete Setup Guide – Fashion Retrieval System

This guide walks through setting up and running the **Multi-Attribute, Zero-Shot Fashion Retrieval System** end-to-end.

---

## 📋 Prerequisites

- Python **3.8+**
- ~3200 `.jpg` fashion images
- OpenRouter account (free tier available)
- **GPU recommended** (CPU supported but slower)

---

## 🚀 Step-by-Step Setup

### Step 1: Get OpenRouter API Key

1. Visit https://openrouter.ai  
2. Sign up (free account)
3. Go to https://openrouter.ai/keys
4. Click **Create Key**
5. Copy the key (`sk-or-v1-...`)
6. (Optional) Add ~$5 credits for smoother testing

**Cost estimate:** ~$0.003/query → ~$5 ≈ 1500+ queries

---

### Step 2: Clone the Repository

```bash
git clone https://github.com/your-username/fashion-retrieval-system.git
cd fashion-retrieval-system
```
Project Structure:
```tree
fashion-retrieval/
├── data/
│   ├── raw/              # Your 3200 .jpg images here
│   └── processed/        # Extracted features (features.pkl)
├── chroma_db/            # Vector database (auto-generated)
├── indexer/
│   ├── __init__.py
│   ├── feature_extractor.py
│   ├── vector_store.py
│   └── index_builder.py
├── retriever/
│   ├── __init__.py
│   ├── query_parser.py
│   ├── search_engine.py
│   └── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
Create .env File
```bash
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
```

Create .gitignore
```bash
.env
__pycache__/
*.pyc
data/raw/*
data/processed/*
chroma_db/
fashion_env/
.DS_Store
.ipynb_checkpoints
```

Create requirements.txt
```bash
torch>=2.0.0
torchvision>=0.15.0
git+https://github.com/openai/CLIP.git
chromadb>=0.4.0
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
tqdm>=4.66.0
requests>=2.31.0
matplotlib>=3.7.0
python-dotenv>=1.0.0
```

Install Dependencies
```bash
python -m venv fashion_env
fashion_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Add Images
Place all .jpg images into:
```bash
data/raw/
ls data/raw | wc -l

```

Code Files
```bash
Ensure the following files exist:
Indexer:
  • indexer/feature_extractor.py
  • indexer/vector_store.py
  • indexer/index_builder.py
Retriever:
  • retriever/query_parser.py
  • retriever/search_engine.py
  • retriever/main.py
Add init files:
  touch indexer/__init__.py
  touch retriever/__init__.py

```
Run Indexing
```bash
cd indexer
python index_builder.py

Expected:
  • Indexes all images
  • Builds ChromaDB collections
  • Takes ~45–60 mins on GPU / ~2–3 hrs on CPU
  • Creates ~250MB of indexed data
```

Run Retrieval
```bash
cd ../retriever
python main.py
```
Example output:
```bash
Processing query: A person wearing a black jacket
Parsed attributes: {'colors': ['black'], 'garments': ['jacket'], ...}

Rank 1:
Image: data/raw/image_0123.jpg
Score: 0.82
```

✅ Final Checklist
  • API key configured
  • Images added
  • Indexing completed
  • Retrieval returns results
  • .env ignored in GitHub
  • Example queries tested


