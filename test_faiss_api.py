import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load dataset
csv_path = "Dataset.csv"
df = pd.read_csv(csv_path)

# Clean and normalize
if "Context" not in df.columns or "Response" not in df.columns:
    raise ValueError("Dataset.csv must have 'Context' and 'Response' columns!")
df = df.dropna(subset=["Context", "Response"])

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings
contexts = df["Context"].tolist()
embeddings = model.encode(contexts, show_progress_bar=True, convert_to_numpy=True)
faiss.normalize_L2(embeddings)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Test query
query = "I'm very depressed. How do I find someone to talk to?"
q_emb = model.encode([query], convert_to_numpy=True)
faiss.normalize_L2(q_emb)

# Search top 20
D, I = index.search(q_emb, k=20)

print("Top 20 results for query:", query)
seen = set()
for rank, idx in enumerate(I[0]):
    if idx < len(df):
        context = str(df.iloc[idx]["Context"])
        response = str(df.iloc[idx]["Response"])
        key = (context, response)
        duplicate = "DUPLICATE" if key in seen else ""
        print(f"{rank+1:2d}. [Score: {D[0][rank]:.3f}] {context[:60]} | {response[:60]} {duplicate}")
        seen.add(key) 