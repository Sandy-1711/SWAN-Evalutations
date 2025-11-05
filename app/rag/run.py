import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ======================
# Config
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent  # app/
DATA_DIR = BASE_DIR / "data"
EMBED_DIR = BASE_DIR / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# Files
EMBED_FILE = EMBED_DIR / "node_embeddings.npy"
META_FILE = EMBED_DIR / "node_metadata.json"
DATA_FILE = DATA_DIR / "dataset.json"

# Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


# ======================
# Internal: I/O helpers
# ======================
def _ensure_files_exist():
    """Check required files exist before reading."""
    for path in [EMBED_FILE, META_FILE, DATA_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")


def _load_data():
    """Load all core assets: embeddings, metadata, and dataset chunks."""
    _ensure_files_exist()
    embeddings = np.load(EMBED_FILE).astype(np.float32)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, metadata, chunks


def _save_embeddings(embeds: np.ndarray):
    """Persist numpy embeddings to disk."""
    np.save(EMBED_FILE, embeds.astype(np.float32))


def _save_json(path: Path, obj):
    """Write a JSON file safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ======================
# Internal: Build Index
# ======================
def _build_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# ======================
# Internal: Chunk → Nodes
# ======================
def _chunk_to_nodes(chunk: Dict, chunk_idx: int) -> List[Tuple[str, str, str, str]]:
    """Convert a single chunk to 5 node representations."""
    chunk_id = f"chunk_{chunk_idx}"
    nodes: List[Tuple[str, str, str, str]] = [
        (f"{chunk_id}_prompt", "Prompt", chunk_id, chunk.get("prompt", "")),
        (f"{chunk_id}_code", "Code", chunk_id, chunk.get("code", "")),
    ]

    # Parts
    parts = chunk.get("output", {}).get("parts", [])
    parts_text = ", ".join(p.get("type", "") for p in parts)
    nodes.append((f"{chunk_id}_parts", "Parts", chunk_id, parts_text))

    # Connections
    conns = chunk.get("output", {}).get("connections", [])
    conn_text = "\n".join(
        " → ".join(map(str, c[:2])) for c in conns if isinstance(c, list) and len(c) >= 2
    )
    nodes.append((f"{chunk_id}_output", "Output", chunk_id, conn_text))

    # Circuit space
    circuit = chunk.get("circuit_space_representation", "")
    nodes.append((f"{chunk_id}_circuit", "Circuit_Space", chunk_id, circuit))
    return nodes


# ======================
# Runtime Initialization
# ======================
embeddings, metadata, chunks = _load_data()
index = _build_index(embeddings)
model = SentenceTransformer(EMBEDDING_MODEL)


# ======================
# Public: Query Function
# ======================
def query(text: str, top_k: int = TOP_K, distance_threshold: float = 0.4) -> List[Dict]:
    """Search most relevant chunks using FAISS + cosine proximity."""
    query_vector = model.encode([text])[0].astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = []
    seen_chunk_ids = set()

    for i, idx in enumerate(indices[0]):
        distance = float(distances[0][i])
        if distance > distance_threshold:
            continue

        node_meta = metadata[idx]
        chunk_id = node_meta["chunk_id"]
        node_type = node_meta["type"]
        node_id = node_meta["node_id"]

        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        try:
            chunk_idx = int(chunk_id.split("_")[-1])
        except ValueError:
            continue
        if not (0 <= chunk_idx < len(chunks)):
            continue

        ch = chunks[chunk_idx]
        results.append(
            {
                "matched_node": node_type,
                "node_id": node_id,
                "chunk_id": chunk_id,
                "score": distance,
                "prompt": ch.get("prompt", ""),
                "code": ch.get("code", ""),
                "output": ch.get("output", {}),
                "circuit_space": ch.get("circuit_space_representation", ""),
            }
        )

    return results


# ======================
# Public: Ingest New Chunk
# ======================
def ingest_feedback_chunk(chunk: Dict, chunk_idx: Optional[int] = None) -> Dict:
    """Append or overwrite a single feedback chunk and update embeddings/index."""
    global embeddings, metadata, chunks, index, model

    if chunk_idx is None:
        chunk_idx = len(chunks)
    chunk_id = f"chunk_{chunk_idx}"

    # Build nodes & embeddings
    nodes = _chunk_to_nodes(chunk, chunk_idx)
    texts = [n[3] for n in nodes]
    new_embeds = model.encode(texts, show_progress_bar=False).astype(np.float32)

    # Update in-memory
    embeddings = np.vstack([embeddings, new_embeds])
    for node_id, node_type, ch_id, _ in nodes:
        metadata.append({"node_id": node_id, "type": node_type, "chunk_id": ch_id})

    if chunk_idx == len(chunks):
        chunks.append(chunk)
    elif 0 <= chunk_idx < len(chunks):
        chunks[chunk_idx] = chunk
    else:
        while len(chunks) < chunk_idx:
            chunks.append({})
        chunks.append(chunk)

    # Update FAISS
    index.add(new_embeds)

    # Persist all data
    _save_embeddings(embeddings)
    _save_json(META_FILE, metadata)
    _save_json(DATA_FILE, chunks)

    return {
        "status": "ok",
        "chunk_id": chunk_id,
        "chunk_index": chunk_idx,
        "new_nodes_added": len(nodes),
        "total_nodes": int(embeddings.shape[0]),
        "total_chunks": len(chunks),
    }


# ======================
# Utility: Context Filter
# ======================
def filter_rag_context(chunks_list: List[Dict], fields: List[str]):
    """Return only selected fields from chunks."""
    return [{key: ch[key] for key in fields if key in ch} for ch in chunks_list]
