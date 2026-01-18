import chromadb
from chromadb.config import Settings
import clip
import torch
from typing import List, Dict
import numpy as np
from collections import defaultdict

class FashionSearchEngine:
    """
    Multi-attribute search engine with weighted fusion.
    Combines results from semantic, color, garment, and scene collections.
    """

    def __init__(self, chroma_db_path="../indexer/chroma_db",
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        # Load CLIP for query encoding
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        # Load ChromaDB collections
        self.client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self.semantic_collection = self.client.get_collection("fashion_semantic")
        self.color_collection = self.client.get_collection("fashion_colors")
        self.garment_collection = self.client.get_collection("fashion_garments")
        self.scene_collection = self.client.get_collection("fashion_scenes")

    def encode_text_query(self, text: str) -> np.ndarray:
        """Encode text query using CLIP"""
        text_tokens = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().flatten()

    def search(self, parsed_query: Dict, top_k=10) -> List[Dict]:
        """
        Multi-attribute search with weighted fusion.
        """

        all_results = defaultdict(lambda: {
            "score": 0.0,
            "metadata": None,
            "matches": 0
        })

        # Base weights
        weights = {
            "semantic": 0.4,
            "color": 0.25,
            "garment": 0.25,
            "scene": 0.1
        }

        # Dynamically adjust weights
        if parsed_query.get("colors"):
            weights["color"] = 0.35
            weights["semantic"] = 0.3

        if parsed_query.get("scene"):
            weights["scene"] = 0.2
            weights["semantic"] = 0.3

        # 1. Semantic search (always)
        semantic_query = parsed_query["semantic"]
        if parsed_query.get("style"):
            semantic_query = f"{parsed_query['style']} style: {semantic_query}"

        query_embedding = self.encode_text_query(semantic_query)

        semantic_results = self.semantic_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k * 3, 100)
        )

        for doc_id, distance, metadata in zip(
            semantic_results["ids"][0],
            semantic_results["distances"][0],
            semantic_results["metadatas"][0]
        ):
            score = 1.0 / (1.0 + distance)
            all_results[doc_id]["score"] += weights["semantic"] * score

            # Preserve metadata safely
            if all_results[doc_id]["metadata"] is None:
                all_results[doc_id]["metadata"] = metadata

        # 2. Color search
        if parsed_query.get("colors"):
            color_query = " ".join(parsed_query["colors"])
            color_results = self.color_collection.query(
                query_texts=[color_query],
                n_results=min(top_k * 3, 100)
            )

            for doc_id, distance in zip(
                color_results["ids"][0],
                color_results["distances"][0]
            ):
                score = 1.0 / (1.0 + distance)
                all_results[doc_id]["score"] += weights["color"] * score
                all_results[doc_id]["matches"] += 1

        # 3. Garment search
        if parsed_query.get("garments"):
            garment_query = " ".join(parsed_query["garments"])
            garment_results = self.garment_collection.query(
                query_texts=[garment_query],
                n_results=min(top_k * 3, 100)
            )

            for doc_id, distance in zip(
                garment_results["ids"][0],
                garment_results["distances"][0]
            ):
                score = 1.0 / (1.0 + distance)
                all_results[doc_id]["score"] += weights["garment"] * score
                all_results[doc_id]["matches"] += 1

        # 4. Scene search
        if parsed_query.get("scene"):
            scene_results = self.scene_collection.query(
                query_texts=[parsed_query["scene"]],
                n_results=min(top_k * 3, 100)
            )

            for doc_id, distance in zip(
                scene_results["ids"][0],
                scene_results["distances"][0]
            ):
                score = 1.0 / (1.0 + distance)
                all_results[doc_id]["score"] += weights["scene"] * score
                all_results[doc_id]["matches"] += 1

        # Sort by (attribute matches, score)
        ranked_results = sorted(
            all_results.items(),
            key=lambda x: (x[1]["matches"], x[1]["score"]),
            reverse=True
        )

        # Final formatting (SAFE)
        final_results = []
        for doc_id, data in ranked_results:
            if data["metadata"] is None:
                continue

            image_path = data["metadata"].get("image_path")
            if not image_path:
                continue

            final_results.append({
                "id": doc_id,
                "image_path": image_path,
                "score": round(data["score"], 4),
                "attribute_matches": data["matches"],
                "metadata": data["metadata"]
            })

            if len(final_results) >= top_k:
                break

        return final_results
