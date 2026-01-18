import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict
import json

class FashionVectorStore:
    """
    Multi-collection vector store for fashion retrieval.
    Separate collections for semantic, color, garment, and scene attributes.
    """
    
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create/get collections
        self.semantic_collection = self.client.get_or_create_collection(
            name="fashion_semantic",
            metadata={"description": "CLIP semantic embeddings"}
        )
        
        self.color_collection = self.client.get_or_create_collection(
            name="fashion_colors",
            metadata={"description": "Color-based embeddings"}
        )
        
        self.garment_collection = self.client.get_or_create_collection(
            name="fashion_garments",
            metadata={"description": "Garment type embeddings"}
        )
        
        self.scene_collection = self.client.get_or_create_collection(
            name="fashion_scenes",
            metadata={"description": "Scene/environment embeddings"}
        )
    
    def add_image(self, image_id: str, features: Dict):
        """Add image features to all collections"""
        
        # Add to semantic collection
        self.semantic_collection.add(
            ids=[image_id],
            embeddings=[features["clip_embedding"].tolist()],
            metadatas=[{
                "image_path": features["image_path"],
                "colors": json.dumps([c["name"] for c in features["colors"]]),
                "garments": json.dumps([g[0] for g in features["garments"]]),
                "scene": features["scene"][0]
            }]
        )
        
        # Add to color collection (using color names as text)
        color_text = " ".join([c["name"] for c in features["colors"][:3]])
        self.color_collection.add(
            ids=[image_id],
            documents=[color_text],
            metadatas=[{"image_path": features["image_path"]}]
        )
        
        # Add to garment collection
        garment_text = " ".join([g[0] for g in features["garments"]])
        self.garment_collection.add(
            ids=[image_id],
            documents=[garment_text],
            metadatas=[{"image_path": features["image_path"]}]
        )
        
        # Add to scene collection
        self.scene_collection.add(
            ids=[image_id],
            documents=[features["scene"][0]],
            metadatas=[{"image_path": features["image_path"]}]
        )
    
    def batch_add_images(self, features_list: List[Dict], batch_size=100):
        """Add multiple images in batches"""
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i+batch_size]
            
            ids = [f"img_{i+j}" for j in range(len(batch))]
            
            # Semantic collection
            self.semantic_collection.add(
                ids=ids,
                embeddings=[f["clip_embedding"].tolist() for f in batch],
                metadatas=[{
                    "image_path": f["image_path"],
                    "colors": json.dumps([c["name"] for c in f["colors"]]),
                    "garments": json.dumps([g[0] for g in f["garments"]]),
                    "scene": f["scene"][0]
                } for f in batch]
            )
            
            # Color collection
            self.color_collection.add(
                ids=ids,
                documents=[" ".join([c["name"] for c in f["colors"][:3]]) for f in batch],
                metadatas=[{"image_path": f["image_path"]} for f in batch]
            )
            
            # Garment collection
            self.garment_collection.add(
                ids=ids,
                documents=[" ".join([g[0] for g in f["garments"]]) for f in batch],
                metadatas=[{"image_path": f["image_path"]} for f in batch]
            )
            
            # Scene collection
            self.scene_collection.add(
                ids=ids,
                documents=[f["scene"][0] for f in batch],
                metadatas=[{"image_path": f["image_path"]} for f in batch]
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(features_list)-1)//batch_size + 1}")
