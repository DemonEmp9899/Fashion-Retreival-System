import torch
import clip
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import cv2
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FashionFeatureExtractor:
    """
    Multi-attribute feature extractor for fashion images.
    Extracts: CLIP embeddings, colors, garment types, scene context
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading CLIP model on {device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Fashion-specific templates for zero-shot classification
        self.garment_templates = [
            "a photo of a person wearing a {}",
            "a photo of someone in a {}",
        ]
        
        self.garment_types = [
            "blazer", "suit jacket", "dress shirt", "button-down shirt",
            "hoodie", "t-shirt", "sweater", "cardigan",
            "raincoat", "jacket", "coat", "windbreaker",
            "jeans", "dress pants", "shorts", "skirt",
            "dress", "jumpsuit", "tank top", "polo shirt"
        ]
        
        self.scene_types = [
            "office interior", "corporate office", 
            "urban street", "city street",
            "park", "outdoor garden",
            "home interior", "indoor setting"
        ]
        
        # Color reference for name mapping
        self.color_names = {
            (255, 0, 0): "red", (0, 0, 255): "blue", (0, 255, 0): "green",
            (255, 255, 0): "yellow", (255, 165, 0): "orange", (128, 0, 128): "purple",
            (255, 192, 203): "pink", (0, 0, 0): "black", (255, 255, 255): "white",
            (128, 128, 128): "gray", (165, 42, 42): "brown", (0, 128, 128): "teal",
            (255, 215, 0): "gold", (192, 192, 192): "silver", (139, 69, 19): "brown"
        }
    
    def extract_clip_embedding(self, image_path: str) -> np.ndarray:
        """Extract CLIP image embedding"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(512)
    
    def extract_dominant_colors(self, image_path: str, n_colors=5) -> List[Dict]:
        """Extract dominant colors using K-means clustering"""
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            
            pixels = img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            counts = np.bincount(kmeans.labels_)
            
            color_info = []
            for color, count in zip(colors, counts):
                color_name = self._get_color_name(tuple(color))
                percentage = (count / len(pixels)) * 100
                color_info.append({
                    "rgb": tuple(color),
                    "name": color_name,
                    "percentage": percentage
                })
            
            color_info.sort(key=lambda x: x['percentage'], reverse=True)
            return color_info
        except Exception as e:
            print(f"Error extracting colors from {image_path}: {e}")
            return []
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Map RGB to nearest color name"""
        min_dist = float('inf')
        closest_name = "unknown"
        
        for ref_rgb, name in self.color_names.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        
        return closest_name
    
    def classify_garments(self, image_path: str, top_k=3) -> List[Tuple[str, float]]:
        """Classify garment types using CLIP zero-shot"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            text_inputs = []
            for garment in self.garment_types:
                for template in self.garment_templates:
                    text_inputs.append(template.format(garment))
            
            text_tokens = clip.tokenize(text_inputs).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_tokens)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).squeeze(0)
            
            scores = similarity.view(len(self.garment_types), len(self.garment_templates)).mean(dim=1)
            probs = scores.softmax(dim=0).cpu().numpy()
            
            top_indices = np.argsort(probs)[-top_k:][::-1]
            results = [(self.garment_types[i], float(probs[i])) for i in top_indices]
            
            return results
        except Exception as e:
            print(f"Error classifying garments in {image_path}: {e}")
            return []
    
    def classify_scene(self, image_path: str) -> Tuple[str, float]:
        """Classify scene/environment using CLIP"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            text_inputs = [f"a photo in {scene}" for scene in self.scene_types]
            text_tokens = clip.tokenize(text_inputs).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_tokens)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).squeeze(0)
                probs = similarity.softmax(dim=0).cpu().numpy()
            
            top_idx = np.argmax(probs)
            return self.scene_types[top_idx], float(probs[top_idx])
        except Exception as e:
            print(f"Error classifying scene in {image_path}: {e}")
            return "unknown", 0.0
    
    def extract_all_features(self, image_path: str) -> Dict:
        """Extract all features for a single image"""
        features = {
            "image_path": image_path,
            "clip_embedding": self.extract_clip_embedding(image_path),
            "colors": self.extract_dominant_colors(image_path),
            "garments": self.classify_garments(image_path),
            "scene": self.classify_scene(image_path)
        }
        return features