import os
from pathlib import Path
from tqdm import tqdm
import pickle
from feature_extractor import FashionFeatureExtractor
from vector_store import FashionVectorStore

def build_index(image_dir: str, output_dir="./data/processed", batch_size=100):
    """
    Main indexing script.
    Processes all images and builds vector database.
    """
    
    # Initialize
    extractor = FashionFeatureExtractor()
    vector_store = FashionVectorStore(persist_directory="./chroma_db")
    
    # Get all image paths
    image_paths = list(Path(image_dir).glob("*.jpg"))
    print(f"Found {len(image_paths)} images")
    
    # Extract features
    all_features = []
    failed_images = []
    
    for img_path in tqdm(image_paths, desc="Extracting features"):
        try:
            features = extractor.extract_all_features(str(img_path))
            all_features.append(features)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            failed_images.append(str(img_path))
    
    print(f"\nSuccessfully processed {len(all_features)} images")
    print(f"Failed: {len(failed_images)} images")
    
    # Save features to disk
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/features.pkl", "wb") as f:
        pickle.dump(all_features, f)
    
    print("\nBuilding vector database...")
    vector_store.batch_add_images(all_features, batch_size=batch_size)
    
    print("\nIndexing complete!")
    print(f"Total images indexed: {len(all_features)}")
    
    return vector_store

if __name__ == "__main__":
    # Update this path to your dataset directory
    IMAGE_DIR = "../data/raw"  # Change this to your image folder
    
    build_index(IMAGE_DIR)