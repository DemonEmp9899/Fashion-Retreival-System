from query_parser import QueryParser
from search_engine import FashionSearchEngine
import os
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def display_results(results, query, max_display=5):
    """Display search results"""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(1, min(len(results), max_display), figsize=(15, 3))
    if len(results) == 1:
        axes = [axes]
    
    for idx, result in enumerate(results[:max_display]):
        print(f"Rank {idx+1}:")
        print(f"  Image: {result['image_path']}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Attribute Matches: {result['attribute_matches']}")
        print(f"  Metadata: {result['metadata']}")
        print()
        
        # Display image
        try:
            img = Image.open(result['image_path'])
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f"Rank {idx+1}\nScore: {result['score']:.2f}", 
                               fontsize=10)
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Image not found", ha='center', va='center')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results_{query[:30].replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("="*80)
    print("Fashion Retrieval System")
    print("="*80)
    
    # Initialize (automatically loads from .env)
    parser = QueryParser()  # Will auto-load OPENROUTER_API_KEY from .env
    search_engine = FashionSearchEngine()
    
    print("\nSystem initialized successfully!")
    print()
    
    # Test queries from assignment
    test_queries = [
        "A person wearing black jacket",
        "Professional business attire inside a modern office",
        "Someone walking on ramp walk",
        "Casual weekend outfit for a city walk",
    ]
    
    for query in test_queries:
        print(f"\nProcessing query: {query}")
        
        # Parse query
        parsed = parser.parse_query(query)
        print(f"Parsed attributes: {parsed}")
        
        # Search
        results = search_engine.search(parsed, top_k=10)
        
        # Display
        display_results(results, query, max_display=5)
        
        input("\nPress Enter for next query...")

if __name__ == "__main__":
    main()