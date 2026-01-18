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
        
        try:
            img = Image.open(result['image_path'])
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(
                f"Rank {idx+1}\nScore: {result['score']:.2f}", fontsize=10
            )
        except Exception:
            axes[idx].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results_{query[:30].replace(' ', '_')}.png", dpi=150)
    plt.show()

def main():
    print("=" * 80)
    print("Fashion Retrieval System")
    print("=" * 80)

    parser = QueryParser()
    search_engine = FashionSearchEngine()

    test_queries = [
        "A person wearing black jacket",
        "Professional business attire inside a modern office",
        "Someone walking on ramp walk",
        "Casual weekend outfit for a city walk",
    ]

    while True:
        print("\nChoose an option:")
        print("1. Write your own query")
        print("2. Use test queries")
        print("3. Exit")

        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            query = input("\nEnter your fashion query: ").strip()
            if not query:
                print("Query cannot be empty!")
                continue

            parsed = parser.parse_query(query)
            print(f"\nParsed attributes: {parsed}")

            results = search_engine.search(parsed, top_k=10)
            display_results(results, query)

        elif choice == "2":
            for query in test_queries:
                print(f"\nProcessing query: {query}")

                parsed = parser.parse_query(query)
                print(f"Parsed attributes: {parsed}")

                results = search_engine.search(parsed, top_k=10)
                display_results(results, query)

                input("\nPress Enter for next test query...")

        elif choice == "3":
            print("\nExiting Fashion Retrieval System. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
