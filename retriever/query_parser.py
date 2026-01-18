import json
import requests
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class QueryParser:
    """
    LLM-based query parser to extract attributes from natural language.
    Uses OpenRouter API for attribute extraction.
    Reads API key from .env file.
    """
    
    def __init__(self, api_key: str = None):
        # If no API key provided, try to load from environment
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            
        if not api_key:
            print("WARNING: No OpenRouter API key found. Using fallback parser.")
            print("To use LLM parsing, add OPENROUTER_API_KEY to .env file")
            self.api_key = None
        else:
            self.api_key = api_key
            print("✓ OpenRouter API key loaded successfully")
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def parse_query(self, query: str) -> Dict:
        """
        Parse natural language query into structured attributes.
        Returns: {colors: [], garments: [], scene: str, style: str, semantic: str}
        """
        
        # If no API key, use fallback
        if not self.api_key:
            return self._simple_parse(query)
        
        prompt = f"""
Extract fashion attributes from the query below.
Return ONLY valid JSON. No explanation.

Schema:
{{
  "colors": list of color strings,
  "garments": list of clothing item strings,
  "scene": string or null,
  "style": string or null,
  "semantic": string
}}

Query: "{query}"

Return ONLY valid JSON, no other text."""

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return self._simple_parse(query)
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Clean up response (remove markdown code blocks if present)
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            
            # Ensure all required fields exist
            return {
                "colors": parsed.get("colors", []),
                "garments": parsed.get("garments", []),
                "scene": parsed.get("scene", None),
                "style": parsed.get("style", None),
                "semantic": parsed.get("semantic", query)
            }
            
        except Exception as e:
            print(f"Error parsing query with LLM: {e}")
            print("Falling back to simple parser")
            return self._simple_parse(query)
    
    def _simple_parse(self, query: str) -> Dict:
        """Fallback parser using keyword matching (no API needed)"""
        query_lower = query.lower()
        
        colors = []
        color_keywords = ["red", "blue", "green", "yellow", "orange", "purple", 
                         "pink", "black", "white", "gray", "grey", "brown", "teal", 
                         "gold", "silver", "navy", "maroon"]
        for color in color_keywords:
            if color in query_lower:
                colors.append(color)
        
        garments = []
        garment_keywords = ["shirt", "pants", "dress", "jacket", "coat", "blazer",
                           "hoodie", "sweater", "jeans", "skirt", "tie", "raincoat",
                           "t-shirt", "tshirt", "polo", "cardigan", "vest"]
        for garment in garment_keywords:
            if garment in query_lower:
                garments.append(garment)
        
        scene = None
        scene_keywords = {
            "office": ["office", "corporate", "workplace", "business"],
            "park": ["park", "outdoor", "garden", "bench"],
            "street": ["street", "urban", "city", "sidewalk"],
            "home": ["home", "indoor", "living", "house"]
        }
        for scene_name, keywords in scene_keywords.items():
            if any(kw in query_lower for kw in keywords):
                scene = scene_name
                break
        
        style = None
        if "casual" in query_lower or "weekend" in query_lower:
            style = "casual"
        elif "professional" in query_lower or "business" in query_lower or "formal" in query_lower:
            style = "professional"
        
        return {
            "colors": colors,
            "garments": garments,
            "scene": scene,
            "style": style,
            "semantic": query
        }