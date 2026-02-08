import requests
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib

DBPEDIA_SPOTLIGHT = "https://api.dbpedia-spotlight.org/en/annotate"
DBPEDIA_LOOKUP = "https://lookup.dbpedia.org/api/search"
HEADERS = {'Accept': 'application/json'}

class DBpediaKnowledge:
    """
    Optimized version with:
      1. Faster DBpedia Lookup API instead of SPARQL
      2. Aggressive caching
      3. Reduced data fetching
      4. Better timeout handling
    """

    def __init__(self, confidence: float = 0.5, support: int = 20, max_workers: int = 10):
        self.confidence = confidence
        self.support = support
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        
        # Cache for entities and lookup results
        self._entity_cache = {}
        self._lookup_cache = {}

    def _text_hash(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.md5(text.encode()).hexdigest()

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract DBpedia entities from text using Spotlight.
        Uses caching to avoid repeated calls for same text.
        """
        cache_key = self._text_hash(text)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        params = {
            'text': text,
            'confidence': self.confidence,
            'support': self.support
        }

        try:
            response = self.session.get(
                DBPEDIA_SPOTLIGHT, 
                params=params, 
                timeout=3  # Reduced timeout
            )
            if response.status_code != 200:
                return []

            data = response.json()
            resources = data.get("Resources", [])
            entities = []

            for r in resources:
                entities.append({
                    "surface": r["@surfaceForm"],
                    "uri": r["@URI"],
                    "types": r.get("@types", "")
                })

            self._entity_cache[cache_key] = entities
            return entities

        except Exception:
            return []

    def lookup_entity(self, uri: str) -> Dict:
        """
        Use DBpedia Lookup API (much faster than SPARQL).
        Gets basic info: abstract, categories, redirects.
        """
        if uri in self._lookup_cache:
            return self._lookup_cache[uri]

        # Extract resource name from URI
        resource_name = uri.split("/")[-1]
        
        params = {
            'query': resource_name,
            'maxResults': 1,
            'format': 'json'
        }

        try:
            response = self.session.get(
                DBPEDIA_LOOKUP,
                params=params,
                timeout=2  # Lookup is fast
            )
            
            if response.status_code != 200:
                return {"uri": uri, "info": {}}

            data = response.json()
            results = data.get("results", [])
            
            if results:
                result = results[0]
                info = {
                    "label": result.get("label", [None])[0] if result.get("label") else None,
                    "description": result.get("description", [None])[0] if result.get("description") else None,
                    "categories": result.get("categories", [])[:5],  # Limit to 5
                    "redirects": result.get("redirects", [])[:3]     # Limit to 3
                }
            else:
                info = {}

            result = {"uri": uri, "info": info}
            self._lookup_cache[uri] = result
            return result

        except Exception:
            return {"uri": uri, "info": {}}

    def query_dbpedia_minimal(self, uri: str) -> Dict:
        """
        Minimal SPARQL query - only fetch most relevant properties.
        Use this only if you need specific semantic properties.
        """
        if uri in self._lookup_cache:
            return self._lookup_cache[uri]

        sparql_endpoint = "https://dbpedia.org/sparql"

        # Only query for specific useful properties
        query = f"""
        SELECT ?property ?value
        WHERE {{
            <{uri}> ?property ?value .
            FILTER(
                ?property = dbo:abstract ||
                ?property = rdfs:label ||
                ?property = dbo:type ||
                ?property = rdf:type
            )
            FILTER(lang(?value) = 'en' || lang(?value) = '')
        }}
        LIMIT 10
        """

        try:
            response = self.session.get(
                sparql_endpoint,
                params={"query": query, "format": "json"},
                timeout=3  # Reduced timeout
            )
            
            if response.status_code != 200:
                return {"uri": uri, "triples": []}

            results = response.json().get("results", {}).get("bindings", [])
            triples = []

            for row in results:
                triples.append({
                    "property": row["property"]["value"],
                    "value": row["value"]["value"]
                })

            result = {"uri": uri, "triples": triples}
            self._lookup_cache[uri] = result
            return result

        except Exception:
            return {"uri": uri, "triples": []}

    def get_external_knowledge(self, text: str, use_lookup: bool = True) -> Dict:
        """
        High-level function to extract entities and enrich them.
        
        Args:
            text: Input text to analyze
            use_lookup: If True, use fast Lookup API. If False, use SPARQL (slower)
        """
        entities = self.extract_entities(text)
        
        if not entities:
            return {
                "entities": [],
                "expanded_knowledge": []
            }

        expanded = []
        fetch_method = self.lookup_entity if use_lookup else self.query_dbpedia_minimal

        # Parallel fetching
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_entity = {
                executor.submit(fetch_method, e["uri"]): e 
                for e in entities
            }
            
            for future in as_completed(future_to_entity):
                entity = future_to_entity[future]
                try:
                    result = future.result()
                    
                    if use_lookup:
                        expanded.append({
                            "surface": entity["surface"],
                            "types": entity["types"],
                            "info": result.get("info", {})
                        })
                    else:
                        expanded.append({
                            "surface": entity["surface"],
                            "types": entity["types"],
                            "triples": result.get("triples", [])
                        })
                except Exception:
                    # Fail gracefully
                    expanded.append({
                        "surface": entity["surface"],
                        "types": entity["types"],
                        "info": {} if use_lookup else {"triples": []}
                    })

        return {
            "entities": entities,
            "expanded_knowledge": expanded
        }

    def clear_cache(self):
        """Clear all cached data"""
        self._entity_cache.clear()
        self._lookup_cache.clear()

    def __del__(self):
        """Clean up session on deletion"""
        self.session.close()


# Usage example
if __name__ == "__main__":
    kb = DBpediaKnowledge(confidence=0.5, support=20, max_workers=10)
    
    text = "SpaceX launched a Falcon 9 rocket from Cape Canaveral."
    
    # Fast version (recommended)
    result = kb.get_external_knowledge(text, use_lookup=True)
    print("Entities found:", len(result["entities"]))
    
    # If you need semantic triples (slower)
    # result = kb.get_external_knowledge(text, use_lookup=False)