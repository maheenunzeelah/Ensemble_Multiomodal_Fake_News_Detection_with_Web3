import json
import re
import pandas as pd
import ast
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import os
from dotenv import load_dotenv
import torch
from external_knowledge import DBpediaKnowledge
# from explanation_module import ExplanationModule
from visual_grounding.vg_score_calculator import get_vgs_calculator
import glob

class CaptionSimilarityAnalyzer:
    """
    Advanced Caption Similarity Analyzer with Dataset Support
    Combines lexical (ROUGE) and semantic (AI-powered) similarity analysis
    """
    
    def __init__(self, api_key: str = None, image_dir: str = "allData_images"):
        # Load SentenceTransformer model once and set device
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        from sentence_transformers import SentenceTransformer
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.dbpedia = DBpediaKnowledge()
        # self.explainer = ExplanationModule()
        self.image_dir = Path(image_dir)
        
        # Initialize VGS calculator (lazy loading - only when needed)
        self._vgs_calculator = None
        
        print(f"✅ Analyzer initialized with image directory: {self.image_dir}")

    @property
    def vgs_calculator(self):
        """Lazy load VGS calculator"""
        if self._vgs_calculator is None:
            print("Loading Grounding DINO model for VGS calculation...")
            self._vgs_calculator = get_vgs_calculator()
        return self._vgs_calculator

    def find_image_path(self, image_id: str) -> Optional[Path]:
        """
        Find image path with any extension (.jpg, .png, .jpeg, etc.)
        
        Args:
            image_id: Image ID (without extension)
        
        Returns:
            Path to image or None if not found
        """
        # Try common extensions
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for ext in extensions:
            image_path = self.image_dir / f"{image_id}{ext}"
            if image_path.exists():
                return image_path
        
        # If not found with exact match, try glob pattern
        pattern = str(self.image_dir / f"{image_id}.*")
        matches = glob.glob(pattern)
        if matches:
            return Path(matches[0])
        
        return None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t]
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Generate n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i+n]))
        return ngrams
    
    def calculate_rouge_n(self, reference: str, candidate: str, n: int) -> Dict[str, float]:
        """Calculate ROUGE-N score"""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        ref_ngrams = self.get_ngrams(ref_tokens, n)
        cand_ngrams = self.get_ngrams(cand_tokens, n)
        
        if len(ref_ngrams) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matches': []}
        
        ref_set = set(ref_ngrams)
        matches = [ng for ng in cand_ngrams if ng in ref_set]
        unique_matches = list(set(matches))
        
        recall = len(unique_matches) / len(ref_ngrams) if ref_ngrams else 0
        precision = len(unique_matches) / len(cand_ngrams) if cand_ngrams else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matches': unique_matches
        }
    
    def longest_common_subsequence(self, s1: List[str], s2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE-L score"""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if len(ref_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'lcs_length': 0}
        
        lcs_length = self.longest_common_subsequence(ref_tokens, cand_tokens)
        
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0
        precision = lcs_length / len(cand_tokens) if cand_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'lcs_length': lcs_length
        }
    
    def get_token_alignment(self, reference: str, candidate: str) -> List[Dict[str, Any]]:
        """Get token-level alignment"""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        ref_set = set(ref_tokens)
        
        return [{'token': token, 'matched': token in ref_set} for token in cand_tokens]
    
    def analyze_semantic_coverage(self, reference: str, candidate: str) -> Dict[str, List[str]]:
        """Analyze token coverage"""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        
        covered = [t for t in ref_tokens if t in cand_set]
        missing = [t for t in ref_tokens if t not in cand_set]
        extra = [t for t in cand_tokens if t not in ref_set]
        
        return {
            'covered': covered,
            'missing': missing,
            'extra': extra
        }
    
    def analyze_person_influence(self, person_names: List[str], caption: str) -> Dict[str, Any]:
        """Analyze if detected person names appear in caption"""
        if not person_names:
            return {
                'present': False,
                'influence': 'none',
                'name_tokens': [],
                'matched_tokens': [],
                'matched_persons': []
            }
        
        caption_lower = caption.lower()
        matched_persons = []
        all_matched_tokens = []
        
        for person in person_names:
            person_tokens = self.tokenize(person)
            matched_tokens = [t for t in person_tokens if t in caption_lower]
            if matched_tokens:
                matched_persons.append(person)
                all_matched_tokens.extend(matched_tokens)
        
        has_match = len(matched_persons) > 0
        
        return {
            'present': has_match,
            'influence': 'high' if has_match else 'low',
            'total_persons': len(person_names),
            'matched_persons': matched_persons,
            'matched_tokens': list(set(all_matched_tokens))
        }
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """Calculate semantic similarity using SentenceTransformer embeddings (optimized)"""
        from sentence_transformers import util
        embeddings = self.st_model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return {
            'semantic_similarity_score': similarity,
            'semantic_analysis': {
                'shared_concepts': [],
                'missing_concepts': [],
                'extra_concepts': [],
                'paraphrases': []
            },
            'contextual_alignment': similarity,
            'explanation': f"Cosine similarity between embeddings: {similarity:.3f}"
        }
    
    def parse_person_names(self, person_names: Any) -> List[str]:
        """
        Parse person_names from various formats (string, list, array-like)
        Clean up np.str_() wrapper syntax
        
        Args:
            person_names: Can be string representation of list, actual list, or string
        
        Returns:
            List of person names
        """
        if pd.isna(person_names) or person_names is None:
            return []
        
        # If already a list
        if isinstance(person_names, list):
            return person_names
        
        # If string representation of a list
        if isinstance(person_names, str):
            # Clean up np.str_() wrapper syntax
            cleaned = person_names.replace("np.str_(", "").replace(")", "").replace("'", "").replace('"', '')
            
            # Try to parse as Python literal first
            try:
                parsed = ast.literal_eval(person_names)
                if isinstance(parsed, list):
                    return [str(p).strip() for p in parsed]
            except (ValueError, SyntaxError):
                pass
            
            # If it's array-like with brackets
            if '[' in cleaned and ']' in cleaned:
                cleaned = cleaned.replace('[', '').replace(']', '')
                names = [n.strip() for n in cleaned.split(',')]
                return [n for n in names if n]
            
            # If comma-separated
            if ',' in cleaned:
                return [name.strip() for name in cleaned.split(',')]
            
            # Single name
            if cleaned.strip():
                return [cleaned.strip()]
    
        return []
    
    def compute_kg_similarity(self, kg1: Dict, kg2: Dict) -> Dict:
        """
        Compute simple knowledge-graph similarity:
        1. shared entities
        2. shared DBpedia classes (types)
        3. shared relations (properties)
        """
        
        ent1 = {e["uri"] for e in kg1["entities"]}
        ent2 = {e["uri"] for e in kg2["entities"]}
        shared_entities = list(ent1.intersection(ent2))

        types1 = set()
        for e in kg1["entities"]:
            types1.update(e["types"].split(','))

        types2 = set()
        for e in kg2["entities"]:
            types2.update(e["types"].split(','))

        shared_types = list(types1.intersection(types2))

        # Handle both lookup mode (info) and SPARQL mode (triples)
        props1 = set()
        for e in kg1["expanded_knowledge"]:
            # Check if using SPARQL mode (has triples)
            if "triples" in e:
                for t in e["triples"]:
                    props1.add(t["property"])
            # If using lookup mode (has info with categories)
            elif "info" in e and e["info"]:
                # Add categories as properties
                if "categories" in e["info"]:
                    props1.update(e["info"]["categories"])

        props2 = set()
        for e in kg2["expanded_knowledge"]:
            # Check if using SPARQL mode (has triples)
            if "triples" in e:
                for t in e["triples"]:
                    props2.add(t["property"])
            # If using lookup mode (has info with categories)
            elif "info" in e and e["info"]:
                # Add categories as properties
                if "categories" in e["info"]:
                    props2.update(e["info"]["categories"])

        shared_relations = list(props1.intersection(props2))

        score = 0.0
        if shared_entities:
            score += 0.5
        if shared_types:
            score += 0.3
        if shared_relations:
            score += 0.2

        return {
            "kg_similarity_score": score,
            "shared_entities": shared_entities,
            "shared_types": shared_types,
            "shared_relations": shared_relations
        }

    def analyze_sample(self, row: pd.Series, index: int = None) -> Dict[str, Any]:
        """
        Analyze a single sample from dataset
        
        Args:
            row: pandas Series with 'person_names', 'caption', 'clean_title', 'id'
            index: Optional row index for identification
        
        Returns:
            Complete analysis results
        """
        person_names = self.parse_person_names(row['person_names'])
        
        # Clean person names - remove np.str_() wrappers
        person_names = [
            name.replace("np.str_(", "").replace(")", "").replace("'", "").strip()
            for name in person_names
        ]
        
        caption = str(row['caption']) if pd.notna(row['caption']) else ""
        headline = str(row['clean_title']) if pd.notna(row['clean_title']) else ""
        
        # Get image ID - try different possible column names
        image_id = None
        for col in ['id', 'index', 'image_id', 'img_id', 'image_filename']:
            if col in row.index and pd.notna(row[col]):
                image_id = str(row[col])
                break
        
        if image_id is None:
            print(f"Warning: No image ID found at index {index}")
            image_id = str(index) if index is not None else "unknown"
        
        if not caption or not headline:
            print(f"Warning: Empty caption or headline at index {index}")
            return None
        
        print(f"\n{'='*70}")
        print(f"Analyzing sample {index if index is not None else ''}")
        print(f"{'='*70}")
        print(f"Image ID: {image_id}")
        print(f"Detected Persons: {person_names}")
        print(f"Caption: {caption[:100]}...")
        print(f"Headline: {headline[:100]}...")
        
        # Calculate lexical metrics
        rouge1 = self.calculate_rouge_n(headline, caption, 1)
        rouge2 = self.calculate_rouge_n(headline, caption, 2)
        rougeL = self.calculate_rouge_l(headline, caption)
        alignment = self.get_token_alignment(headline, caption)
        coverage = self.analyze_semantic_coverage(headline, caption)
        person_influence = self.analyze_person_influence(person_names, caption)
        semantic = self.calculate_semantic_similarity(headline, caption)
        
        # Calculate VGS using Grounding DINO
        vgs_score = 0.0
        image_path = self.find_image_path(image_id)
        
        if image_path:
            print(f"📷 Computing VGS for image: {image_path}")
            try:
                vgs_score = self.vgs_calculator.compute_vgs(
                    image_path=str(image_path),
                    headline=headline,
                    person_names=person_names
                )
                print(f" VGS Score: {vgs_score:.3f}")
            except Exception as e:
                print(f" Error computing VGS: {e}")
                vgs_score = 0.0
        else:
            print(f"Image not found for ID: {image_id}")
        
        # Knowledge graph similarity
        headline_kg = self.dbpedia.get_external_knowledge(headline)
        caption_kg = self.dbpedia.get_external_knowledge(caption)
        kg_similarity = self.compute_kg_similarity(headline_kg, caption_kg)
        
        return {
            'index': index,
            'image_id': image_id,
            'person_names': person_names,
            'caption': caption,
            'headline': headline,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'alignment': alignment,
            'coverage': coverage,
            'person_influence': person_influence,
            'semantic': semantic,
            'vgs': vgs_score,  # Just the score, not a dict
            'kg_similarity': kg_similarity,
            'headline_kg': headline_kg,
            'caption_kg': caption_kg,
            'label': row.get('label', None)  # Optional label if present
        }
    
    def analyze_dataset(self, df: pd.DataFrame, 
                       num_samples: Optional[int] = None,
                       random_samples: bool = True,
                       random_state: int = 42) -> List[Dict[str, Any]]:
        """
        Analyze multiple samples from dataset
        
        Args:
            df: DataFrame with columns 'person_names', 'caption', 'clean_title', 'id'
            num_samples: Number of samples to analyze (None = all)
            random_samples: If True, randomly sample; if False, take first N
            random_state: Random seed for reproducibility
        
        Returns:
            List of analysis results
        """
        # Validate columns
        required_cols = ['person_names', 'caption', 'clean_title','image_filename']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for ID column
        id_cols = ['id', 'index', 'image_id', 'img_id','image_filename']
        has_id = any(col in df.columns for col in id_cols)
        if not has_id:
            print(f" Warning: No ID column found. Expected one of: {id_cols}")
        
        # Sample data
        # if num_samples is not None and num_samples < len(df):
        #     if random_samples:
        #         sample_df = df.sample(n=num_samples, random_state=random_state)
        #     else:
        #         sample_df = df.head(num_samples)
        # else:
        sample_df = df
        
        print(f"\n🔍 Analyzing {len(sample_df)} samples from dataset...")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        results = []
        for idx, (original_idx, row) in enumerate(sample_df.iterrows(), 1):
            print(f"\n--- Processing sample {idx}/{len(sample_df)} (row {original_idx}) ---")
            result = self.analyze_sample(row, index=original_idx)
            
            if result:
                # explanation = self.explainer.explain(result)
                # print(f"\n💡 Explanation:\n{explanation}")
                results.append(result)
        
        return results
    
    def create_summary_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a summary DataFrame from analysis results
        
        Args:
            results: List of analysis results
        
        Returns:
            DataFrame with key metrics
        """
        summary_data = []
        
        for result in results:
            summary_data.append({
                'index': result['index'],
                'clean_title': result.get('headline', ''),
                'image_filename': result.get('image_id', ''),
                'label': result.get('label', ''),
                'person_names': ', '.join(result['person_names']),
                'num_persons': len(result['person_names']),
                'caption_length': len(result['caption']),
                'headline_length': len(result['headline']),
                'rouge1_f1': result['rouge1']['f1'],
                'rouge2_f1': result['rouge2']['f1'],
                'rougeL_f1': result['rougeL']['f1'],
                'semantic_score': result['semantic']['semantic_similarity_score'],
                'contextual_alignment': result['semantic']['contextual_alignment'],
                'vgs': result['vgs'],  # Direct score value
                'person_influence': result['person_influence']['influence'],
                'matched_persons': ', '.join(result['person_influence']['matched_persons']),
                'covered_tokens': len(result['coverage']['covered']),
                'missing_tokens': len(result['coverage']['missing']),
                'extra_tokens': len(result['coverage']['extra'])
            })
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = 'analysis_results.json'):
        """Save analysis results to JSON file with cleaned person names"""
        # Clean person_names before saving
        cleaned_results = []
        for result in results:
            result_copy = result.copy()
            # Ensure person_names is a clean list of strings
            if 'person_names' in result_copy:
                result_copy['person_names'] = [
                    str(name).replace("np.str_(", "").replace(")", "").replace("'", "").strip()
                    for name in result_copy['person_names']
                ]
            cleaned_results.append(result_copy)
        
        with open(output_path, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        print(f"\n✅ Results saved to '{output_path}'")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics"""
        summary_df = self.create_summary_dataframe(results)
        
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nTotal samples analyzed: {len(results)}")
        print(f"\n📊 Average Scores:")
        print(f"  ROUGE-1 F1: {summary_df['rouge1_f1'].mean():.3f} (±{summary_df['rouge1_f1'].std():.3f})")
        print(f"  ROUGE-2 F1: {summary_df['rouge2_f1'].mean():.3f} (±{summary_df['rouge2_f1'].std():.3f})")
        print(f"  ROUGE-L F1: {summary_df['rougeL_f1'].mean():.3f} (±{summary_df['rougeL_f1'].std():.3f})")
        print(f"  Visual Grounding Score (VGS): {summary_df['vgs'].mean():.3f} (±{summary_df['vgs'].std():.3f})")
        
        print(f"\n👥 Person Detection:")
        print(f"  Avg persons per image: {summary_df['num_persons'].mean():.2f}")
        print(f"  Person influence (high): {(summary_df['person_influence'] == 'high').sum()}/{len(results)}")
        
        print(f"\n💡 Interpretation Patterns:")
        high_vgs_low_rouge = ((summary_df['vgs'] >= 0.8) & (summary_df['rouge1_f1'] < 0.5)).sum()
        print(f"  High VGS + Low ROUGE (accurate but divergent): {high_vgs_low_rouge}/{len(results)}")
        
        print("\n" + "="*70)
        
        return summary_df


# Example usage
if __name__ == "__main__":

    # Initialize analyzer with image directory
    analyzer = CaptionSimilarityAnalyzer(
        image_dir="allData_images"
    )
    
    # Load your dataset
    df = pd.read_csv('datasets/processed/all_data_test_news_with_captions.csv')
    
    print("="*70)
    print("DATASET-BASED CAPTION SIMILARITY ANALYSIS WITH VGS")
    print("="*70)
    
    # Analyze random samples
    # num_samples = 5  # CHANGE THIS NUMBER AS NEEDED
    results = analyzer.analyze_dataset(
        df, 
        # num_samples=num_samples,
        # random_samples=True,
        # random_state=38
    )
    
    # Print summary
    summary_df = analyzer.print_summary(results)
    
    # Save results
    analyzer.save_results(results, 'all_data_caption_analysis_results.json')
    
    # Save summary to CSV
    summary_df.to_csv('all_data_caption_analysis_summary.csv', index=False)
    print("✅ Summary saved to 'all_data_caption_analysis_summary.csv'")
    
    # Print detailed results for first sample
    if results:
        print("\n" + "="*70)
        print("DETAILED EXAMPLE (First Sample)")
        print("="*70)
        first_result = results[0]
        print(f"\n📸 Caption: {first_result['caption']}")
        print(f"📰 Headline: {first_result['headline']}")
        print(f"👥 Detected: {first_result['person_names']}")
        print(f"\n📊 Scores:")
        print(f"  ROUGE-1: {first_result['rouge1']['f1']:.3f}")
        print(f"  Semantic: {first_result['semantic']['semantic_similarity_score']:.3f}")
        print(f"  VGS (Grounding DINO): {first_result['vgs']:.3f}")
        print(f"\n💡 Explanation: {first_result['semantic']['explanation']}")
#%%
# import json

# # Load the JSON file
# with open("../all_data_caption_analysis_results.json", "r") as f:
#     data = json.load(f)

# # Check type
# print(type(data))  # Will tell you if it's a list or dict

# # Count total objects
# if isinstance(data, list):
#     print("Total objects:", len(data))
# elif isinstance(data, dict):
#     print("Total keys:", len(data))
# %%
