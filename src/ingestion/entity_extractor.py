"""
Entity Extractor for education policy documents.

Extracts all types of entities needed for bridge table construction:
- Legal references (Section numbers, Articles, Rules, Clauses)
- GO references (GO numbers, departments, dates)
- Court case references
- Schemes (Nadu-Nedu, Amma Vodi, etc.)
- Districts (all 13 AP districts + variations)
- Metrics (PTR, GER, NER, etc.)
- Social categories (SC, ST, OBC, EWS)
- School types, educational levels, keywords
"""

import re
import json
import spacy
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """
    Comprehensive entity extraction for education policy documents.
    
    Extracts structured entities that feed into the bridge table system:
    - Legal references for citation tracking
    - Government order references for supersession chains
    - Schemes, districts, metrics for topic matching
    - Keywords for semantic understanding
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize entity extractor with dictionaries and NLP models.
        
        Args:
            data_dir: Path to data directory (defaults to project data/)
        """
        if data_dir is None:
            data_dir = project_root / "data"
        
        self.data_dir = Path(data_dir)
        self.dictionaries_dir = self.data_dir / "dictionaries"
        
        # Load dictionaries
        self._load_dictionaries()
        
        # Initialize spaCy model (load if available, otherwise use fallback)
        self._init_nlp_model()
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info(f"EntityExtractor initialized with {len(self.districts)} districts, "
                   f"{len(self.schemes)} schemes, {len(self.metrics)} metrics")
    
    def _load_dictionaries(self):
        """Load all dictionary data from JSON files."""
        try:
            # Education terms
            with open(self.dictionaries_dir / "education_terms.json", 'r') as f:
                edu_data = json.load(f)
                self.metrics = list(edu_data.get("terms", {}).keys())
                self.metric_synonyms = edu_data.get("terms", {})
            
            # AP Gazetteer
            with open(self.dictionaries_dir / "ap_gazetteer.json", 'r') as f:
                gazetteer = json.load(f)
                self.districts = gazetteer.get("districts", [])
                self.schemes = gazetteer.get("schemes", [])
        
        except FileNotFoundError as e:
            logger.warning(f"Dictionary file not found: {e}. Using minimal defaults.")
            self._load_default_dictionaries()
    
    def _load_default_dictionaries(self):
        """Load hardcoded dictionaries as fallback."""
        # AP Districts (all 13 + common variations)
        self.districts = [
            "Visakhapatnam", "Vizianagaram", "Srikakulam", "East Godavari", "West Godavari",
            "Krishna", "Guntur", "Prakasam", "Nellore", "Chittoor", "Kadapa", "Anantapur", "Kurnool",
            # Common variations
            "Vizag", "Rajahmundry", "Vijayawada", "Tirupati", "Eluru", "Machilipatnam",
            "Ongole", "Nandyal", "Hindupur", "Tadepalligudem"
        ]
        
        # Major education schemes
        self.schemes = [
            "Nadu-Nedu", "Jagananna Amma Vodi", "Amma Vodi", "Jagananna Gorumudda", 
            "Mid-Day Meal", "PM POSHAN", "Sarva Shiksha Abhiyan", "SSA",
            "Rashtriya Madhyamik Shiksha Abhiyan", "RMSA", "Samagra Shiksha",
            "Beti Bachao Beti Padhao", "Digital India", "Skill India"
        ]
        
        # Education metrics and indicators
        self.metrics = ["PTR", "GER", "NER", "dropout rate", "enrollment", "attendance"]
        self.metric_synonyms = {
            "PTR": ["pupil-teacher ratio", "teacher-pupil ratio", "student teacher ratio"],
            "GER": ["gross enrollment ratio", "gross enrolment ratio"],
            "NER": ["net enrollment ratio", "net enrolment ratio"]
        }
    
    def _init_nlp_model(self):
        """Initialize spaCy NLP model for entity recognition."""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy en_core_web_sm model")
        except OSError:
            try:
                # Fallback to basic English
                self.nlp = spacy.load("en_core_web_lg")
                logger.info("Loaded spaCy en_core_web_lg model")
            except OSError:
                logger.warning("No spaCy model found. Using pattern-based extraction only.")
                self.nlp = None
    
    def _compile_patterns(self):
        """Compile regex patterns for structured entity extraction."""
        
        # Legal references
        self.section_pattern = re.compile(
            r'\b(?:Section|Sec\.?|§)\s*(\d+)(?:\s*\(\s*(\d+)\s*\))?(?:\s*\(\s*([a-z]+)\s*\))?',
            re.IGNORECASE
        )
        
        self.article_pattern = re.compile(
            r'\bArticle\s*(\d+)(?:\s*([A-Z]+))?',
            re.IGNORECASE
        )
        
        self.rule_pattern = re.compile(
            r'\b(?:Rule|R\.?)\s*(\d+)(?:\s*\(\s*(\d+)\s*\))?',
            re.IGNORECASE
        )
        
        self.clause_pattern = re.compile(
            r'\bClause\s*(\d+)',
            re.IGNORECASE
        )
        
        # GO (Government Order) references
        self.go_pattern = re.compile(
            r'\bG\.?O\.?\s*(?:Ms\.?|MS\.?)?\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?(?:\s*(?:dated?|dt\.?)\s*([\d\-\.]+))?',
            re.IGNORECASE
        )
        
        # More specific GO patterns
        self.go_detailed_pattern = re.compile(
            r'\bG\.?O\.?\s*(Ms\.?|MS\.?|Rt\.?|RT\.?)\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?',
            re.IGNORECASE
        )
        
        # Court case patterns
        self.case_pattern = re.compile(
            r'([A-Z][a-zA-Z\s&]+)\s+(?:v\.?s?\.?|versus)\s+([A-Z][a-zA-Z\s&]+)',
            re.MULTILINE
        )
        
        # Date patterns
        self.date_pattern = re.compile(
            r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b|'
            r'\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b|'
            r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|'
            r'July|August|September|October|November|December)\s+(\d{4})\b',
            re.IGNORECASE
        )
        
        # Academic year pattern
        self.academic_year_pattern = re.compile(
            r'\b(\d{4})-?(\d{2,4})\b'
        )
        
        # Supersession patterns
        self.supersession_pattern = re.compile(
            r'\b(?:in\s+)?supersession\s+of\s+.*?G\.?O\.?\s*(?:Ms\.?|MS\.?)?\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?',
            re.IGNORECASE
        )
    
    def extract_legal_references(self, text: str) -> List[Dict]:
        """
        Extract legal references (sections, articles, rules, clauses).
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of legal reference dictionaries
        """
        legal_refs = []
        
        # Extract sections
        for match in self.section_pattern.finditer(text):
            section_num = match.group(1)
            subsection = match.group(2) if match.group(2) else None
            clause = match.group(3) if match.group(3) else None
            
            ref_text = match.group(0)
            
            legal_refs.append({
                "type": "section",
                "number": section_num,
                "subsection": subsection,
                "clause": clause,
                "text": ref_text,
                "position": match.span(),
                "act": self._infer_act_context(text, match.start())
            })
        
        # Extract articles
        for match in self.article_pattern.finditer(text):
            article_num = match.group(1)
            suffix = match.group(2) if match.group(2) else None
            
            legal_refs.append({
                "type": "article",
                "number": article_num,
                "suffix": suffix,
                "text": match.group(0),
                "position": match.span(),
                "act": self._infer_act_context(text, match.start())
            })
        
        # Extract rules
        for match in self.rule_pattern.finditer(text):
            rule_num = match.group(1)
            sub_rule = match.group(2) if match.group(2) else None
            
            legal_refs.append({
                "type": "rule",
                "number": rule_num,
                "sub_rule": sub_rule,
                "text": match.group(0),
                "position": match.span(),
                "act": self._infer_act_context(text, match.start())
            })
        
        # Extract clauses
        for match in self.clause_pattern.finditer(text):
            clause_num = match.group(1)
            
            legal_refs.append({
                "type": "clause",
                "number": clause_num,
                "text": match.group(0),
                "position": match.span(),
                "act": self._infer_act_context(text, match.start())
            })
        
        return legal_refs
    
    def _infer_act_context(self, text: str, position: int) -> Optional[str]:
        """
        Infer which Act/law the reference belongs to based on context.
        
        Args:
            text: Full text
            position: Position of the reference
            
        Returns:
            Inferred act name or None
        """
        # Look for act names in the surrounding context (±200 chars)
        start = max(0, position - 200)
        end = min(len(text), position + 200)
        context = text[start:end].lower()
        
        # Common education acts
        if "rte" in context or "right to education" in context:
            return "RTE Act"
        elif "education act" in context and ("1982" in context or "ap" in context or "andhra" in context):
            return "AP Education Act 1982"
        elif "constitution" in context:
            return "Constitution of India"
        elif "ncte" in context:
            return "NCTE Act"
        
        return None
    
    def extract_go_references(self, text: str) -> List[Dict]:
        """
        Extract Government Order references.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of GO reference dictionaries
        """
        go_refs = []
        
        # Extract detailed GO patterns first
        for match in self.go_detailed_pattern.finditer(text):
            dept = match.group(1).upper() if match.group(1) else "MS"
            go_num = match.group(2)
            year = match.group(3) if match.group(3) else None
            
            # Look for date in surrounding context
            go_text = match.group(0)
            date_match = self._find_nearby_date(text, match.end())
            
            go_refs.append({
                "type": "government_order",
                "number": int(go_num),
                "year": int(year) if year else None,
                "department": dept,
                "date": date_match,
                "text": go_text,
                "position": match.span()
            })
        
        # Extract basic GO patterns
        for match in self.go_pattern.finditer(text):
            # Skip if already captured by detailed pattern
            if any(abs(match.start() - ref["position"][0]) < 10 for ref in go_refs):
                continue
                
            go_num = match.group(1)
            year = match.group(2) if match.group(2) else None
            date_str = match.group(3) if match.group(3) else None
            
            go_refs.append({
                "type": "government_order",
                "number": int(go_num),
                "year": int(year) if year else None,
                "department": "MS",  # Default
                "date": self._parse_date(date_str) if date_str else None,
                "text": match.group(0),
                "position": match.span()
            })
        
        return go_refs
    
    def _find_nearby_date(self, text: str, position: int, window: int = 50) -> Optional[str]:
        """Find a date near the given position."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end]
        
        date_match = self.date_pattern.search(context)
        if date_match:
            return self._parse_date(date_match.group(0))
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse various date formats to ISO format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            ISO formatted date string or None
        """
        if not date_str:
            return None
        
        # Try different date formats
        formats = [
            "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y",
            "%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d",
            "%d %B %Y", "%d %b %Y"
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str.strip(), fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        return None
    
    def extract_schemes(self, text: str) -> List[str]:
        """
        Extract education scheme names.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of scheme names found
        """
        found_schemes = []
        text_lower = text.lower()
        
        for scheme in self.schemes:
            # Check for exact match (case insensitive)
            if scheme.lower() in text_lower:
                found_schemes.append(scheme)
        
        # Deduplicate while preserving order
        return list(dict.fromkeys(found_schemes))
    
    def extract_districts(self, text: str) -> List[str]:
        """
        Extract AP district names and variations.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of district names found
        """
        found_districts = []
        
        for district in self.districts:
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(district) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_districts.append(district)
        
        # Deduplicate while preserving order
        return list(dict.fromkeys(found_districts))
    
    def extract_metrics(self, text: str) -> List[str]:
        """
        Extract education metrics and indicators.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of metrics found
        """
        found_metrics = []
        text_lower = text.lower()
        
        for metric in self.metrics:
            # Check metric name
            if metric.lower() in text_lower:
                found_metrics.append(metric)
            
            # Check synonyms
            synonyms = self.metric_synonyms.get(metric, [])
            for synonym in synonyms:
                if synonym.lower() in text_lower:
                    found_metrics.append(metric)
                    break
        
        # Deduplicate while preserving order
        return list(dict.fromkeys(found_metrics))
    
    def extract_social_categories(self, text: str) -> List[str]:
        """
        Extract social categories (SC, ST, OBC, EWS, etc.).
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of social categories found
        """
        categories = ["SC", "ST", "OBC", "EWS", "General", "BPL", "APL", "DG", "EBC"]
        found_categories = []
        
        # Use word boundaries to match categories
        for category in categories:
            pattern = r'\b' + re.escape(category) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_categories.append(category)
        
        # Also check full forms
        full_forms = {
            "Scheduled Caste": "SC",
            "Scheduled Tribe": "ST", 
            "Other Backward Class": "OBC",
            "Economically Weaker Section": "EWS",
            "Disadvantaged Group": "DG",
            "Below Poverty Line": "BPL",
            "Above Poverty Line": "APL"
        }
        
        for full_form, abbrev in full_forms.items():
            if full_form.lower() in text.lower():
                found_categories.append(abbrev)
        
        return list(dict.fromkeys(found_categories))
    
    def extract_school_types(self, text: str) -> List[str]:
        """
        Extract school types (government, private aided, etc.).
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of school types found
        """
        school_types = [
            "government", "private aided", "private unaided", "central",
            "residential", "ashram", "tribal", "model", "composite"
        ]
        
        found_types = []
        text_lower = text.lower()
        
        for school_type in school_types:
            if school_type in text_lower:
                found_types.append(school_type)
        
        return list(dict.fromkeys(found_types))
    
    def extract_educational_levels(self, text: str) -> List[str]:
        """
        Extract educational levels.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of educational levels found
        """
        levels = [
            "primary", "upper primary", "elementary", "secondary", 
            "higher secondary", "pre-primary", "ECCE"
        ]
        
        found_levels = []
        text_lower = text.lower()
        
        for level in levels:
            if level in text_lower:
                found_levels.append(level)
        
        return list(dict.fromkeys(found_levels))
    
    def extract_spacy_entities(self, text: str) -> List[Dict]:
        """
        Extract entities using spaCy NER.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of spaCy entities
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def extract_all_entities(self, text: str) -> Dict:
        """
        Extract all types of entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing all extracted entities
        """
        if not text or not text.strip():
            return self._empty_entity_dict()
        
        try:
            entities = {
                "legal_refs": self.extract_legal_references(text),
                "go_refs": self.extract_go_references(text),
                "schemes": self.extract_schemes(text),
                "districts": self.extract_districts(text),
                "metrics": self.extract_metrics(text),
                "social_categories": self.extract_social_categories(text),
                "school_types": self.extract_school_types(text),
                "educational_levels": self.extract_educational_levels(text),
                "spacy_entities": self.extract_spacy_entities(text)
            }
            
            # Extract high-value keywords from spaCy entities
            entities["keywords"] = self._extract_keywords_from_entities(entities)
            
            logger.debug(f"Extracted entities: {sum(len(v) if isinstance(v, list) else 0 for v in entities.values())} total")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return self._empty_entity_dict()
    
    def _extract_keywords_from_entities(self, entities: Dict) -> List[str]:
        """Extract important keywords from various entity types."""
        keywords = []
        
        # Add scheme names as keywords
        keywords.extend(entities.get("schemes", []))
        
        # Add district names as keywords  
        keywords.extend(entities.get("districts", []))
        
        # Add metrics as keywords
        keywords.extend(entities.get("metrics", []))
        
        # Add relevant spaCy entities as keywords
        for ent in entities.get("spacy_entities", []):
            if ent["label"] in ["ORG", "PERSON", "GPE", "EVENT"]:
                # Clean and add if not too generic
                keyword = ent["text"].strip()
                if len(keyword) > 3 and keyword.lower() not in ["the", "and", "for", "with"]:
                    keywords.append(keyword)
        
        # Deduplicate and return
        return list(dict.fromkeys(keywords))
    
    def _empty_entity_dict(self) -> Dict:
        """Return empty entity dictionary with correct structure."""
        return {
            "legal_refs": [],
            "go_refs": [],
            "schemes": [],
            "districts": [],
            "metrics": [],
            "social_categories": [],
            "school_types": [],
            "educational_levels": [],
            "spacy_entities": [],
            "keywords": []
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the entity extractor
    extractor = EntityExtractor()
    
    # Test text with various entities
    test_text = """
    Section 12(1)(c) of the RTE Act mandates 25% reservation for EWS and DG children 
    in private unaided schools. As per GO MS No. 67/2023 dated 15.04.2023, 
    the Nadu-Nedu programme will be implemented in Visakhapatnam and Guntur districts.
    
    The PTR should be maintained at 30:1 in primary schools as per the norms.
    In supersession of GO MS No. 45/2018, this order covers SC and ST students.
    """
    
    entities = extractor.extract_all_entities(test_text)
    
    print("Extracted Entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"{entity_type}: {entity_list}")