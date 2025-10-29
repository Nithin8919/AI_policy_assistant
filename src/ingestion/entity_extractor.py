"""
Enhanced Entity Extractor for AP Education Policy Documents.

Extracts domain-specific entities including:
- Legal references (Sections, Articles, Rules)
- Government Orders (GO references)
- AP Education Schemes
- AP Districts
- Social categories (SC/ST/OBC)
- School types
- Educational levels
- Metrics and statistics
- Keywords
- Named entities (using spaCy)
"""

import re
from typing import Dict, List, Set, Optional
from collections import Counter, defaultdict
import logging

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy --break-system-packages")

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """
    Extract domain-specific entities from education policy documents.
    
    Combines pattern matching with NLP for comprehensive entity extraction.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize entity extractor with patterns and models."""
        self.data_dir = data_dir
        
        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Initialize all patterns
        self._init_patterns()
        self._init_ap_schemes()
        self._init_ap_districts()
        self._init_categories()
        
        logger.info("EntityExtractor initialized")
    
    def _init_patterns(self):
        """Initialize regex patterns for entity extraction."""
        
        # ============================================================================
        # GO REFERENCE PATTERNS (Enhanced)
        # ============================================================================
        
        # Primary GO pattern (comprehensive)
        self.GO_PATTERN = re.compile(
            r'''
            (?:G\.O\.|GO|G\.O|Ms\.No\.|MS\s+No\.|Rt\.No\.|RT\s+No\.)  # Prefix variations
            \s*
            (?:MS|RT|No\.?)?                                           # Type (MS/RT/No)
            \s*
            (?:No\.?)?                                                 # Optional "No."
            \s*
            (\d+)                                                      # GO number (capture group 1)
            (?:
                \s*/\s*(\d{4})                                        # Optional /YYYY format
                |
                \s+dated?\s+(\d{1,2}[\./-]\d{1,2}[\./-]\d{2,4})      # Optional date
                |
                \s+dt\.?\s+(\d{1,2}[\./-]\d{1,2}[\./-]\d{2,4})       # Alternative date format
            )?
            ''',
            re.IGNORECASE | re.VERBOSE
        )
        
        # Additional GO patterns for context-specific matches
        self.GO_PATTERNS_ADDITIONAL = [
            re.compile(r'vide\s+(?:G\.O\.|GO)(?:MS|RT)?\.?\s*No\.?\s*(\d+)', re.IGNORECASE),
            re.compile(r'in\s+(?:G\.O\.|GO)(?:MS|RT)?\.?\s*No\.?\s*(\d+)', re.IGNORECASE),
            re.compile(r'as per\s+(?:G\.O\.|GO)(?:MS|RT)?\.?\s*No\.?\s*(\d+)', re.IGNORECASE),
            re.compile(r'(?:G\.O\.|GO)(?:MS|RT)?\.?\s*No\.?\s*(\d+)/(\d{4})', re.IGNORECASE),
            re.compile(r'MS\s+No\.?\s*(\d+)\s+dt\.?\s+(\d{1,2}[\./-]\d{1,2}[\./-]\d{2,4})', re.IGNORECASE)
        ]
        
        # ============================================================================
        # LEGAL REFERENCE PATTERNS
        # ============================================================================
        
        self.legal_patterns = [
            # Section patterns
            re.compile(r'\bSection\s+(\d+)(?:\s*\(\s*(\d+)\s*\))?(?:\s*\(\s*([a-z]+)\s*\))?', re.IGNORECASE),
            re.compile(r'\bSec\.?\s+(\d+)(?:\s*\(\s*(\d+)\s*\))?', re.IGNORECASE),
            
            # Article patterns
            re.compile(r'\bArticle\s+(\d+)(?:\s*([A-Z]+))?', re.IGNORECASE),
            re.compile(r'\bArt\.?\s+(\d+)(?:\s*([A-Z]+))?', re.IGNORECASE),
            
            # Rule patterns
            re.compile(r'\bRule\s+(\d+)(?:\s*\(\s*(\d+)\s*\))?', re.IGNORECASE),
            re.compile(r'\bR\.?\s+(\d+)', re.IGNORECASE),
            
            # Clause patterns
            re.compile(r'\bClause\s+(\d+)', re.IGNORECASE),
            
            # Chapter patterns
            re.compile(r'\bChapter\s+([IVX]+|\d+)', re.IGNORECASE),
        ]
        
        # ============================================================================
        # METRIC PATTERNS
        # ============================================================================
        
        self.metric_patterns = [
            re.compile(r'\b(gross enrollment ratio|GER)\b', re.IGNORECASE),
            re.compile(r'\b(net enrollment ratio|NER)\b', re.IGNORECASE),
            re.compile(r'\b(pupil[- ]teacher ratio|PTR|student[- ]teacher ratio)\b', re.IGNORECASE),
            re.compile(r'\b(dropout rate|drop[- ]out)\b', re.IGNORECASE),
            re.compile(r'\b(retention rate)\b', re.IGNORECASE),
            re.compile(r'\b(transition rate)\b', re.IGNORECASE),
            re.compile(r'\b(pass percentage|pass rate)\b', re.IGNORECASE),
            re.compile(r'\b(learning outcome|learning outcomes)\b', re.IGNORECASE),
            re.compile(r'\b(enrollment|enrolment)\b', re.IGNORECASE),
        ]
    
    def _init_ap_schemes(self):
        """Initialize AP education schemes list."""
        self.AP_SCHEMES = [
            # Major Flagship Schemes
            "Nadu-Nedu", "Nadu Nedu", "Naadu-Nedu",
            "Amma Vodi", "Amma Vodi Scheme", "Ammavodi",
            "Jagananna Gorumudda", "Gorumudda", "Mid Day Meal", "MDM Scheme",
            "Jagananna Vidya Deevena", "Vidya Deevena",
            "Jagananna Vidya Kanuka", "Vidya Kanuka",
            "Jagananna Amma Odi",
            
            # Infrastructure & Development
            "Smart School Programme", "Smart Schools",
            "Digital Classroom Initiative",
            "Mana Badi Nadu Nedu", "Vasathi Deevena",
            
            # Teacher & Training
            "Teacher Training Programme", "In-Service Teacher Training",
            "DIET Programme", "District Institute of Education and Training",
            
            # Student Welfare
            "SC/ST Welfare Schemes", "BC Welfare Hostel Scheme",
            "EWS Scholarship Scheme", "Minority Welfare Scheme",
            "Girl Child Education Scheme",
            "Kasturba Gandhi Balika Vidyalaya", "KGBV",
            
            # Inclusive Education
            "Inclusive Education Scheme", "CWSN Support Programme",
            "Children With Special Needs",
            
            # Quality Improvement
            "Quality Improvement Programme",
            "School Quality Assessment",
            "Learning Enhancement Programme",
            
            # Vocational & Skill Development
            "Skill Development Programme",
            "Vocational Education Scheme",
            
            # Other Schemes
            "Samajik Abhisarata Programme",
            "Sarva Shiksha Abhiyan", "SSA",
            "Rashtriya Madhyamik Shiksha Abhiyan", "RMSA",
            "Samagra Shiksha",
            "Beti Bachao Beti Padhao", "BBBP",
            "Atal Tinkering Labs", "ATL",
            "Grama/Ward Sachivalayam Education Programme",
            "Bridge Course Programme"
        ]
        
        # Create scheme pattern
        self.SCHEME_PATTERN = re.compile(
            r'\b(' + '|'.join(re.escape(scheme) for scheme in self.AP_SCHEMES) + r')\b',
            re.IGNORECASE
        )
    
    def _init_ap_districts(self):
        """Initialize AP districts list with variations."""
        self.AP_DISTRICTS = [
            # District names with variations
            "Anantapur", "Ananthapuramu", "Ananthapuram",
            "Chittoor", "Chittor",
            "East Godavari", "East Godavari District", "Godavari East",
            "Guntur",
            "Krishna", "Krishna District",
            "Kurnool",
            "Nellore", "Nellore District", "SPSR Nellore",
            "Sri Potti Sriramulu Nellore",
            "Prakasam", "Prakasam District",
            "Srikakulam", "Sri Kakulam",
            "Visakhapatnam", "Vishakhapatnam", "Vizag", "Vizag District",
            "Vizianagaram", "Vijayanagaram",
            "West Godavari", "West Godavari District", "Godavari West",
            "YSR Kadapa", "YSR", "Kadapa", "Cuddapah"
        ]
        
        # Create district pattern
        self.DISTRICT_PATTERN = re.compile(
            r'\b(' + '|'.join(re.escape(district) for district in self.AP_DISTRICTS) + r')\s*(?:district)?',
            re.IGNORECASE
        )
    
    def _init_categories(self):
        """Initialize social categories, school types, educational levels."""
        
        # Social Categories
        self.SOCIAL_CATEGORIES = [
            "SC", "Scheduled Caste", "Scheduled Castes",
            "ST", "Scheduled Tribe", "Scheduled Tribes",
            "OBC", "Other Backward Class", "Other Backward Classes",
            "BC", "Backward Class", "Backward Classes",
            "EWS", "Economically Weaker Section", "Economically Weaker Sections",
            "BC-A", "BC-B", "BC-C", "BC-D", "BC-E",
            "Girl Child", "Girl Students", "Female Students",
            "Boy Students", "Male Students",
            "Below Poverty Line", "BPL", "Above Poverty Line", "APL",
            "Rural Students", "Urban Students", "Tribal Area Students",
            "Minority Students", "Muslim Minority", "Christian Minority",
            "Children With Special Needs", "CWSN",
            "Differently Abled", "Divyang", "Orphan Students", "Orphans"
        ]
        
        self.SOCIAL_CATEGORY_PATTERN = re.compile(
            r'\b(' + '|'.join(re.escape(cat) for cat in self.SOCIAL_CATEGORIES) + r')\b',
            re.IGNORECASE
        )
        
        # School Types
        self.SCHOOL_TYPES = [
            "Government School", "Govt School", "Government Schools",
            "Zilla Parishad School", "ZP School", "ZPHS", "ZPSS",
            "Mandal Parishad School", "MPP School", "MPPS",
            "Municipal School", "Corporation School",
            "Private School", "Private Schools",
            "Aided School", "Aided Schools",
            "Unaided School", "Unaided Schools",
            "Private Aided", "Private Unaided",
            "Residential School", "Residential Schools",
            "Welfare Residential School",
            "Tribal Welfare School", "APTWRS",
            "BC Welfare Hostel", "SC/ST Welfare Hostel",
            "Kasturba Gandhi Balika Vidyalaya", "KGBV",
            "Model School", "Model Schools",
            "Sainik School", "Navodaya Vidyalaya", "JNV",
            "Kendriya Vidyalaya", "KV",
            "Primary School", "Upper Primary School",
            "High School", "Higher Secondary School",
            "Junior College"
        ]
        
        self.SCHOOL_TYPE_PATTERN = re.compile(
            r'\b(' + '|'.join(re.escape(stype) for stype in self.SCHOOL_TYPES) + r')\b',
            re.IGNORECASE
        )
        
        # Educational Levels
        self.EDUCATIONAL_LEVELS = [
            "Pre-Primary", "Pre Primary", "Anganwadi",
            "Primary", "Primary Level", "Classes I-V", "Class 1 to 5",
            "Upper Primary", "Classes VI-VIII", "Class 6 to 8",
            "Secondary", "High School", "Classes IX-X", "Class 9 to 10",
            "Higher Secondary", "Intermediate", "Classes XI-XII", "Class 11 to 12",
            "Junior College",
            "Class I", "Class II", "Class III", "Class IV", "Class V",
            "Class VI", "Class VII", "Class VIII",
            "Class IX", "Class X", "Class XI", "Class XII",
            "SSC", "10th Standard", "10th Class",
            "Intermediate", "12th Standard", "12th Class",
            "Science Stream", "Commerce Stream", "Arts Stream",
            "MPC", "BiPC", "CEC", "HEC"
        ]
        
        self.EDUCATIONAL_LEVEL_PATTERN = re.compile(
            r'\b(' + '|'.join(re.escape(level) for level in self.EDUCATIONAL_LEVELS) + r')\b',
            re.IGNORECASE
        )
    
    # ============================================================================
    # EXTRACTION METHODS
    # ============================================================================
    
    def extract_go_refs(self, text: str) -> List[str]:
        """Extract Government Order references."""
        go_refs = []
        
        # Primary pattern
        for match in self.GO_PATTERN.finditer(text):
            go_ref = self._format_go_reference(match)
            if go_ref:
                go_refs.append(go_ref)
        
        # Additional patterns for context-specific matches
        for pattern in self.GO_PATTERNS_ADDITIONAL:
            for match in pattern.finditer(text):
                go_ref = self._format_go_reference(match)
                if go_ref:
                    go_refs.append(go_ref)
        
        return list(set(go_refs))  # Deduplicate
    
    def _format_go_reference(self, match) -> Optional[str]:
        """Format GO reference from regex match."""
        try:
            groups = match.groups()
            go_number = groups[0] if groups else None
            
            if not go_number:
                return None
            
            # Try to extract year and date
            year = None
            date = None
            
            for i in range(1, len(groups)):
                if groups[i] and len(groups[i]) == 4 and groups[i].isdigit():
                    year = groups[i]
                elif groups[i] and ('/' in groups[i] or '-' in groups[i] or '.' in groups[i]):
                    date = groups[i]
            
            # Format GO reference
            if year:
                return f"G.O.MS.No. {go_number}/{year}"
            elif date:
                return f"G.O.MS.No. {go_number} dated {date}"
            else:
                return f"G.O.MS.No. {go_number}"
        except Exception as e:
            logger.debug(f"Error formatting GO reference: {e}")
            return None
    
    def extract_legal_refs(self, text: str) -> List[str]:
        """Extract legal references (Sections, Articles, Rules)."""
        legal_refs = []
        
        for pattern in self.legal_patterns:
            for match in pattern.finditer(text):
                ref = match.group(0)
                # Clean up the reference
                ref = re.sub(r'\s+', ' ', ref).strip()
                legal_refs.append(ref)
        
        return list(set(legal_refs))
    
    def extract_schemes(self, text: str) -> List[str]:
        """Extract AP education scheme names."""
        schemes = []
        
        for match in self.SCHEME_PATTERN.finditer(text):
            scheme = match.group(0)
            schemes.append(scheme)
        
        return list(set(schemes))
    
    def extract_districts(self, text: str) -> List[str]:
        """Extract AP district names."""
        districts = []
        
        for match in self.DISTRICT_PATTERN.finditer(text):
            district = match.group(1)  # Get the captured district name
            # Normalize district name
            district = district.strip()
            districts.append(district)
        
        return list(set(districts))
    
    def extract_social_categories(self, text: str) -> List[str]:
        """Extract social categories (SC/ST/OBC etc)."""
        categories = []
        
        for match in self.SOCIAL_CATEGORY_PATTERN.finditer(text):
            category = match.group(0)
            categories.append(category)
        
        return list(set(categories))
    
    def extract_school_types(self, text: str) -> List[str]:
        """Extract school types."""
        school_types = []
        
        for match in self.SCHOOL_TYPE_PATTERN.finditer(text):
            school_type = match.group(0)
            school_types.append(school_type)
        
        return list(set(school_types))
    
    def extract_educational_levels(self, text: str) -> List[str]:
        """Extract educational levels."""
        levels = []
        
        for match in self.EDUCATIONAL_LEVEL_PATTERN.finditer(text):
            level = match.group(0)
            levels.append(level)
        
        return list(set(levels))
    
    def extract_metrics(self, text: str) -> List[str]:
        """Extract education metrics and statistics."""
        metrics = []
        
        for pattern in self.metric_patterns:
            for match in pattern.finditer(text):
                metric = match.group(0)
                metrics.append(metric)
        
        return list(set(metrics))
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract important keywords using frequency analysis."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'this', 'that', 'these', 'those', 'it', 'its', 'as', 'will', 'shall'
        }
        
        # Extract words (4+ characters)
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        
        # Filter stop words
        words = [w for w in words if w not in stop_words]
        
        # Count frequency
        word_freq = Counter(words)
        
        # Return top N keywords
        return [word for word, count in word_freq.most_common(top_n)]
    
    def extract_spacy_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy."""
        if not self.nlp or not text:
            return []
        
        # Limit text length for performance
        if len(text) > 100000:
            text = text[:100000]
        
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
        
        return entities
    
    def extract_all_entities(self, text: str) -> Dict[str, List]:
        """
        Extract all entity types from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing all extracted entities by type
        """
        if not text or not text.strip():
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
        
        entities = {
            "legal_refs": self.extract_legal_refs(text),
            "go_refs": self.extract_go_refs(text),
            "schemes": self.extract_schemes(text),
            "districts": self.extract_districts(text),
            "metrics": self.extract_metrics(text),
            "social_categories": self.extract_social_categories(text),
            "school_types": self.extract_school_types(text),
            "educational_levels": self.extract_educational_levels(text),
            "spacy_entities": self.extract_spacy_entities(text),
            "keywords": self.extract_keywords(text)
        }
        
        return entities


# Testing
if __name__ == "__main__":
    extractor = EntityExtractor()
    
    print("Testing Enhanced Entity Extractor\n")
    
    # Test text with multiple entity types
    test_text = """
    G.O.MS.No. 67/2023 dated 15.04.2023 announces the Nadu-Nedu scheme 
    implementation in Visakhapatnam district and Krishna district. 
    The scheme will benefit SC/ST students in Government Schools and 
    Zilla Parishad Schools. As per Section 5 of the AP Education Act 1982,
    the pupil-teacher ratio (PTR) should be maintained at 30:1 for primary level.
    The enrollment in Classes I-V has increased. Amma Vodi scheme provides 
    financial assistance to mothers.
    """
    
    print("Test Text:")
    print(test_text)
    print("\n" + "="*80 + "\n")
    
    # Extract all entities
    entities = extractor.extract_all_entities(test_text)
    
    # Display results
    print("Extraction Results:\n")
    for entity_type, entity_list in entities.items():
        if entity_type == "spacy_entities":
            print(f"{entity_type}: {len(entity_list)} entities")
            for ent in entity_list[:5]:  # Show first 5
                print(f"  - {ent['text']} ({ent['label']})")
        else:
            print(f"{entity_type}: {entity_list}")
    
    print("\n" + "="*80 + "\n")
    print("Entity Extractor Test Complete!")
    print(f"\nTotal entities extracted: {sum(len(v) if isinstance(v, list) else 0 for v in entities.values())}")