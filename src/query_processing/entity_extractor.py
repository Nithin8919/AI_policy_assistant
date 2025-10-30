"""
Query Entity Extraction Module

Extracts structured entities from queries:
- Districts (with admin levels)
- Schemes (with aliases)
- Metrics (with units)
- Dates and temporal spans
- Legal references (Acts, Sections, Rules, GOs)
- Educational entities (schools, levels, subjects)
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Structured entity with metadata"""
    text: str
    type: str
    canonical: str
    confidence: float
    span: Tuple[int, int]  # Start, end position in query
    metadata: Dict[str, any]


class QueryEntityExtractor:
    """
    Comprehensive entity extraction from queries using
    gazetteer matching, pattern recognition, and contextual rules.
    """
    
    def __init__(self, dictionaries_path: str = "data/dictionaries"):
        """
        Initialize entity extractor with dictionaries.
        
        Args:
            dictionaries_path: Path to dictionaries directory
        """
        self.dictionaries_path = Path(dictionaries_path)
        self.gazetteer = self._load_gazetteer()
        self.acronyms = self._load_acronyms()
        self.education_terms = self._load_education_terms()
        
        # Build comprehensive scheme database
        self.schemes = self._build_scheme_database()
        self.metrics = self._build_metrics_database()
        self.districts = self._build_district_database()
        
        logger.info("QueryEntityExtractor initialized")
    
    def _load_gazetteer(self) -> Dict[str, List[str]]:
        """Load AP gazetteer"""
        try:
            path = self.dictionaries_path / "ap_gazetteer.json"
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load gazetteer: {e}")
            return {}
    
    def _load_acronyms(self) -> Dict[str, List[str]]:
        """Load acronyms"""
        try:
            path = self.dictionaries_path / "acronyms.json"
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load acronyms: {e}")
            return {}
    
    def _load_education_terms(self) -> Dict[str, List[str]]:
        """Load education terms"""
        try:
            path = self.dictionaries_path / "education_terms.json"
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get("terms", {})
        except Exception as e:
            logger.warning(f"Could not load education terms: {e}")
            return {}
    
    def _build_scheme_database(self) -> Dict[str, Dict]:
        """Build comprehensive scheme database with aliases"""
        schemes = {
            "nadu-nedu": {
                "canonical": "Nadu-Nedu",
                "aliases": ["nadu nedu", "nadunedu", "naduneedu"],
                "type": "infrastructure",
                "vertical": "schemes"
            },
            "amma vodi": {
                "canonical": "Jagananna Amma Vodi",
                "aliases": ["ammavodi", "amma vodi", "jagananna amma vodi"],
                "type": "financial_assistance",
                "vertical": "schemes"
            },
            "gorumudda": {
                "canonical": "Jagananna Gorumudda",
                "aliases": ["goru mudda", "mid day meal", "mdm"],
                "type": "nutrition",
                "vertical": "schemes"
            },
            "vidya deevena": {
                "canonical": "Vidya Deevena",
                "aliases": ["vidyadeevena", "fee reimbursement"],
                "type": "financial_assistance",
                "vertical": "schemes"
            },
            "ssa": {
                "canonical": "Sarva Shiksha Abhiyan",
                "aliases": ["sarva shiksha abhiyan", "ssa"],
                "type": "universal_education",
                "vertical": "schemes"
            },
            "rmsa": {
                "canonical": "Rashtriya Madhyamik Shiksha Abhiyan",
                "aliases": ["rashtriya madhyamik shiksha abhiyan", "rmsa"],
                "type": "secondary_education",
                "vertical": "schemes"
            }
        }
        
        # Add from gazetteer
        for scheme in self.gazetteer.get("schemes", []):
            key = scheme.lower()
            if key not in schemes:
                schemes[key] = {
                    "canonical": scheme,
                    "aliases": [scheme.lower()],
                    "type": "general",
                    "vertical": "schemes"
                }
        
        return schemes
    
    def _build_metrics_database(self) -> Dict[str, Dict]:
        """Build metrics database with units and categories"""
        metrics = {
            "enrollment": {
                "canonical": "enrollment",
                "aliases": ["enrolment", "student enrollment", "admission"],
                "unit": "count",
                "category": "access"
            },
            "ptr": {
                "canonical": "Pupil-Teacher Ratio",
                "aliases": ["ptr", "pupil teacher ratio", "teacher pupil ratio"],
                "unit": "ratio",
                "category": "quality"
            },
            "ger": {
                "canonical": "Gross Enrollment Ratio",
                "aliases": ["ger", "gross enrollment ratio"],
                "unit": "percentage",
                "category": "access"
            },
            "ner": {
                "canonical": "Net Enrollment Ratio",
                "aliases": ["ner", "net enrollment ratio"],
                "unit": "percentage",
                "category": "access"
            },
            "dropout": {
                "canonical": "Dropout Rate",
                "aliases": ["dropout", "dropout rate", "attrition"],
                "unit": "percentage",
                "category": "retention"
            },
            "budget": {
                "canonical": "Budget Allocation",
                "aliases": ["budget", "allocation", "expenditure", "spending"],
                "unit": "currency",
                "category": "financial"
            },
            "infrastructure": {
                "canonical": "Infrastructure",
                "aliases": ["infrastructure", "facilities", "buildings"],
                "unit": "count",
                "category": "infrastructure"
            }
        }
        
        # Add from education terms
        for term, expansions in self.education_terms.items():
            if term.lower() not in metrics:
                metrics[term.lower()] = {
                    "canonical": expansions[0] if expansions else term,
                    "aliases": [exp.lower() for exp in expansions],
                    "unit": "unknown",
                    "category": "general"
                }
        
        return metrics
    
    def _build_district_database(self) -> Dict[str, Dict]:
        """Build district database with admin levels"""
        districts = {}
        
        # AP districts with metadata
        ap_districts = [
            ("Visakhapatnam", ["vizag", "visakhapatnam", "vishakhapatnam"], "north_coastal"),
            ("Vijayawada", ["vijayawada", "bezawada"], "central"),
            ("Guntur", ["guntur"], "central"),
            ("Tirupati", ["tirupati", "tirupathi"], "rayalaseema"),
            ("Rajahmundry", ["rajahmundry", "rajahmundri", "rajamahendravaram"], "coastal"),
            ("Anantapur", ["anantapur", "ananthpur"], "rayalaseema"),
            ("Chittoor", ["chittoor"], "rayalaseema"),
            ("Kurnool", ["kurnool"], "rayalaseema"),
            ("Nellore", ["nellore"], "coastal"),
            ("Kadapa", ["kadapa", "cuddapah"], "rayalaseema"),
            ("Prakasam", ["prakasam", "ongole"], "coastal"),
            ("Srikakulam", ["srikakulam"], "north_coastal"),
            ("Vizianagaram", ["vizianagaram"], "north_coastal"),
        ]
        
        for canonical, aliases, region in ap_districts:
            key = canonical.lower()
            districts[key] = {
                "canonical": canonical,
                "aliases": aliases,
                "region": region,
                "state": "Andhra Pradesh",
                "admin_level": "district"
            }
        
        # Add from gazetteer
        for district in self.gazetteer.get("districts", []):
            key = district.lower()
            if key not in districts:
                districts[key] = {
                    "canonical": district,
                    "aliases": [district.lower()],
                    "region": "unknown",
                    "state": "Andhra Pradesh",
                    "admin_level": "district"
                }
        
        return districts
    
    def extract(self, query: str) -> Dict[str, List[Entity]]:
        """
        Extract all entities from query.
        
        Args:
            query: Query string (preferably normalized)
            
        Returns:
            Dictionary of entity lists by type
        """
        entities = {
            "districts": [],
            "schemes": [],
            "metrics": [],
            "dates": [],
            "legal_references": [],
            "go_numbers": [],
            "educational_levels": [],
            "subjects": [],
        }
        
        query_lower = query.lower()
        
        # Extract districts
        entities["districts"] = self._extract_districts(query, query_lower)
        
        # Extract schemes
        entities["schemes"] = self._extract_schemes(query, query_lower)
        
        # Extract metrics
        entities["metrics"] = self._extract_metrics(query, query_lower)
        
        # Extract dates
        entities["dates"] = self._extract_dates(query)
        
        # Extract legal references
        entities["legal_references"] = self._extract_legal_references(query)
        
        # Extract GO numbers
        entities["go_numbers"] = self._extract_go_numbers(query)
        
        # Extract educational levels
        entities["educational_levels"] = self._extract_educational_levels(query, query_lower)
        
        # Extract subjects
        entities["subjects"] = self._extract_subjects(query, query_lower)
        
        logger.debug(f"Extracted entities from query: {entities}")
        return entities
    
    def _extract_districts(self, query: str, query_lower: str) -> List[Entity]:
        """Extract district entities with confidence scoring"""
        entities = []
        
        for key, district_info in self.districts.items():
            # Check canonical name
            pattern = r'\b' + re.escape(district_info["canonical"].lower()) + r'\b'
            match = re.search(pattern, query_lower)
            
            if match:
                entities.append(Entity(
                    text=query[match.start():match.end()],
                    type="district",
                    canonical=district_info["canonical"],
                    confidence=1.0,
                    span=(match.start(), match.end()),
                    metadata={
                        "region": district_info["region"],
                        "state": district_info["state"],
                        "admin_level": district_info["admin_level"]
                    }
                ))
                continue
            
            # Check aliases
            for alias in district_info["aliases"]:
                pattern = r'\b' + re.escape(alias) + r'\b'
                match = re.search(pattern, query_lower)
                if match:
                    entities.append(Entity(
                        text=query[match.start():match.end()],
                        type="district",
                        canonical=district_info["canonical"],
                        confidence=0.9,  # Slightly lower for aliases
                        span=(match.start(), match.end()),
                        metadata={
                            "region": district_info["region"],
                            "state": district_info["state"],
                            "admin_level": district_info["admin_level"],
                            "matched_via": "alias"
                        }
                    ))
                    break
        
        return entities
    
    def _extract_schemes(self, query: str, query_lower: str) -> List[Entity]:
        """Extract scheme entities"""
        entities = []
        
        for key, scheme_info in self.schemes.items():
            # Check all aliases
            for alias in [scheme_info["canonical"].lower()] + scheme_info["aliases"]:
                pattern = r'\b' + re.escape(alias) + r'\b'
                match = re.search(pattern, query_lower)
                if match:
                    confidence = 1.0 if alias == scheme_info["canonical"].lower() else 0.9
                    entities.append(Entity(
                        text=query[match.start():match.end()],
                        type="scheme",
                        canonical=scheme_info["canonical"],
                        confidence=confidence,
                        span=(match.start(), match.end()),
                        metadata={
                            "scheme_type": scheme_info["type"],
                            "vertical": scheme_info["vertical"]
                        }
                    ))
                    break
        
        return entities
    
    def _extract_metrics(self, query: str, query_lower: str) -> List[Entity]:
        """Extract metric entities"""
        entities = []
        
        for key, metric_info in self.metrics.items():
            # Check all aliases
            for alias in [metric_info["canonical"].lower()] + metric_info["aliases"]:
                pattern = r'\b' + re.escape(alias) + r'\b'
                match = re.search(pattern, query_lower)
                if match:
                    entities.append(Entity(
                        text=query[match.start():match.end()],
                        type="metric",
                        canonical=metric_info["canonical"],
                        confidence=0.95,
                        span=(match.start(), match.end()),
                        metadata={
                            "unit": metric_info["unit"],
                            "category": metric_info["category"]
                        }
                    ))
                    break
        
        return entities
    
    def _extract_dates(self, query: str) -> List[Entity]:
        """Extract date and temporal entities"""
        entities = []
        
        # Academic year pattern: 2023-24, 2023-2024
        pattern_ay = r'\b(\d{4})-(\d{2,4})\b'
        for match in re.finditer(pattern_ay, query):
            entities.append(Entity(
                text=match.group(0),
                type="academic_year",
                canonical=match.group(0),
                confidence=1.0,
                span=(match.start(), match.end()),
                metadata={"year_start": match.group(1), "year_end": match.group(2)}
            ))
        
        # Financial year pattern: FY 2023-24
        pattern_fy = r'\bFY\s*(\d{4})-(\d{2,4})\b'
        for match in re.finditer(pattern_fy, query, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                type="financial_year",
                canonical=match.group(0),
                confidence=1.0,
                span=(match.start(), match.end()),
                metadata={"year_start": match.group(1), "year_end": match.group(2)}
            ))
        
        # Standard date: DD/MM/YYYY, DD-MM-YYYY
        pattern_date = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'
        for match in re.finditer(pattern_date, query):
            entities.append(Entity(
                text=match.group(0),
                type="date",
                canonical=match.group(0),
                confidence=0.9,
                span=(match.start(), match.end()),
                metadata={"day": match.group(1), "month": match.group(2), "year": match.group(3)}
            ))
        
        # Year only: 2023, 2024
        pattern_year = r'\b(19|20)\d{2}\b'
        for match in re.finditer(pattern_year, query):
            # Avoid matching academic years already caught
            if not any(e.span[0] <= match.start() < e.span[1] for e in entities):
                entities.append(Entity(
                    text=match.group(0),
                    type="year",
                    canonical=match.group(0),
                    confidence=0.7,
                    span=(match.start(), match.end()),
                    metadata={"year": match.group(0)}
                ))
        
        return entities
    
    def _extract_legal_references(self, query: str) -> List[Entity]:
        """Extract legal references (Acts, Sections, Rules)"""
        entities = []
        
        # Act references
        pattern_act = r'(?:AP|Andhra Pradesh)?\s*(\w+(?:\s+\w+)*)\s+Act[,\s]+(\d{4})'
        for match in re.finditer(pattern_act, query, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                type="act",
                canonical=match.group(0),
                confidence=0.95,
                span=(match.start(), match.end()),
                metadata={"act_name": match.group(1), "year": match.group(2)}
            ))
        
        # Section references
        pattern_section = r'Section\s+(\d+)(?:\(([a-z0-9]+)\))?(?:\(([a-z0-9]+)\))?'
        for match in re.finditer(pattern_section, query, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                type="section",
                canonical=match.group(0),
                confidence=1.0,
                span=(match.start(), match.end()),
                metadata={
                    "section": match.group(1),
                    "subsection": match.group(2),
                    "clause": match.group(3)
                }
            ))
        
        # Rule references
        pattern_rule = r'Rule\s+(\d+)(?:\(([a-z0-9]+)\))?'
        for match in re.finditer(pattern_rule, query, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                type="rule",
                canonical=match.group(0),
                confidence=1.0,
                span=(match.start(), match.end()),
                metadata={"rule": match.group(1), "subsection": match.group(2)}
            ))
        
        return entities
    
    def _extract_go_numbers(self, query: str) -> List[Entity]:
        """Extract Government Order numbers"""
        entities = []
        
        patterns = [
            r'G\.O\.(?:Ms|Rt|MS|RT)\.?\s*No\.?\s*(\d+)',
            r'GO\s+(?:MS|RT|Ms|Rt)\s+(?:No\.?)?\s*(\d+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(0),
                    type="go_number",
                    canonical=match.group(0).upper(),
                    confidence=1.0,
                    span=(match.start(), match.end()),
                    metadata={"go_number": match.group(1)}
                ))
        
        return entities
    
    def _extract_educational_levels(self, query: str, query_lower: str) -> List[Entity]:
        """Extract educational level entities"""
        entities = []
        
        levels = {
            "primary": ["primary", "elementary", "classes 1-5", "class 1 to 5"],
            "upper primary": ["upper primary", "classes 6-8", "class 6 to 8"],
            "secondary": ["secondary", "high school", "classes 9-10", "class 9 to 10"],
            "higher secondary": ["higher secondary", "+2", "classes 11-12", "intermediate"],
        }
        
        for canonical, aliases in levels.items():
            for alias in aliases:
                pattern = r'\b' + re.escape(alias) + r'\b'
                match = re.search(pattern, query_lower)
                if match:
                    entities.append(Entity(
                        text=query[match.start():match.end()],
                        type="educational_level",
                        canonical=canonical,
                        confidence=0.9,
                        span=(match.start(), match.end()),
                        metadata={"level": canonical}
                    ))
                    break
        
        return entities
    
    def _extract_subjects(self, query: str, query_lower: str) -> List[Entity]:
        """Extract subject entities"""
        entities = []
        
        subjects = ["mathematics", "science", "english", "telugu", "hindi", "social studies", "physics", "chemistry", "biology"]
        
        for subject in subjects:
            pattern = r'\b' + re.escape(subject) + r'\b'
            match = re.search(pattern, query_lower)
            if match:
                entities.append(Entity(
                    text=query[match.start():match.end()],
                    type="subject",
                    canonical=subject.title(),
                    confidence=0.9,
                    span=(match.start(), match.end()),
                    metadata={"subject": subject}
                ))
        
        return entities
    
    def to_dict(self, entities: Dict[str, List[Entity]]) -> Dict[str, List[Dict]]:
        """Convert entities to dictionary format"""
        result = {}
        for entity_type, entity_list in entities.items():
            result[entity_type] = [asdict(e) for e in entity_list]
        return result


# Convenience function for backwards compatibility
def extract_entities(query: str, dictionaries: dict = None) -> dict:
    """Extract entities from query (backwards compatible)"""
    extractor = QueryEntityExtractor()
    entities = extractor.extract(query)
    
    # Convert to simpler format
    simple_format = {
        "districts": [e.canonical for e in entities["districts"]],
        "schemes": [e.canonical for e in entities["schemes"]],
        "metrics": [e.canonical for e in entities["metrics"]],
        "dates": [e.text for e in entities["dates"]],
    }
    
    return simple_format


# Main API function
def process_entity_extraction(query: str, include_metadata: bool = True) -> Dict[str, any]:
    """
    Process query through entity extraction pipeline.
    
    Args:
        query: Query string (preferably normalized)
        include_metadata: Whether to include full metadata
        
    Returns:
        Extracted entities with optional metadata
    """
    extractor = QueryEntityExtractor()
    entities = extractor.extract(query)
    
    if include_metadata:
        return extractor.to_dict(entities)
    else:
        return extract_entities(query)
