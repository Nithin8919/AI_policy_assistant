"""
Judicial Database Builder for Case Law and Legal Precedents.

Creates a structured judicial database with:
- Case metadata (parties, court, judges, date)
- Legal principles and precedents
- Citation analysis and relationships
- Court hierarchy mapping
- Judgment classification
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime

from src.vertical_builders.base_builder import BaseVerticalBuilder, extract_number_from_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JudicialDatabaseBuilder(BaseVerticalBuilder):
    """
    Build specialized database for judicial cases and precedents.
    
    Creates structured access to case law with precedent relationships,
    legal principles, and citation analysis.
    """
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize judicial database builder."""
        super().__init__(data_dir, output_dir)
        
        # Initialize judicial patterns
        self._init_judicial_patterns()
        
        # Initialize court hierarchy
        self._init_court_hierarchy()
        
        # Statistics
        self.judicial_stats = {
            "cases_processed": 0,
            "precedents_identified": 0,
            "citations_extracted": 0,
            "legal_principles": 0
        }
    
    def _init_judicial_patterns(self):
        """Initialize patterns for judicial case extraction."""
        # Case citation patterns
        self.citation_patterns = [
            # AIR 2019 SC 1234
            re.compile(r'AIR\s+(\d{4})\s+(SC|[A-Z]{2,})\s+(\d+)', re.IGNORECASE),
            
            # (2019) 5 SCC 123
            re.compile(r'\((\d{4})\)\s+(\d+)\s+(SCC|SCR|All|Bom|Cal|Del|Ker|Mad|Ori)\s+(\d+)', re.IGNORECASE),
            
            # 2019 SCC 5 123
            re.compile(r'(\d{4})\s+(SCC|SCR|All|Bom|Cal|Del|Ker|Mad|Ori)\s+(\d+)\s+(\d+)', re.IGNORECASE)
        ]
        
        # Party name patterns (vs/v./versus)
        self.party_patterns = [
            re.compile(r'([A-Z][A-Za-z\s&.,]+?)\s+(?:vs?\.?|versus)\s+([A-Z][A-Za-z\s&.,]+)', re.IGNORECASE),
            re.compile(r'([A-Z][A-Za-z\s&.,]{5,})\s+v\.\s+([A-Z][A-Za-z\s&.,]{5,})', re.IGNORECASE)
        ]
        
        # Court identification patterns
        self.court_patterns = {
            "supreme_court": [
                re.compile(r'Supreme Court of India|Hon\'ble Supreme Court', re.IGNORECASE),
                re.compile(r'\bSC\b|\bSCI\b', re.IGNORECASE)
            ],
            "high_court": [
                re.compile(r'High Court of ([A-Za-z\s]+)', re.IGNORECASE),
                re.compile(r'([A-Za-z]+)\s+High Court', re.IGNORECASE)
            ],
            "tribunal": [
                re.compile(r'([A-Za-z\s]+)\s+Tribunal', re.IGNORECASE)
            ]
        }
        
        # Judge name patterns
        self.judge_patterns = [
            re.compile(r'(?:Hon\'ble\s+)?(?:Mr\.|Ms\.|Justice)\s+([A-Z][A-Za-z\s\.]+)', re.IGNORECASE),
            re.compile(r'(?:CJ|J\.)\s+([A-Z][A-Za-z\s\.]+)', re.IGNORECASE)
        ]
        
        # Legal principle indicators
        self.principle_keywords = [
            "held", "decided", "ruled", "principle", "ratio", "obiter dicta",
            "legal position", "settled law", "precedent", "binding"
        ]
        
        # Case status indicators
        self.status_keywords = {
            "decided": ["decided", "disposed", "allowed", "dismissed"],
            "pending": ["pending", "reserved", "admitted"],
            "interim": ["interim", "stay", "injunction"]
        }
    
    def _init_court_hierarchy(self):
        """Initialize court hierarchy mapping."""
        self.court_hierarchy = {
            "supreme_court": {
                "level": 1,
                "jurisdiction": "National",
                "binding_on": ["high_court", "district_court", "tribunal"]
            },
            "high_court": {
                "level": 2,
                "jurisdiction": "State",
                "binding_on": ["district_court", "tribunal"]
            },
            "district_court": {
                "level": 3,
                "jurisdiction": "District",
                "binding_on": []
            },
            "tribunal": {
                "level": 3,
                "jurisdiction": "Specialized",
                "binding_on": []
            }
        }
    
    def get_vertical_name(self) -> str:
        """Get vertical name for output directory."""
        return "judicial"
    
    def extract_case_citation(self, text: str) -> List[Dict]:
        """Extract case citations from text."""
        citations = []
        
        for pattern in self.citation_patterns:
            for match in pattern.finditer(text):
                citation = {
                    "full_citation": match.group(0).strip(),
                    "year": match.group(1),
                    "reporter": match.group(2),
                    "volume": match.group(3) if len(match.groups()) > 2 else None,
                    "page": match.group(4) if len(match.groups()) > 3 else None
                }
                citations.append(citation)
        
        return citations
    
    def extract_parties(self, text: str) -> Optional[Dict]:
        """Extract party names from case text."""
        for pattern in self.party_patterns:
            match = pattern.search(text)
            if match:
                return {
                    "petitioner": match.group(1).strip(),
                    "respondent": match.group(2).strip()
                }
        return None
    
    def identify_court(self, text: str) -> Optional[Dict]:
        """Identify court from text."""
        for court_type, patterns in self.court_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    court_info = {
                        "type": court_type,
                        "name": match.group(0).strip()
                    }
                    
                    # Extract specific court name for high courts
                    if court_type == "high_court" and len(match.groups()) > 0:
                        court_info["state"] = match.group(1).strip()
                    
                    # Add hierarchy info
                    court_info.update(self.court_hierarchy.get(court_type, {}))
                    
                    return court_info
        
        return None
    
    def extract_judges(self, text: str) -> List[str]:
        """Extract judge names from text."""
        judges = []
        
        for pattern in self.judge_patterns:
            for match in pattern.finditer(text):
                judge_name = match.group(1).strip()
                if len(judge_name) > 3 and judge_name not in judges:
                    judges.append(judge_name)
        
        return judges
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract judgment date from text."""
        # Date patterns in Indian legal documents
        date_patterns = [
            re.compile(r'(\d{1,2})[\./-](\d{1,2})[\./-](\d{4})'),
            re.compile(r'(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+(\d{4})', re.IGNORECASE)
        ]
        
        for pattern in date_patterns:
            match = pattern.search(text)
            if match:
                if len(match.groups()) == 3:
                    return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
                else:
                    return match.group(0)
        
        return None
    
    def extract_legal_principles(self, text: str) -> List[Dict]:
        """Extract legal principles and holdings."""
        principles = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check if sentence contains principle indicators
            for keyword in self.principle_keywords:
                if keyword.lower() in sentence.lower():
                    principle = {
                        "text": sentence,
                        "type": self._classify_principle(sentence),
                        "keywords": [keyword]
                    }
                    principles.append(principle)
                    break
        
        return principles
    
    def _classify_principle(self, text: str) -> str:
        """Classify the type of legal principle."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["held", "decided", "ruled"]):
            return "holding"
        elif "ratio" in text_lower:
            return "ratio_decidendi"
        elif "obiter" in text_lower:
            return "obiter_dicta"
        elif any(word in text_lower for word in ["principle", "law", "precedent"]):
            return "legal_principle"
        else:
            return "general"
    
    def extract_subject_matter(self, text: str) -> List[str]:
        """Extract legal subject matter/areas."""
        # Common legal areas in education domain
        legal_areas = [
            "education law", "constitutional law", "administrative law",
            "service law", "fundamental rights", "directive principles",
            "right to education", "reservation", "admission",
            "fee regulation", "private schools", "government schools"
        ]
        
        found_areas = []
        text_lower = text.lower()
        
        for area in legal_areas:
            if area in text_lower:
                found_areas.append(area)
        
        return found_areas
    
    def determine_case_status(self, text: str) -> str:
        """Determine case status from text."""
        text_lower = text.lower()
        
        for status, keywords in self.status_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return status
        
        return "unknown"
    
    def build_case_entry(self, doc_chunks: List[Dict], doc_id: str) -> Dict:
        """Build a single case entry from chunks."""
        # Combine text from all chunks
        combined_text = " ".join(chunk.get("text", "") for chunk in doc_chunks)
        
        # Extract case information
        parties = self.extract_parties(combined_text)
        court_info = self.identify_court(combined_text)
        judges = self.extract_judges(combined_text)
        citations = self.extract_case_citation(combined_text)
        judgment_date = self.extract_date(combined_text)
        legal_principles = self.extract_legal_principles(combined_text)
        subject_matter = self.extract_subject_matter(combined_text)
        case_status = self.determine_case_status(combined_text)
        
        # Extract metadata
        metadata = self.extract_metadata_from_chunks(doc_chunks)
        
        # Find related legal references from entities
        legal_refs = []
        statutory_refs = []
        
        for chunk in doc_chunks:
            entities = chunk.get("entities", {})
            legal_refs.extend(entities.get("legal_refs", []))
            statutory_refs.extend(entities.get("statutory_refs", []))
        
        # Build case entry
        case_entry = {
            "doc_id": doc_id,
            "case_name": f"{parties['petitioner']} vs {parties['respondent']}" if parties else "Unknown Case",
            "parties": parties,
            "court": court_info,
            "judges": judges,
            "judgment_date": judgment_date,
            "citations": citations,
            "legal_principles": legal_principles,
            "subject_matter": subject_matter,
            "status": case_status,
            "legal_references": list(set(legal_refs)),
            "statutory_references": list(set(statutory_refs)),
            "chunks": [chunk.get("chunk_id") for chunk in doc_chunks],
            "metadata": metadata
        }
        
        # Update stats
        self.judicial_stats["cases_processed"] += 1
        self.judicial_stats["citations_extracted"] += len(citations)
        self.judicial_stats["legal_principles"] += len(legal_principles)
        
        return case_entry
    
    def build_precedent_graph(self, cases: Dict) -> Dict:
        """Build precedent relationship graph."""
        precedent_graph = {
            "nodes": {},
            "edges": []
        }
        
        # Create nodes for each case
        for case_id, case_data in cases.items():
            precedent_graph["nodes"][case_id] = {
                "case_name": case_data.get("case_name"),
                "court_level": case_data.get("court", {}).get("level", 3),
                "year": case_data.get("metadata", {}).get("year"),
                "subject_matter": case_data.get("subject_matter", [])
            }
        
        # Find citation relationships
        for case_id, case_data in cases.items():
            case_text = " ".join(chunk.get("text", "") for chunk in case_data.get("chunks", []))
            
            # Look for citations to other cases in our database
            for other_case_id, other_case_data in cases.items():
                if case_id != other_case_id:
                    other_case_name = other_case_data.get("case_name", "")
                    
                    # Check if this case cites the other case
                    if other_case_name.lower() in case_text.lower():
                        precedent_graph["edges"].append({
                            "from": other_case_id,  # Precedent case
                            "to": case_id,          # Citing case
                            "relationship": "cited_by"
                        })
                        
                        self.judicial_stats["precedents_identified"] += 1
        
        return precedent_graph
    
    def build_database(self) -> Dict:
        """
        Build complete judicial database.
        
        Returns:
            Complete judicial database structure
        """
        # Load processed data
        chunks = self.load_processed_chunks()
        
        if not chunks:
            logger.error("No chunks loaded for judicial database building")
            return {}
        
        # Filter for judicial documents
        judicial_chunks = self.filter_chunks_by_doc_type(chunks, ["case", "judgment", "judicial"])
        
        if not judicial_chunks:
            logger.warning("No judicial chunks found")
            return {}
        
        # Group by document (each document is typically one case)
        doc_chunks = self.group_chunks_by_document(judicial_chunks)
        
        # Build case database
        cases = {}
        
        for doc_id, doc_chunk_list in doc_chunks.items():
            case_entry = self.build_case_entry(doc_chunk_list, doc_id)
            cases[doc_id] = case_entry
        
        # Build precedent graph
        precedent_graph = self.build_precedent_graph(cases)
        
        # Create comprehensive judicial database
        judicial_database = {
            "metadata": {
                "database_type": "judicial",
                "creation_date": datetime.now().isoformat(),
                "builder_version": "1.0.0",
                "total_cases": len(cases),
                "statistics": self.judicial_stats
            },
            "cases": cases,
            "precedent_graph": precedent_graph,
            "court_hierarchy": self.court_hierarchy,
            "subject_matter_index": self._build_subject_matter_index(cases)
        }
        
        # Update stats
        self.stats["database_entries_created"] = len(cases)
        
        return judicial_database
    
    def _build_subject_matter_index(self, cases: Dict) -> Dict:
        """Build index of cases by subject matter."""
        subject_index = defaultdict(list)
        
        for case_id, case_data in cases.items():
            for subject in case_data.get("subject_matter", []):
                subject_index[subject].append({
                    "case_id": case_id,
                    "case_name": case_data.get("case_name"),
                    "year": case_data.get("metadata", {}).get("year")
                })
        
        return dict(subject_index)


# Testing
if __name__ == "__main__":
    builder = JudicialDatabaseBuilder()
    
    print("Testing Judicial Database Builder...")
    
    # Test party extraction
    test_text = "State of Andhra Pradesh vs. Boddu Seshagiri Rao"
    parties = builder.extract_parties(test_text)
    print(f"Parties: {parties}")
    
    # Test court identification
    court = builder.identify_court("High Court of Andhra Pradesh")
    print(f"Court: {court}")
    
    # Test citation extraction
    citations = builder.extract_case_citation("AIR 2019 SC 1234")
    print(f"Citations: {citations}")
    
    print("\nJudicial Database Builder ready!")