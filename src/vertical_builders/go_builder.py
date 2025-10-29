"""
Government Orders Database Builder.

Creates a specialized GO database with:
- Active vs superseded GO tracking
- Supersession chain analysis  
- Topic-wise GO organization
- Department-wise indexing
- Implementation status tracking
- Effective date management
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime, date
from dataclasses import dataclass

from .base_builder import BaseVerticalBuilder, normalize_text_for_matching, extract_number_from_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GOReference:
    """Structure for GO reference information."""
    number: int
    year: Optional[int] = None
    department: str = "MS"
    date: Optional[str] = None
    full_text: str = ""


class GODatabaseBuilder(BaseVerticalBuilder):
    """
    Build specialized database for Government Orders.
    
    Creates structured access to GOs with supersession tracking,
    active status management, and topic-based organization.
    """
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize GO database builder."""
        super().__init__(data_dir, output_dir)
        
        # GO-specific patterns
        self._init_go_patterns()
        
        # Statistics
        self.go_stats = {
            "total_gos_processed": 0,
            "active_gos": 0,
            "superseded_gos": 0,
            "supersession_chains": 0,
            "orphan_gos": 0,
            "department_breakdown": {}
        }
    
    def _init_go_patterns(self):
        """Initialize regex patterns for GO parsing."""
        # GO number patterns (comprehensive)
        self.go_patterns = [
            # G.O.MS.No. 67/2023 or G.O.MS.No. 67 dated 15.04.2023
            re.compile(r'\bG\.?O\.?\s*(Ms\.?|MS\.?|Rt\.?|RT\.?)\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?(?:\s*(?:dated?|dt\.?)\s*([\d\-\.\/]+))?', re.IGNORECASE),
            
            # Simpler patterns: GO MS 67, MS No. 67
            re.compile(r'\b(?:GO|G\.O\.?)\s+(Ms\.?|MS\.?|Rt\.?|RT\.?)\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?', re.IGNORECASE),
            re.compile(r'\b(Ms\.?|MS\.?|Rt\.?|RT\.?)\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?', re.IGNORECASE)
        ]
        
        # Supersession patterns
        self.supersession_patterns = [
            re.compile(r'\bin supersession of\s+([^.!?]+?)(?:[.!?]|$)', re.IGNORECASE),
            re.compile(r'\bthis (?:order|GO|government order)\s+supersedes\s+([^.!?]+?)(?:[.!?]|$)', re.IGNORECASE),
            re.compile(r'\b(?:earlier|previous)\s+([^.!?]*?G\.?O\.?[^.!?]*?)\s+is hereby (?:cancelled|superseded)', re.IGNORECASE),
            re.compile(r'\b([^.!?]*?G\.?O\.?[^.!?]*?)\s+(?:stands?|is)\s+(?:superseded|cancelled)', re.IGNORECASE)
        ]
        
        # Implementation patterns
        self.implementation_patterns = [
            re.compile(r'\bin (?:pursuance|exercise) of\s+([^.!?]+?)(?:[.!?]|$)', re.IGNORECASE),
            re.compile(r'\bfor (?:the )?implementation of\s+([^.!?]+?)(?:[.!?]|$)', re.IGNORECASE),
            re.compile(r'\bunder\s+([^.!?]*?(?:Section|Article|Rule)[^.!?]*?)(?:[.!?]|$)', re.IGNORECASE)
        ]
        
        # Date patterns
        self.date_patterns = [
            re.compile(r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b'),
            re.compile(r'\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b'),
            re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE)
        ]
        
        # Department codes
        self.department_codes = {
            'MS': 'Main Secretariat',
            'RT': 'Roads & Transport', 
            'HM': 'Home',
            'REV': 'Revenue',
            'FIN': 'Finance',
            'EDU': 'Education',
            'HEALTH': 'Health & Family Welfare',
            'RURAL': 'Rural Development'
        }
    
    def get_vertical_name(self) -> str:
        """Get vertical name for output directory."""
        return "government_orders"
    
    def extract_go_reference(self, text: str) -> Optional[GOReference]:
        """
        Extract GO reference information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            GOReference object or None
        """
        for pattern in self.go_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                
                # Handle different pattern structures
                if len(groups) == 4:  # Full pattern with date
                    dept, number, year, date_str = groups
                elif len(groups) == 3:  # Without date or different structure
                    if groups[2] and groups[2].isdigit() and len(groups[2]) == 4:
                        dept, number, year = groups
                        date_str = None
                    else:
                        dept, number, date_str = groups
                        year = None
                elif len(groups) == 2:  # Simple pattern
                    dept, number = groups
                    year = None
                    date_str = None
                else:
                    continue
                
                try:
                    go_ref = GOReference(
                        number=int(number),
                        year=int(year) if year else None,
                        department=dept.upper().replace('.', ''),
                        date=self._parse_date(date_str) if date_str else None,
                        full_text=match.group(0)
                    )
                    return go_ref
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format."""
        if not date_str:
            return None
        
        # Month name mapping
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for pattern in self.date_patterns:
            match = pattern.search(date_str)
            if match:
                groups = match.groups()
                
                try:
                    if len(groups) == 3:
                        if groups[1].isalpha():  # Month name format
                            day, month_name, year = groups
                            month = months.get(month_name.lower())
                            if month:
                                return f"{year}-{month:02d}-{int(day):02d}"
                        else:  # Numeric format
                            first, second, third = groups
                            if len(third) == 4:  # DD.MM.YYYY
                                day, month, year = first, second, third
                            else:  # YYYY.MM.DD
                                year, month, day = first, second, third
                            
                            return f"{year}-{int(month):02d}-{int(day):02d}"
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def extract_supersession_info(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract supersession information from GO chunks.
        
        Args:
            chunks: List of chunks for a GO document
            
        Returns:
            List of supersession records
        """
        supersessions = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id")
            
            for pattern in self.supersession_patterns:
                for match in pattern.finditer(text):
                    superseded_text = match.group(1).strip()
                    
                    # Extract GO reference from superseded text
                    superseded_go = self.extract_go_reference(superseded_text)
                    
                    supersession = {
                        "chunk_id": chunk_id,
                        "superseded_text": superseded_text,
                        "superseded_go": superseded_go.__dict__ if superseded_go else None,
                        "context": text[max(0, match.start()-50):match.end()+50]
                    }
                    
                    supersessions.append(supersession)
        
        return supersessions
    
    def extract_implementation_info(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract implementation information (what laws/sections this GO implements).
        
        Args:
            chunks: List of chunks for a GO document
            
        Returns:
            List of implementation records
        """
        implementations = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id")
            
            for pattern in self.implementation_patterns:
                for match in pattern.finditer(text):
                    implemented_text = match.group(1).strip()
                    
                    implementation = {
                        "chunk_id": chunk_id,
                        "implements": implemented_text,
                        "context": text[max(0, match.start()-50):match.end()+50],
                        "type": self._classify_implementation_type(implemented_text)
                    }
                    
                    implementations.append(implementation)
        
        return implementations
    
    def _classify_implementation_type(self, impl_text: str) -> str:
        """Classify the type of implementation."""
        impl_lower = impl_text.lower()
        
        if "section" in impl_lower:
            return "legal_section"
        elif "article" in impl_lower:
            return "constitutional_article"
        elif "rule" in impl_lower:
            return "rule"
        elif "act" in impl_lower:
            return "act"
        else:
            return "general"
    
    def build_supersession_chains(self, go_database: Dict) -> Dict[str, List[str]]:
        """
        Build supersession chains showing GO evolution.
        
        Args:
            go_database: Dictionary of GO entries
            
        Returns:
            Dictionary mapping GO IDs to their supersession chains
        """
        chains = {}
        
        # Build supersession graph
        superseded_by = {}  # superseded_go -> superseding_go
        supersedes = defaultdict(list)  # superseding_go -> [superseded_gos]
        
        for go_id, go_data in go_database.items():
            supersession_info = go_data.get("supersession_info", [])
            
            for supersession in supersession_info:
                superseded_go_data = supersession.get("superseded_go")
                if superseded_go_data:
                    # Create superseded GO identifier
                    superseded_key = f"{superseded_go_data['department']}_{superseded_go_data['number']}"
                    if superseded_go_data.get('year'):
                        superseded_key += f"_{superseded_go_data['year']}"
                    
                    superseded_by[superseded_key] = go_id
                    supersedes[go_id].append(superseded_key)
        
        # Build chains by following supersession links
        for go_id in go_database.keys():
            chain = [go_id]
            
            # Follow chain backwards (what this GO supersedes)
            current = go_id
            while current in supersedes and supersedes[current]:
                # Take first superseded GO (could be multiple)
                superseded = supersedes[current][0]
                chain.append(superseded)
                current = superseded
                
                # Prevent infinite loops
                if len(chain) > 10:
                    break
            
            if len(chain) > 1:
                chains[go_id] = chain
        
        return chains
    
    def determine_active_status(self, go_data: Dict, supersession_chains: Dict) -> str:
        """
        Determine if a GO is active, superseded, or unknown.
        
        Args:
            go_data: GO data dictionary
            supersession_chains: Supersession chains mapping
            
        Returns:
            Status: 'active', 'superseded', or 'unknown'
        """
        go_id = go_data.get("doc_id")
        
        # Check if this GO is superseded by another
        for chain_go, chain in supersession_chains.items():
            if go_id in chain[1:]:  # Not the first (current) GO in chain
                return "superseded"
        
        # Check if this GO has supersession info (it supersedes others)
        if go_data.get("supersession_info"):
            return "active"
        
        # Check date - very old GOs are likely superseded
        go_ref = go_data.get("go_reference")
        if go_ref and go_ref.get("year"):
            year = go_ref["year"]
            current_year = datetime.now().year
            if current_year - year > 10:  # More than 10 years old
                return "likely_superseded"
        
        # Default to active if recent or unclear
        return "active"
    
    def organize_by_topics(self, go_database: Dict) -> Dict[str, List[str]]:
        """
        Organize GOs by bridge topics.
        
        Args:
            go_database: Dictionary of GO entries
            
        Returns:
            Dictionary mapping topics to GO IDs
        """
        topic_gos = defaultdict(list)
        
        for go_id, go_data in go_database.items():
            # Get bridge topics from chunks
            chunks = go_data.get("chunk_data", [])
            
            for chunk in chunks:
                bridge_topics = chunk.get("bridge_topics", [])
                for topic_match in bridge_topics:
                    topic_id = topic_match.get("topic_id")
                    if topic_id:
                        if go_id not in topic_gos[topic_id]:
                            topic_gos[topic_id].append(go_id)
        
        return dict(topic_gos)
    
    def organize_by_department(self, go_database: Dict) -> Dict[str, List[str]]:
        """
        Organize GOs by department.
        
        Args:
            go_database: Dictionary of GO entries
            
        Returns:
            Dictionary mapping departments to GO IDs
        """
        dept_gos = defaultdict(list)
        
        for go_id, go_data in go_database.items():
            go_ref = go_data.get("go_reference", {})
            department = go_ref.get("department", "UNKNOWN")
            dept_gos[department].append(go_id)
        
        # Update stats
        self.go_stats["department_breakdown"] = {
            dept: len(gos) for dept, gos in dept_gos.items()
        }
        
        return dict(dept_gos)
    
    def build_go_timeline(self, go_database: Dict) -> List[Dict]:
        """
        Build chronological timeline of GOs.
        
        Args:
            go_database: Dictionary of GO entries
            
        Returns:
            List of GO entries sorted by date
        """
        timeline = []
        
        for go_id, go_data in go_database.items():
            go_ref = go_data.get("go_reference", {})
            
            timeline_entry = {
                "go_id": go_id,
                "go_number": go_ref.get("number"),
                "year": go_ref.get("year"),
                "date": go_ref.get("date"),
                "department": go_ref.get("department"),
                "title": go_data.get("title", ""),
                "status": go_data.get("status"),
                "supersedes": len(go_data.get("supersession_info", [])) > 0
            }
            
            timeline.append(timeline_entry)
        
        # Sort by date (most recent first)
        timeline.sort(key=lambda x: (
            x.get("year", 0),
            x.get("date", ""),
            x.get("go_number", 0)
        ), reverse=True)
        
        return timeline
    
    def build_database(self) -> Dict:
        """
        Build complete GO database.
        
        Returns:
            Complete GO database structure
        """
        # Load processed data
        chunks = self.load_processed_chunks()
        relations = self.load_relations()
        
        if not chunks:
            logger.error("No chunks loaded for GO database building")
            return {}
        
        # Filter for government order documents
        go_chunks = self.filter_chunks_by_doc_type(chunks, ["government_order"])
        
        if not go_chunks:
            logger.warning("No government order chunks found")
            return {}
        
        # Group chunks by document
        doc_chunks = self.group_chunks_by_document(go_chunks)
        
        # Build GO database
        go_database = {}
        
        for doc_id, doc_chunks_list in doc_chunks.items():
            # Extract metadata
            metadata = self.extract_metadata_from_chunks(doc_chunks_list)
            
            # Extract GO reference from document
            combined_text = " ".join(chunk.get("text", "") for chunk in doc_chunks_list)
            go_ref = self.extract_go_reference(combined_text)
            
            # Extract supersession information
            supersession_info = self.extract_supersession_info(doc_chunks_list)
            
            # Extract implementation information
            implementation_info = self.extract_implementation_info(doc_chunks_list)
            
            # Get relations for this document
            doc_relations = self.find_relations_for_document(relations, doc_id)
            
            # Build GO entry
            go_entry = {
                "doc_id": doc_id,
                "title": metadata.get("title", ""),
                "go_reference": go_ref.__dict__ if go_ref else None,
                "issue_date": go_ref.date if go_ref else metadata.get("year"),
                "department": go_ref.department if go_ref else "UNKNOWN",
                "go_number": go_ref.number if go_ref else None,
                "year": go_ref.year if go_ref else metadata.get("year"),
                "supersession_info": supersession_info,
                "implementation_info": implementation_info,
                "relations": doc_relations,
                "total_chunks": len(doc_chunks_list),
                "chunk_ids": [chunk.get("chunk_id") for chunk in doc_chunks_list],
                "chunk_data": doc_chunks_list,  # Store for topic analysis
                "temporal_info": metadata.get("temporal_info", {}),
                "file_path": metadata.get("file_path"),
                "processing_date": datetime.now().isoformat()
            }
            
            go_database[doc_id] = go_entry
            self.go_stats["total_gos_processed"] += 1
        
        # Build supersession chains
        supersession_chains = self.build_supersession_chains(go_database)
        
        # Determine active status for each GO
        for go_id, go_data in go_database.items():
            status = self.determine_active_status(go_data, supersession_chains)
            go_data["status"] = status
            
            if status == "active":
                self.go_stats["active_gos"] += 1
            elif status in ["superseded", "likely_superseded"]:
                self.go_stats["superseded_gos"] += 1
        
        # Build topic and department organizations
        topic_organization = self.organize_by_topics(go_database)
        department_organization = self.organize_by_department(go_database)
        
        # Build timeline
        go_timeline = self.build_go_timeline(go_database)
        
        # Filter active GOs for quick access
        active_gos = {
            go_id: go_data for go_id, go_data in go_database.items()
            if go_data.get("status") == "active"
        }
        
        # Create comprehensive GO database
        complete_database = {
            "metadata": {
                "database_type": "government_orders",
                "creation_date": datetime.now().isoformat(),
                "builder_version": "1.0.0",
                "total_gos": len(go_database),
                "active_gos": len(active_gos),
                "statistics": self.go_stats
            },
            "all_gos": go_database,
            "active_gos": active_gos,
            "supersession_chains": supersession_chains,
            "topic_organization": topic_organization,
            "department_organization": department_organization,
            "go_timeline": go_timeline,
            "department_index": {
                dept: {
                    "full_name": self.department_codes.get(dept, dept),
                    "go_count": len(go_ids),
                    "go_ids": go_ids
                }
                for dept, go_ids in department_organization.items()
            }
        }
        
        # Update final stats
        self.go_stats["supersession_chains"] = len(supersession_chains)
        self.stats["database_entries_created"] = len(go_database)
        
        return complete_database


# Example usage and testing
if __name__ == "__main__":
    # Test the GO database builder
    builder = GODatabaseBuilder()
    
    print("Testing GO Database Builder...")
    
    # Test GO reference extraction
    test_texts = [
        "G.O.MS.No. 67/2023 dated 15.04.2023",
        "GO MS No. 45 dt. 10.03.2018",
        "In supersession of G.O.MS.No. 34/2010",
        "MS No. 123 dated 25.12.2020"
    ]
    
    print("Testing GO reference extraction:")
    for text in test_texts:
        go_ref = builder.extract_go_reference(text)
        if go_ref:
            print(f"Text: {text}")
            print(f"Extracted: {go_ref}")
            print()
    
    # Test date parsing
    test_dates = ["15.04.2023", "10-03-2018", "25th December 2020"]
    print("Testing date parsing:")
    for date_str in test_dates:
        parsed = builder._parse_date(date_str)
        print(f"Date: {date_str} -> {parsed}")
    
    print("\nGO Database Builder ready for processing!")