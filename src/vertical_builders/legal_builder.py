"""
Legal Database Builder for Acts and Rules.

Creates a structured legal database with:
- Acts index with section hierarchy
- Rules index with clause structure
- Cross-reference mapping (which sections cite which)
- Amendment history tracking
- Legal precedent connections
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime

from .base_builder import BaseVerticalBuilder, normalize_text_for_matching, extract_number_from_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LegalDatabaseBuilder(BaseVerticalBuilder):
    """
    Build specialized database for Acts and Rules.
    
    Creates structured access to legal documents with section hierarchy,
    cross-references, and amendment tracking for legal agents.
    """
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize legal database builder."""
        super().__init__(data_dir, output_dir)
        
        # Legal document patterns
        self._init_legal_patterns()
        
        # Statistics
        self.legal_stats = {
            "acts_processed": 0,
            "rules_processed": 0,
            "total_sections": 0,
            "total_cross_references": 0,
            "amendments_tracked": 0
        }
    
    def _init_legal_patterns(self):
        """Initialize regex patterns for legal document parsing."""
        # Section patterns
        self.section_patterns = [
            re.compile(r'\bSection\s+(\d+)(?:\s*\(\s*(\d+)\s*\))?(?:\s*\(\s*([a-z]+)\s*\))?', re.IGNORECASE),
            re.compile(r'\bSec\.?\s+(\d+)(?:\s*\(\s*(\d+)\s*\))?(?:\s*\(\s*([a-z]+)\s*\))?', re.IGNORECASE)
        ]
        
        # Chapter patterns
        self.chapter_patterns = [
            re.compile(r'\bChapter\s+([IVX]+|\d+)', re.IGNORECASE),
            re.compile(r'\bChap\.?\s+([IVX]+|\d+)', re.IGNORECASE)
        ]
        
        # Rule patterns
        self.rule_patterns = [
            re.compile(r'\bRule\s+(\d+)(?:\s*\(\s*(\d+)\s*\))?', re.IGNORECASE),
            re.compile(r'\bR\.?\s+(\d+)(?:\s*\(\s*(\d+)\s*\))?', re.IGNORECASE)
        ]
        
        # Article patterns (for Constitution)
        self.article_patterns = [
            re.compile(r'\bArticle\s+(\d+)(?:\s*([A-Z]+))?', re.IGNORECASE),
            re.compile(r'\bArt\.?\s+(\d+)(?:\s*([A-Z]+))?', re.IGNORECASE)
        ]
        
        # Amendment patterns
        self.amendment_patterns = [
            re.compile(r'\bamendment\b', re.IGNORECASE),
            re.compile(r'\bamended\b', re.IGNORECASE),
            re.compile(r'\bin supersession\b', re.IGNORECASE),
            re.compile(r'\bsubstituted\b', re.IGNORECASE)
        ]
    
    def get_vertical_name(self) -> str:
        """Get vertical name for output directory."""
        return "legal"
    
    def extract_section_hierarchy(self, chunks: List[Dict]) -> Dict:
        """
        Extract section hierarchy from legal document chunks.
        
        Args:
            chunks: List of chunks from a legal document
            
        Returns:
            Section hierarchy dictionary
        """
        sections = {}
        
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id")
            
            # Find all sections mentioned in this chunk
            for pattern in self.section_patterns:
                for match in pattern.finditer(text):
                    section_num = match.group(1)
                    subsection = match.group(2) if match.lastindex >= 2 else None
                    clause = match.group(3) if match.lastindex >= 3 else None
                    
                    # Build section identifier
                    section_id = section_num
                    if subsection:
                        section_id += f"({subsection})"
                    if clause:
                        section_id += f"({clause})"
                    
                    # Store section information
                    if section_id not in sections:
                        sections[section_id] = {
                            "section_number": section_num,
                            "subsection": subsection,
                            "clause": clause,
                            "full_id": section_id,
                            "chunk_ids": [],
                            "text_snippets": [],
                            "cross_references": []
                        }
                    
                    # Add chunk reference
                    if chunk_id not in sections[section_id]["chunk_ids"]:
                        sections[section_id]["chunk_ids"].append(chunk_id)
                    
                    # Extract surrounding text
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 200)
                    snippet = text[start:end].strip()
                    
                    if snippet not in sections[section_id]["text_snippets"]:
                        sections[section_id]["text_snippets"].append(snippet)
        
        return sections
    
    def extract_cross_references(self, chunks: List[Dict], relations: List[Dict]) -> Dict[str, List[str]]:
        """
        Extract cross-references between legal sections.
        
        Args:
            chunks: Document chunks
            relations: Extracted relations
            
        Returns:
            Dictionary mapping sections to what they reference
        """
        cross_refs = defaultdict(list)
        
        # Process relations for citations
        for relation in relations:
            if relation.get("relation_type") == "cites":
                source = relation.get("source_chunk_id", "")
                target = relation.get("target_reference", "")
                
                # Extract section from target reference
                target_section = self._extract_section_from_reference(target)
                if target_section:
                    cross_refs[source].append(target_section)
        
        # Also scan chunk text directly for references
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id")
            
            # Find references like "as per Section 12", "under Section 6"
            ref_patterns = [
                re.compile(r'\b(?:as per|under|pursuant to|in accordance with)\s+Section\s+(\d+(?:\([^)]+\))*)', re.IGNORECASE),
                re.compile(r'\bSection\s+(\d+(?:\([^)]+\))*)\s+(?:provides|mandates|states)', re.IGNORECASE)
            ]
            
            for pattern in ref_patterns:
                for match in pattern.finditer(text):
                    referenced_section = match.group(1)
                    if referenced_section not in cross_refs[chunk_id]:
                        cross_refs[chunk_id].append(referenced_section)
        
        return dict(cross_refs)
    
    def _extract_section_from_reference(self, reference: str) -> Optional[str]:
        """Extract clean section identifier from reference text."""
        for pattern in self.section_patterns:
            match = pattern.search(reference)
            if match:
                section_num = match.group(1)
                subsection = match.group(2) if match.lastindex >= 2 else None
                clause = match.group(3) if match.lastindex >= 3 else None
                
                section_id = section_num
                if subsection:
                    section_id += f"({subsection})"
                if clause:
                    section_id += f"({clause})"
                
                return section_id
        
        return None
    
    def identify_amendments(self, chunks: List[Dict], metadata: Dict) -> List[Dict]:
        """
        Identify amendments and changes in legal documents.
        
        Args:
            chunks: Document chunks
            metadata: Document metadata
            
        Returns:
            List of amendment records
        """
        amendments = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id")
            
            # Check for amendment indicators
            has_amendment = any(pattern.search(text) for pattern in self.amendment_patterns)
            
            if has_amendment:
                amendment = {
                    "chunk_id": chunk_id,
                    "amendment_type": self._classify_amendment_type(text),
                    "text_snippet": text[:300] + "..." if len(text) > 300 else text,
                    "year": metadata.get("year"),
                    "document": metadata.get("title", "")
                }
                
                # Try to extract what's being amended
                amended_refs = self._extract_amended_references(text)
                if amended_refs:
                    amendment["amends"] = amended_refs
                
                amendments.append(amendment)
        
        return amendments
    
    def _classify_amendment_type(self, text: str) -> str:
        """Classify the type of amendment."""
        text_lower = text.lower()
        
        if "substituted" in text_lower:
            return "substitution"
        elif "inserted" in text_lower or "added" in text_lower:
            return "insertion"
        elif "omitted" in text_lower or "deleted" in text_lower:
            return "deletion"
        elif "supersession" in text_lower:
            return "supersession"
        else:
            return "modification"
    
    def _extract_amended_references(self, text: str) -> List[str]:
        """Extract references to what is being amended."""
        refs = []
        
        # Look for patterns like "Section X is amended", "in Section Y"
        patterns = [
            re.compile(r'Section\s+(\d+(?:\([^)]+\))*)\s+(?:is|shall be)\s+(?:amended|substituted)', re.IGNORECASE),
            re.compile(r'in\s+Section\s+(\d+(?:\([^)]+\))*)', re.IGNORECASE)
        ]
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                section_ref = match.group(1)
                if section_ref not in refs:
                    refs.append(section_ref)
        
        return refs
    
    def build_acts_database(self, chunks: List[Dict], relations: List[Dict]) -> Dict:
        """
        Build Acts database from chunks and relations.
        
        Args:
            chunks: Filtered chunks for Acts
            relations: All relations
            
        Returns:
            Acts database structure
        """
        acts_db = {}
        
        # Group chunks by document
        doc_chunks = self.group_chunks_by_document(chunks)
        
        for doc_id, doc_chunks_list in doc_chunks.items():
            # Extract metadata
            metadata = self.extract_metadata_from_chunks(doc_chunks_list)
            
            # Extract section hierarchy
            sections = self.extract_section_hierarchy(doc_chunks_list)
            
            # Get relations for this document
            doc_relations = self.find_relations_for_document(relations, doc_id)
            
            # Extract cross-references
            cross_refs = self.extract_cross_references(doc_chunks_list, doc_relations)
            
            # Identify amendments
            amendments = self.identify_amendments(doc_chunks_list, metadata)
            
            # Build act entry
            act_entry = {
                "doc_id": doc_id,
                "title": metadata.get("title", ""),
                "enactment_year": metadata.get("year"),
                "doc_type": "act",
                "total_sections": len(sections),
                "sections": sections,
                "cross_references": cross_refs,
                "amendments": amendments,
                "total_chunks": len(doc_chunks_list),
                "chunk_ids": [chunk.get("chunk_id") for chunk in doc_chunks_list],
                "temporal_info": metadata.get("temporal_info", {}),
                "file_path": metadata.get("file_path"),
                "processing_date": datetime.now().isoformat()
            }
            
            # Add act-specific analysis
            act_entry["key_sections"] = self._identify_key_sections(sections)
            act_entry["section_count_by_type"] = self._analyze_section_types(sections)
            
            acts_db[doc_id] = act_entry
            self.legal_stats["acts_processed"] += 1
            self.legal_stats["total_sections"] += len(sections)
        
        return acts_db
    
    def build_rules_database(self, chunks: List[Dict], relations: List[Dict]) -> Dict:
        """
        Build Rules database from chunks and relations.
        
        Args:
            chunks: Filtered chunks for Rules
            relations: All relations
            
        Returns:
            Rules database structure
        """
        rules_db = {}
        
        # Group chunks by document
        doc_chunks = self.group_chunks_by_document(chunks)
        
        for doc_id, doc_chunks_list in doc_chunks.items():
            # Extract metadata
            metadata = self.extract_metadata_from_chunks(doc_chunks_list)
            
            # Extract rule hierarchy (similar to sections but for rules)
            rules = self.extract_rule_hierarchy(doc_chunks_list)
            
            # Get relations for this document
            doc_relations = self.find_relations_for_document(relations, doc_id)
            
            # Find implementing relations (rules implement acts)
            implementing_relations = [
                rel for rel in doc_relations 
                if rel.get("relation_type") == "implements"
            ]
            
            # Build rule entry
            rule_entry = {
                "doc_id": doc_id,
                "title": metadata.get("title", ""),
                "notification_year": metadata.get("year"),
                "doc_type": "rule",
                "total_rules": len(rules),
                "rules": rules,
                "implements": [rel.get("target_reference") for rel in implementing_relations],
                "total_chunks": len(doc_chunks_list),
                "chunk_ids": [chunk.get("chunk_id") for chunk in doc_chunks_list],
                "temporal_info": metadata.get("temporal_info", {}),
                "file_path": metadata.get("file_path"),
                "processing_date": datetime.now().isoformat()
            }
            
            rules_db[doc_id] = rule_entry
            self.legal_stats["rules_processed"] += 1
        
        return rules_db
    
    def extract_rule_hierarchy(self, chunks: List[Dict]) -> Dict:
        """Extract rule hierarchy similar to section hierarchy."""
        rules = {}
        
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id")
            
            # Find all rules mentioned in this chunk
            for pattern in self.rule_patterns:
                for match in pattern.finditer(text):
                    rule_num = match.group(1)
                    sub_rule = match.group(2) if match.lastindex >= 2 else None
                    
                    # Build rule identifier
                    rule_id = rule_num
                    if sub_rule:
                        rule_id += f"({sub_rule})"
                    
                    # Store rule information
                    if rule_id not in rules:
                        rules[rule_id] = {
                            "rule_number": rule_num,
                            "sub_rule": sub_rule,
                            "full_id": rule_id,
                            "chunk_ids": [],
                            "text_snippets": []
                        }
                    
                    # Add chunk reference
                    if chunk_id not in rules[rule_id]["chunk_ids"]:
                        rules[rule_id]["chunk_ids"].append(chunk_id)
                    
                    # Extract surrounding text
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 200)
                    snippet = text[start:end].strip()
                    
                    if snippet not in rules[rule_id]["text_snippets"]:
                        rules[rule_id]["text_snippets"].append(snippet)
        
        return rules
    
    def _identify_key_sections(self, sections: Dict) -> List[str]:
        """Identify key sections that are most referenced or important."""
        # Simple heuristic: sections with most chunks or longest text
        section_scores = {}
        
        for section_id, section_data in sections.items():
            score = 0
            score += len(section_data.get("chunk_ids", [])) * 2  # Number of chunks
            score += len(section_data.get("text_snippets", []))  # Number of snippets
            score += len(section_data.get("cross_references", []))  # References
            
            section_scores[section_id] = score
        
        # Return top 10 sections
        sorted_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        return [section_id for section_id, score in sorted_sections[:10]]
    
    def _analyze_section_types(self, sections: Dict) -> Dict[str, int]:
        """Analyze types of sections (main, subsection, clause)."""
        counts = {"main_sections": 0, "subsections": 0, "clauses": 0}
        
        for section_id, section_data in sections.items():
            if section_data.get("clause"):
                counts["clauses"] += 1
            elif section_data.get("subsection"):
                counts["subsections"] += 1
            else:
                counts["main_sections"] += 1
        
        return counts
    
    def build_cross_reference_index(self, acts_db: Dict, rules_db: Dict) -> Dict:
        """
        Build comprehensive cross-reference index across all legal documents.
        
        Args:
            acts_db: Acts database
            rules_db: Rules database
            
        Returns:
            Cross-reference index
        """
        cross_ref_index = defaultdict(lambda: {"cited_by": [], "cites": []})
        
        # Process acts
        for doc_id, act_data in acts_db.items():
            sections = act_data.get("sections", {})
            cross_refs = act_data.get("cross_references", {})
            
            for section_id in sections.keys():
                full_ref = f"{doc_id}#{section_id}"
                
                # What this section cites
                for chunk_id, refs in cross_refs.items():
                    for ref in refs:
                        cross_ref_index[full_ref]["cites"].append(ref)
                        cross_ref_index[ref]["cited_by"].append(full_ref)
        
        # Process rules
        for doc_id, rule_data in rules_db.items():
            implements = rule_data.get("implements", [])
            
            for implementation in implements:
                cross_ref_index[doc_id]["implements"] = implements
                if implementation not in cross_ref_index:
                    cross_ref_index[implementation] = {"cited_by": [], "cites": []}
                cross_ref_index[implementation]["implemented_by"] = cross_ref_index[implementation].get("implemented_by", []) + [doc_id]
        
        return dict(cross_ref_index)
    
    def build_database(self) -> Dict:
        """
        Build complete legal database.
        
        Returns:
            Complete legal database structure
        """
        # Load processed data
        chunks = self.load_processed_chunks()
        relations = self.load_relations()
        
        if not chunks:
            logger.error("No chunks loaded for legal database building")
            return {}
        
        # Filter for legal documents
        legal_chunks = self.filter_chunks_by_doc_type(chunks, ["act", "rule"])
        
        if not legal_chunks:
            logger.warning("No legal document chunks found")
            return {}
        
        # Separate acts and rules
        act_chunks = self.filter_chunks_by_doc_type(legal_chunks, ["act"])
        rule_chunks = self.filter_chunks_by_doc_type(legal_chunks, ["rule"])
        
        # Build individual databases
        acts_db = self.build_acts_database(act_chunks, relations)
        rules_db = self.build_rules_database(rule_chunks, relations)
        
        # Build cross-reference index
        cross_ref_index = self.build_cross_reference_index(acts_db, rules_db)
        
        # Create comprehensive legal database
        legal_database = {
            "metadata": {
                "database_type": "legal",
                "creation_date": datetime.now().isoformat(),
                "builder_version": "1.0.0",
                "total_documents": len(acts_db) + len(rules_db),
                "statistics": self.legal_stats
            },
            "acts": acts_db,
            "rules": rules_db,
            "cross_reference_index": cross_ref_index,
            "legal_hierarchy": self._build_legal_hierarchy(acts_db, rules_db),
            "amendment_timeline": self._build_amendment_timeline(acts_db, rules_db)
        }
        
        # Update stats
        self.stats["database_entries_created"] = len(acts_db) + len(rules_db)
        
        return legal_database
    
    def _build_legal_hierarchy(self, acts_db: Dict, rules_db: Dict) -> Dict:
        """Build hierarchical view of legal framework."""
        hierarchy = {
            "constitution": [],  # Articles
            "acts": list(acts_db.keys()),
            "rules": []
        }
        
        # Group rules by the acts they implement
        for rule_id, rule_data in rules_db.items():
            implements = rule_data.get("implements", [])
            hierarchy["rules"].append({
                "rule_id": rule_id,
                "implements": implements,
                "title": rule_data.get("title", "")
            })
        
        return hierarchy
    
    def _build_amendment_timeline(self, acts_db: Dict, rules_db: Dict) -> List[Dict]:
        """Build chronological timeline of amendments."""
        amendments = []
        
        # Collect amendments from acts
        for doc_id, act_data in acts_db.items():
            for amendment in act_data.get("amendments", []):
                amendments.append({
                    **amendment,
                    "document_id": doc_id,
                    "document_type": "act"
                })
        
        # Sort by year
        amendments.sort(key=lambda x: x.get("year", 0))
        
        return amendments


# Example usage and testing
if __name__ == "__main__":
    # Test the legal database builder
    builder = LegalDatabaseBuilder()
    
    print("Testing Legal Database Builder...")
    
    # This would normally be called as part of the process_and_save method
    # For testing, we can create mock data
    
    # Test section extraction patterns
    test_text = """
    Section 12(1)(c) of the RTE Act mandates that private schools shall reserve
    25% of their seats. As per Section 6, all children have right to education.
    Rule 4(2) provides implementation guidelines.
    """
    
    print("Testing pattern extraction:")
    print(f"Test text: {test_text}")
    
    # Test section patterns
    for pattern in builder.section_patterns:
        matches = pattern.findall(test_text)
        if matches:
            print(f"Section matches: {matches}")
    
    # Test rule patterns  
    for pattern in builder.rule_patterns:
        matches = pattern.findall(test_text)
        if matches:
            print(f"Rule matches: {matches}")
    
    print("\nLegal Database Builder ready for processing!")