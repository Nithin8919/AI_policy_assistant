"""Track GO supersession chains for maintaining current policy status"""
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SupersessionEntry:
    """Represents a supersession relationship"""
    superseding_doc_id: str  # The new GO that supersedes
    superseded_doc_id: str   # The old GO being superseded
    supersession_type: str   # "full", "partial", "modification"
    date_superseded: Optional[str]
    evidence: str           # Text evidence for the supersession
    confidence: float
    status: str            # "active", "superseded", "partially_superseded"

@dataclass
class SupersessionChain:
    """Represents a chain of supersessions for a topic/policy"""
    chain_id: str
    topic: str
    current_doc_id: str     # Currently active GO
    chain_entries: List[SupersessionEntry]
    last_updated: str

class SupersessionTracker:
    """Track Government Order supersession chains for policy compliance"""
    
    def __init__(self):
        """Initialize the supersession tracker"""
        self.supersession_chains = {}
        self.document_status = {}  # doc_id -> status mapping
        
        # Compile supersession patterns
        self._compile_supersession_patterns()
    
    def _compile_supersession_patterns(self):
        """Compile regex patterns for detecting supersessions"""
        
        # Full supersession patterns
        self.full_supersession_patterns = [
            r'[Ii]n\s+supersession\s+of\s+G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)(?:/(\d{4}))?(?:\s*dated?\s*(\d{1,2}[.-]\d{1,2}[.-]\d{2,4}))?',
            r'[Tt]his\s+order\s+supersedes?\s+G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)(?:/(\d{4}))?',
            r'[Ss]upersedes?\s+G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)(?:/(\d{4}))?',
            r'[Ii]n\s+place\s+of\s+G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)(?:/(\d{4}))?'
        ]
        
        # Partial supersession/modification patterns
        self.partial_supersession_patterns = [
            r'[Ii]n\s+(?:partial\s+)?modification\s+of\s+G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)(?:/(\d{4}))?',
            r'[Mm]odifies?\s+G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)(?:/(\d{4}))?',
            r'[Aa]mends?\s+G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)(?:/(\d{4}))?',
            r'[Ii]n\s+amendment\s+(?:to|of)\s+G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)(?:/(\d{4}))?'
        ]
        
        # Generic supersession patterns (lower confidence)
        self.generic_supersession_patterns = [
            r'[Ss]upersedes?\s+(?:the\s+)?(?:earlier\s+)?(?:order|GO)',
            r'[Ii]n\s+supersession\s+of\s+(?:earlier\s+)?(?:order|GO)',
            r'[Rr]eplaces?\s+(?:the\s+)?(?:earlier\s+)?(?:order|GO)'
        ]
        
        # Compile patterns
        self.compiled_full = [re.compile(p, re.IGNORECASE) for p in self.full_supersession_patterns]
        self.compiled_partial = [re.compile(p, re.IGNORECASE) for p in self.partial_supersession_patterns]
        self.compiled_generic = [re.compile(p, re.IGNORECASE) for p in self.generic_supersession_patterns]
    
    def extract_supersession_from_text(self, text: str, doc_id: str) -> List[SupersessionEntry]:
        """Extract supersession relationships from document text"""
        entries = []
        
        # Check for full supersessions
        for pattern in self.compiled_full:
            matches = pattern.finditer(text)
            for match in matches:
                # Extract GO number and year if present
                go_number = match.group(1) if match.groups() else None
                go_year = match.group(2) if len(match.groups()) >= 2 and match.group(2) else None
                go_date = match.group(3) if len(match.groups()) >= 3 and match.group(3) else None
                
                if go_number:
                    superseded_id = self._construct_go_id(go_number, go_year)
                    
                    entry = SupersessionEntry(
                        superseding_doc_id=doc_id,
                        superseded_doc_id=superseded_id,
                        supersession_type="full",
                        date_superseded=go_date,
                        evidence=self._extract_context(text, match.start(), match.end()),
                        confidence=0.95,
                        status="superseded"
                    )
                    entries.append(entry)
        
        # Check for partial supersessions/modifications
        for pattern in self.compiled_partial:
            matches = pattern.finditer(text)
            for match in matches:
                go_number = match.group(1) if match.groups() else None
                go_year = match.group(2) if len(match.groups()) >= 2 and match.group(2) else None
                
                if go_number:
                    superseded_id = self._construct_go_id(go_number, go_year)
                    
                    entry = SupersessionEntry(
                        superseding_doc_id=doc_id,
                        superseded_doc_id=superseded_id,
                        supersession_type="partial",
                        date_superseded=None,
                        evidence=self._extract_context(text, match.start(), match.end()),
                        confidence=0.9,
                        status="partially_superseded"
                    )
                    entries.append(entry)
        
        # Check for generic supersessions (lower confidence)
        for pattern in self.compiled_generic:
            matches = pattern.finditer(text)
            for match in matches:
                entry = SupersessionEntry(
                    superseding_doc_id=doc_id,
                    superseded_doc_id="unknown_go",  # Cannot determine specific GO
                    supersession_type="full",
                    date_superseded=None,
                    evidence=self._extract_context(text, match.start(), match.end()),
                    confidence=0.7,
                    status="superseded"
                )
                entries.append(entry)
        
        return entries
    
    def build_supersession_chains(self, all_supersessions: List[SupersessionEntry]) -> Dict[str, SupersessionChain]:
        """Build supersession chains from individual supersession entries"""
        chains = {}
        
        # Group supersessions by topic (inferred from GO content/similarity)
        topic_groups = self._group_supersessions_by_topic(all_supersessions)
        
        # Build chains for each topic
        for topic, supersessions in topic_groups.items():
            chain_id = f"chain_{topic.lower().replace(' ', '_')}"
            
            # Sort supersessions by date (if available) or document order
            sorted_supersessions = self._sort_supersessions_chronologically(supersessions)
            
            # Find the current active document
            current_doc = self._find_current_active_document(sorted_supersessions)
            
            chain = SupersessionChain(
                chain_id=chain_id,
                topic=topic,
                current_doc_id=current_doc,
                chain_entries=sorted_supersessions,
                last_updated=datetime.now().isoformat()
            )
            
            chains[chain_id] = chain
        
        return chains
    
    def update_document_status(self, supersession_chains: Dict[str, SupersessionChain]):
        """Update document status based on supersession chains"""
        self.document_status = {}
        
        for chain in supersession_chains.values():
            # Mark current document as active
            self.document_status[chain.current_doc_id] = "active"
            
            # Mark superseded documents
            for entry in chain.chain_entries:
                if entry.supersession_type == "full":
                    self.document_status[entry.superseded_doc_id] = "superseded"
                elif entry.supersession_type == "partial":
                    self.document_status[entry.superseded_doc_id] = "partially_superseded"
    
    def get_current_go_for_topic(self, topic: str) -> Optional[str]:
        """Get the currently active GO for a specific topic"""
        topic_lower = topic.lower()
        
        for chain in self.supersession_chains.values():
            if topic_lower in chain.topic.lower():
                return chain.current_doc_id
        
        return None
    
    def get_supersession_chain_for_doc(self, doc_id: str) -> Optional[SupersessionChain]:
        """Get the supersession chain that contains a specific document"""
        for chain in self.supersession_chains.values():
            # Check if doc is current
            if chain.current_doc_id == doc_id:
                return chain
            
            # Check if doc is in chain entries
            for entry in chain.chain_entries:
                if entry.superseding_doc_id == doc_id or entry.superseded_doc_id == doc_id:
                    return chain
        
        return None
    
    def get_document_status(self, doc_id: str) -> str:
        """Get the status of a specific document"""
        return self.document_status.get(doc_id, "unknown")
    
    def is_document_active(self, doc_id: str) -> bool:
        """Check if a document is currently active (not superseded)"""
        return self.get_document_status(doc_id) == "active"
    
    def get_superseding_document(self, doc_id: str) -> Optional[str]:
        """Get the document that supersedes the given document"""
        for chain in self.supersession_chains.values():
            for entry in chain.chain_entries:
                if entry.superseded_doc_id == doc_id:
                    return entry.superseding_doc_id
        
        return None
    
    def get_superseded_documents(self, doc_id: str) -> List[str]:
        """Get all documents that are superseded by the given document"""
        superseded = []
        
        for chain in self.supersession_chains.values():
            for entry in chain.chain_entries:
                if entry.superseding_doc_id == doc_id:
                    superseded.append(entry.superseded_doc_id)
        
        return superseded
    
    def validate_supersession_chains(self) -> Dict[str, Any]:
        """Validate supersession chains and identify potential issues"""
        issues = {
            "circular_references": [],
            "orphaned_documents": [],
            "conflicting_supersessions": [],
            "low_confidence_entries": []
        }
        
        # Check for circular references
        for chain in self.supersession_chains.values():
            visited = set()
            for entry in chain.chain_entries:
                if entry.superseded_doc_id in visited:
                    issues["circular_references"].append({
                        "chain_id": chain.chain_id,
                        "document": entry.superseded_doc_id
                    })
                visited.add(entry.superseding_doc_id)
        
        # Check for low confidence entries
        for chain in self.supersession_chains.values():
            for entry in chain.chain_entries:
                if entry.confidence < 0.8:
                    issues["low_confidence_entries"].append({
                        "chain_id": chain.chain_id,
                        "superseding": entry.superseding_doc_id,
                        "superseded": entry.superseded_doc_id,
                        "confidence": entry.confidence,
                        "evidence": entry.evidence
                    })
        
        return issues
    
    def save_supersession_data(self, output_file: str) -> bool:
        """Save supersession chains and document status to file"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            serializable_chains = {}
            for chain_id, chain in self.supersession_chains.items():
                serializable_chains[chain_id] = {
                    "chain_id": chain.chain_id,
                    "topic": chain.topic,
                    "current_doc_id": chain.current_doc_id,
                    "chain_entries": [asdict(entry) for entry in chain.chain_entries],
                    "last_updated": chain.last_updated
                }
            
            data = {
                "supersession_chains": serializable_chains,
                "document_status": self.document_status,
                "metadata": {
                    "total_chains": len(self.supersession_chains),
                    "total_documents": len(self.document_status),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved supersession data to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save supersession data: {e}")
            return False
    
    def load_supersession_data(self, input_file: str) -> bool:
        """Load supersession chains and document status from file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load chains
            self.supersession_chains = {}
            chains_data = data.get("supersession_chains", {})
            
            for chain_id, chain_data in chains_data.items():
                # Reconstruct SupersessionEntry objects
                chain_entries = []
                for entry_data in chain_data["chain_entries"]:
                    entry = SupersessionEntry(**entry_data)
                    chain_entries.append(entry)
                
                # Reconstruct SupersessionChain
                chain = SupersessionChain(
                    chain_id=chain_data["chain_id"],
                    topic=chain_data["topic"],
                    current_doc_id=chain_data["current_doc_id"],
                    chain_entries=chain_entries,
                    last_updated=chain_data["last_updated"]
                )
                
                self.supersession_chains[chain_id] = chain
            
            # Load document status
            self.document_status = data.get("document_status", {})
            
            logger.info(f"Loaded supersession data from {input_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load supersession data: {e}")
            return False
    
    def _construct_go_id(self, go_number: str, go_year: Optional[str] = None) -> str:
        """Construct standardized GO ID from number and year"""
        if go_year:
            return f"GO_MS_{go_number}_{go_year}"
        else:
            return f"GO_MS_{go_number}"
    
    def _extract_context(self, text: str, start: int, end: int, context_size: int = 100) -> str:
        """Extract context around the matched span"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end].strip()
    
    def _group_supersessions_by_topic(self, supersessions: List[SupersessionEntry]) -> Dict[str, List[SupersessionEntry]]:
        """Group supersessions by inferred topic"""
        # For now, use a simple grouping based on document patterns
        # This could be enhanced with more sophisticated topic modeling
        
        topic_groups = {}
        
        for supersession in supersessions:
            # Infer topic from evidence or document pattern
            topic = self._infer_topic_from_evidence(supersession.evidence)
            
            if topic not in topic_groups:
                topic_groups[topic] = []
            
            topic_groups[topic].append(supersession)
        
        return topic_groups
    
    def _infer_topic_from_evidence(self, evidence: str) -> str:
        """Infer topic from supersession evidence text"""
        evidence_lower = evidence.lower()
        
        # Common education policy topics
        if any(term in evidence_lower for term in ["teacher", "recruitment", "transfer", "posting"]):
            return "teacher_management"
        elif any(term in evidence_lower for term in ["midday", "meal", "feeding", "nutrition"]):
            return "midday_meal"
        elif any(term in evidence_lower for term in ["infrastructure", "building", "construction", "nadu"]):
            return "infrastructure"
        elif any(term in evidence_lower for term in ["admission", "enrollment", "reservation"]):
            return "admissions"
        elif any(term in evidence_lower for term in ["examination", "assessment", "evaluation"]):
            return "examinations"
        elif any(term in evidence_lower for term in ["scholarship", "financial", "assistance", "amma"]):
            return "scholarships"
        elif any(term in evidence_lower for term in ["rte", "right to education", "compliance"]):
            return "rte_compliance"
        else:
            return "general_education"
    
    def _sort_supersessions_chronologically(self, supersessions: List[SupersessionEntry]) -> List[SupersessionEntry]:
        """Sort supersessions chronologically"""
        # Sort by date if available, otherwise by document ID pattern
        def sort_key(entry):
            if entry.date_superseded:
                try:
                    # Try to parse date for sorting
                    date_str = entry.date_superseded.replace('-', '/').replace('.', '/')
                    return datetime.strptime(date_str, '%d/%m/%Y').timestamp()
                except:
                    pass
            
            # Fallback to document ID pattern
            return entry.superseding_doc_id
        
        return sorted(supersessions, key=sort_key)
    
    def _find_current_active_document(self, supersessions: List[SupersessionEntry]) -> str:
        """Find the currently active document in a supersession chain"""
        if not supersessions:
            return "unknown"
        
        # The document that supersedes others but is not superseded itself
        superseding_docs = {entry.superseding_doc_id for entry in supersessions}
        superseded_docs = {entry.superseded_doc_id for entry in supersessions}
        
        # Current document supersedes others but is not superseded
        current_candidates = superseding_docs - superseded_docs
        
        if current_candidates:
            # If multiple candidates, pick the most recent
            return max(current_candidates)
        elif superseding_docs:
            # Fallback to the last superseding document
            return max(superseding_docs)
        else:
            return "unknown"
    
    def get_supersession_stats(self) -> Dict[str, Any]:
        """Get statistics about supersession tracking"""
        total_chains = len(self.supersession_chains)
        total_documents = len(self.document_status)
        
        status_distribution = {}
        for status in self.document_status.values():
            status_distribution[status] = status_distribution.get(status, 0) + 1
        
        # Chain statistics
        chain_lengths = [len(chain.chain_entries) for chain in self.supersession_chains.values()]
        avg_chain_length = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0
        
        # Confidence statistics
        all_confidences = []
        for chain in self.supersession_chains.values():
            for entry in chain.chain_entries:
                all_confidences.append(entry.confidence)
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        return {
            "total_chains": total_chains,
            "total_documents": total_documents,
            "status_distribution": status_distribution,
            "average_chain_length": avg_chain_length,
            "average_confidence": avg_confidence,
            "chain_topics": [chain.topic for chain in self.supersession_chains.values()]
        }


