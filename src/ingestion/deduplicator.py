"""
Document Deduplicator for education policy documents.

Detects duplicate or near-duplicate documents to maintain corpus quality.
Handles exact duplicates, OCR variations, and amended versions.
"""

import hashlib
import re
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentDeduplicator:
    """
    Detect and handle duplicate documents in the corpus.
    
    Strategies:
    1. Exact duplicates - identical content hash
    2. Near duplicates - high text similarity (OCR variations)
    3. Amended versions - similar content with amendment patterns
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Minimum similarity to consider documents as duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.content_hashes = {}
        self.title_index = defaultdict(list)
        self.text_fingerprints = {}
        
        logger.info(f"DocumentDeduplicator initialized with threshold {similarity_threshold}")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?]', '', text)
        
        return text.strip()
    
    def calculate_content_hash(self, text: str) -> str:
        """Calculate hash for exact duplicate detection."""
        normalized = self.normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def calculate_text_fingerprint(self, text: str, shingle_size: int = 5) -> Set[str]:
        """Calculate text fingerprint using character shingles."""
        normalized = self.normalize_text(text)
        if len(normalized) < shingle_size:
            return {normalized}
        
        shingles = set()
        for i in range(len(normalized) - shingle_size + 1):
            shingle = normalized[i:i + shingle_size]
            shingles.add(shingle)
        
        return shingles
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, 
                                self.normalize_text(text1), 
                                self.normalize_text(text2))
        return matcher.ratio()
    
    def calculate_jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def is_amended_version(self, text1: str, text2: str, title1: str = "", title2: str = "") -> bool:
        """
        Check if documents are different versions (original vs amended).
        
        Args:
            text1, text2: Document texts
            title1, title2: Document titles
            
        Returns:
            True if documents appear to be different versions
        """
        # Check for amendment indicators in titles
        amendment_patterns = [
            r'\bamendment\b',
            r'\bamended\b',
            r'\bmodification\b',
            r'\brevised\b',
            r'\bupdated\b'
        ]
        
        title_combined = f"{title1} {title2}".lower()
        for pattern in amendment_patterns:
            if re.search(pattern, title_combined):
                return True
        
        # Check for supersession patterns in text
        supersession_patterns = [
            r'\bin supersession of\b',
            r'\bamends?\b',
            r'\bmodifies?\b',
            r'\bsubstitutes?\b'
        ]
        
        text_combined = f"{text1} {text2}".lower()
        for pattern in supersession_patterns:
            if re.search(pattern, text_combined):
                return True
        
        return False
    
    def find_exact_duplicates(self, documents: List[Dict]) -> Dict:
        """
        Find documents with identical content.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary mapping content hashes to lists of duplicate documents
        """
        hash_to_docs = defaultdict(list)
        
        for doc in documents:
            text = doc.get('text', '')
            content_hash = self.calculate_content_hash(text)
            hash_to_docs[content_hash].append(doc)
        
        # Filter to only return groups with duplicates
        duplicates = {h: docs for h, docs in hash_to_docs.items() if len(docs) > 1}
        
        logger.info(f"Found {len(duplicates)} groups of exact duplicates")
        return duplicates
    
    def find_near_duplicates(self, documents: List[Dict]) -> List[Tuple[Dict, Dict, float]]:
        """
        Find near-duplicate documents using text similarity.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of tuples (doc1, doc2, similarity_score)
        """
        near_duplicates = []
        
        # Create fingerprints for all documents
        doc_fingerprints = []
        for doc in documents:
            text = doc.get('text', '')
            fingerprint = self.calculate_text_fingerprint(text)
            doc_fingerprints.append((doc, fingerprint))
        
        # Compare all pairs
        for i, (doc1, fp1) in enumerate(doc_fingerprints):
            for j, (doc2, fp2) in enumerate(doc_fingerprints[i+1:], i+1):
                similarity = self.calculate_jaccard_similarity(fp1, fp2)
                
                if similarity >= self.similarity_threshold:
                    # Double-check with sequence matcher
                    text_similarity = self.calculate_similarity(
                        doc1.get('text', ''), 
                        doc2.get('text', '')
                    )
                    
                    if text_similarity >= self.similarity_threshold:
                        near_duplicates.append((doc1, doc2, text_similarity))
        
        logger.info(f"Found {len(near_duplicates)} near-duplicate pairs")
        return near_duplicates
    
    def find_title_duplicates(self, documents: List[Dict]) -> Dict:
        """
        Find documents with identical or very similar titles.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary mapping normalized titles to lists of documents
        """
        title_to_docs = defaultdict(list)
        
        for doc in documents:
            title = doc.get('title', doc.get('metadata', {}).get('title', ''))
            if title:
                normalized_title = self.normalize_text(title)
                title_to_docs[normalized_title].append(doc)
        
        # Filter to only return groups with duplicates
        duplicates = {t: docs for t, docs in title_to_docs.items() if len(docs) > 1}
        
        logger.info(f"Found {len(duplicates)} groups of title duplicates")
        return duplicates
    
    def resolve_duplicates(self, duplicate_groups: Dict, resolution_strategy: str = "keep_latest") -> List[Dict]:
        """
        Resolve duplicate documents by selecting which to keep.
        
        Args:
            duplicate_groups: Groups of duplicate documents
            resolution_strategy: How to resolve duplicates
            
        Returns:
            List of documents to keep
        """
        kept_documents = []
        removed_documents = []
        
        for group_id, docs in duplicate_groups.items():
            if len(docs) <= 1:
                kept_documents.extend(docs)
                continue
            
            if resolution_strategy == "keep_latest":
                # Keep document with latest date/year
                latest_doc = max(docs, key=lambda d: self._extract_year(d))
                kept_documents.append(latest_doc)
                removed_documents.extend([d for d in docs if d != latest_doc])
            
            elif resolution_strategy == "keep_longest":
                # Keep document with most content
                longest_doc = max(docs, key=lambda d: len(d.get('text', '')))
                kept_documents.append(longest_doc)
                removed_documents.extend([d for d in docs if d != longest_doc])
            
            elif resolution_strategy == "keep_first":
                # Keep first document found
                kept_documents.append(docs[0])
                removed_documents.extend(docs[1:])
            
            else:
                # Keep all (mark but don't remove)
                kept_documents.extend(docs)
        
        logger.info(f"Kept {len(kept_documents)} documents, removed {len(removed_documents)}")
        return kept_documents
    
    def _extract_year(self, document: Dict) -> int:
        """Extract year from document for date-based resolution."""
        # Try metadata first
        metadata = document.get('metadata', {})
        year = metadata.get('year')
        
        if isinstance(year, (int, float)) and 1900 <= year <= 2030:
            return int(year)
        
        # Try to extract from title or text
        text_to_search = f"{document.get('title', '')} {document.get('text', '')}"
        year_matches = re.findall(r'\b(19|20)\d{2}\b', text_to_search)
        
        if year_matches:
            years = [int(y) for y in year_matches if 1900 <= int(y) <= 2030]
            if years:
                return max(years)  # Return most recent year found
        
        return 0  # Default for unknown year
    
    def analyze_duplicates(self, documents: List[Dict]) -> Dict:
        """
        Comprehensive duplicate analysis.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Complete duplicate analysis report
        """
        logger.info(f"Analyzing {len(documents)} documents for duplicates")
        
        # Find different types of duplicates
        exact_duplicates = self.find_exact_duplicates(documents)
        near_duplicates = self.find_near_duplicates(documents)
        title_duplicates = self.find_title_duplicates(documents)
        
        # Identify amended versions
        amended_pairs = []
        for doc1, doc2, similarity in near_duplicates:
            if self.is_amended_version(
                doc1.get('text', ''), 
                doc2.get('text', ''),
                doc1.get('title', ''), 
                doc2.get('title', '')
            ):
                amended_pairs.append((doc1, doc2, similarity))
        
        # Generate statistics
        stats = {
            "total_documents": len(documents),
            "exact_duplicate_groups": len(exact_duplicates),
            "exact_duplicate_documents": sum(len(docs) for docs in exact_duplicates.values()),
            "near_duplicate_pairs": len(near_duplicates),
            "title_duplicate_groups": len(title_duplicates),
            "amended_version_pairs": len(amended_pairs),
            "duplicate_ratio": 0.0
        }
        
        # Calculate overall duplicate ratio
        all_duplicate_docs = set()
        
        # Add exact duplicates
        for docs in exact_duplicates.values():
            for doc in docs:
                all_duplicate_docs.add(doc.get('doc_id', id(doc)))
        
        # Add near duplicates
        for doc1, doc2, _ in near_duplicates:
            all_duplicate_docs.add(doc1.get('doc_id', id(doc1)))
            all_duplicate_docs.add(doc2.get('doc_id', id(doc2)))
        
        if len(documents) > 0:
            stats["duplicate_ratio"] = len(all_duplicate_docs) / len(documents)
        
        return {
            "statistics": stats,
            "exact_duplicates": exact_duplicates,
            "near_duplicates": near_duplicates,
            "title_duplicates": title_duplicates,
            "amended_versions": amended_pairs
        }
    
    def create_deduplication_report(self, analysis_result: Dict) -> str:
        """Create human-readable deduplication report."""
        stats = analysis_result["statistics"]
        
        report = f"""
Document Deduplication Report
============================

Total Documents Analyzed: {stats['total_documents']}

Exact Duplicates:
- Groups: {stats['exact_duplicate_groups']}
- Total Documents: {stats['exact_duplicate_documents']}

Near Duplicates:
- Similar Pairs: {stats['near_duplicate_pairs']}

Title Duplicates:
- Groups: {stats['title_duplicate_groups']}

Amended Versions:
- Version Pairs: {stats['amended_version_pairs']}

Overall Duplicate Ratio: {stats['duplicate_ratio']:.1%}

Recommendations:
"""
        
        if stats["exact_duplicate_groups"] > 0:
            report += "- Remove exact duplicates (keep latest version)\n"
        
        if stats["near_duplicate_pairs"] > 0:
            report += "- Review near duplicates for OCR variations\n"
        
        if stats["amended_version_pairs"] > 0:
            report += "- Mark amended versions with proper relationships\n"
        
        if stats["duplicate_ratio"] > 0.1:
            report += "- High duplicate ratio detected - thorough review recommended\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Test the deduplicator
    deduplicator = DocumentDeduplicator()
    
    # Test documents with duplicates
    test_docs = [
        {
            "doc_id": "doc1",
            "title": "RTE Act 2009",
            "text": "Section 12 of the Right to Education Act mandates 25% reservation in private schools.",
            "metadata": {"year": 2009}
        },
        {
            "doc_id": "doc2", 
            "title": "RTE Act 2009",
            "text": "Section 12 of the Right to Education Act mandates 25% reservation in private schools.",
            "metadata": {"year": 2009}
        },
        {
            "doc_id": "doc3",
            "title": "RTE Act Amendment 2012", 
            "text": "Section 12 of the Right to Education Act mandates 25% reservation in private schools. This amendment clarifies implementation.",
            "metadata": {"year": 2012}
        },
        {
            "doc_id": "doc4",
            "title": "Different Document",
            "text": "This is completely different content about education policy.",
            "metadata": {"year": 2020}
        }
    ]
    
    # Analyze duplicates
    analysis = deduplicator.analyze_duplicates(test_docs)
    
    print("Duplicate Analysis:")
    print(f"Exact duplicates: {len(analysis['exact_duplicates'])}")
    print(f"Near duplicates: {len(analysis['near_duplicates'])}")
    print(f"Amended versions: {len(analysis['amended_versions'])}")
    
    # Generate report
    report = deduplicator.create_deduplication_report(analysis)
    print("\n" + report)