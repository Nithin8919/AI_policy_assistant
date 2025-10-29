"""
Topic Matcher for education policy documents.

Matches chunks to pre-defined bridge topics during ingestion, enabling automatic
bridge table population. Uses keyword overlap, entity matching, and semantic 
similarity to determine topic relevance.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter
import math
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TopicMatcher:
    """
    Match document chunks to bridge topics for automatic topic population.
    
    Uses multiple matching strategies:
    1. Exact keyword matching
    2. Synonym/variation matching  
    3. Entity overlap (schemes, districts, metrics)
    4. Contextual pattern matching
    5. TF-IDF-like scoring for relevance
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize topic matcher with bridge topics and matching rules.
        
        Args:
            data_dir: Path to data directory (defaults to project data/)
        """
        if data_dir is None:
            data_dir = project_root / "data"
        
        self.data_dir = Path(data_dir)
        self.knowledge_graph_dir = self.data_dir / "knowledge_graph"
        
        # Load bridge topics
        self.bridge_topics = self._load_bridge_topics()
        
        # Build matching indices
        self._build_keyword_index()
        self._build_entity_index()
        self._build_pattern_index()
        
        # Scoring thresholds
        self.min_score_threshold = 1.5
        self.high_confidence_threshold = 3.0
        
        logger.info(f"TopicMatcher initialized with {len(self.bridge_topics)} bridge topics")
    
    def _load_bridge_topics(self) -> Dict:
        """Load bridge topics from seed file."""
        try:
            with open(self.knowledge_graph_dir / "seed_bridge_topics.json", 'r') as f:
                data = json.load(f)
                return data.get("topics", {})
        except FileNotFoundError:
            logger.warning("Bridge topics file not found. Using empty topics.")
            return {}
    
    def _build_keyword_index(self):
        """Build inverted index of keywords to topics for fast matching."""
        self.keyword_to_topics = defaultdict(list)
        
        for topic_id, topic_data in self.bridge_topics.items():
            keywords = topic_data.get("keywords", [])
            
            for keyword in keywords:
                # Index exact keyword
                self.keyword_to_topics[keyword.lower()].append(topic_id)
                
                # Index individual words from multi-word keywords
                words = keyword.lower().split()
                if len(words) > 1:
                    for word in words:
                        if len(word) > 2:  # Skip very short words
                            self.keyword_to_topics[word].append(topic_id)
    
    def _build_entity_index(self):
        """Build index of entities (schemes, metrics) to topics."""
        self.scheme_to_topics = defaultdict(list)
        self.metric_to_topics = defaultdict(list)
        
        for topic_id, topic_data in self.bridge_topics.items():
            # Index scheme names
            schemes = topic_data.get("scheme_names", [])
            for scheme in schemes:
                self.scheme_to_topics[scheme.lower()].append(topic_id)
            
            # Index metric codes
            metric_codes = topic_data.get("metric_codes", [])
            for metric in metric_codes:
                self.metric_to_topics[metric.lower()].append(topic_id)
    
    def _build_pattern_index(self):
        """Build patterns for specific topic matching."""
        # Special patterns for common topics
        self.topic_patterns = {
            "rte_section_12_1_c": [
                re.compile(r'\bsection\s+12\s*\(\s*1\s*\)\s*\(\s*c\s*\)', re.IGNORECASE),
                re.compile(r'\b25\s*%\s*reservation', re.IGNORECASE),
                re.compile(r'\beconomically\s+weaker\s+section', re.IGNORECASE),
                re.compile(r'\bdisadvantaged\s+group', re.IGNORECASE)
            ],
            "ptr_norms": [
                re.compile(r'\bptr\b', re.IGNORECASE),
                re.compile(r'\bpupil[- ]teacher\s+ratio', re.IGNORECASE),
                re.compile(r'\bteacher[- ]pupil\s+ratio', re.IGNORECASE),
                re.compile(r'\b\d+:\d+\b')  # Ratio patterns like 30:1
            ],
            "nadu_nedu_programme": [
                re.compile(r'\bnadu[- ]nedu\b', re.IGNORECASE),
                re.compile(r'\binfrastructure\s+development', re.IGNORECASE),
                re.compile(r'\bschool\s+building', re.IGNORECASE)
            ],
            "jagananna_amma_vodi": [
                re.compile(r'\bjagananna\s+amma\s+vodi\b', re.IGNORECASE),
                re.compile(r'\bamma\s+vodi\b', re.IGNORECASE),
                re.compile(r'\bfee\s+reimbursement\b', re.IGNORECASE),
                re.compile(r'\bmother[\'\"]?s\s+lap\b', re.IGNORECASE)
            ]
        }
    
    def calculate_keyword_score(self, text: str, topic_id: str) -> float:
        """
        Calculate keyword match score for a topic.
        
        Args:
            text: Input text to analyze
            topic_id: Bridge topic ID
            
        Returns:
            Keyword match score
        """
        if topic_id not in self.bridge_topics:
            return 0.0
        
        topic_keywords = self.bridge_topics[topic_id].get("keywords", [])
        if not topic_keywords:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        for keyword in topic_keywords:
            keyword_lower = keyword.lower()
            
            # Exact phrase match (higher weight)
            if keyword_lower in text_lower:
                score += 2.0
            else:
                # Partial word matches (lower weight)
                words = keyword_lower.split()
                word_matches = sum(1 for word in words if word in text_lower)
                if word_matches > 0:
                    score += (word_matches / len(words)) * 1.0
        
        return score
    
    def calculate_entity_score(self, entities: Dict, topic_id: str) -> float:
        """
        Calculate entity overlap score for a topic.
        
        Args:
            entities: Extracted entities from text
            topic_id: Bridge topic ID
            
        Returns:
            Entity overlap score
        """
        if topic_id not in self.bridge_topics:
            return 0.0
        
        topic_data = self.bridge_topics[topic_id]
        score = 0.0
        
        # Check scheme matches
        chunk_schemes = [s.lower() for s in entities.get("schemes", [])]
        topic_schemes = [s.lower() for s in topic_data.get("scheme_names", [])]
        
        for scheme in chunk_schemes:
            if scheme in topic_schemes:
                score += 3.0  # High weight for scheme matches
        
        # Check metric matches
        chunk_metrics = [m.lower() for m in entities.get("metrics", [])]
        topic_metrics = [m.lower() for m in topic_data.get("metric_codes", [])]
        
        for metric in chunk_metrics:
            if metric in topic_metrics:
                score += 2.0
        
        # Check district matches (if topic is district-specific)
        chunk_districts = [d.lower() for d in entities.get("districts", [])]
        topic_districts = [d.lower() for d in topic_data.get("districts", [])]
        
        if topic_districts:  # Only score if topic has specific districts
            for district in chunk_districts:
                if district in topic_districts:
                    score += 1.5
        
        # Check keyword matches from entities
        chunk_keywords = [k.lower() for k in entities.get("keywords", [])]
        topic_keywords = [k.lower() for k in topic_data.get("keywords", [])]
        
        for keyword in chunk_keywords:
            if keyword in topic_keywords:
                score += 1.0
        
        return score
    
    def calculate_pattern_score(self, text: str, topic_id: str) -> float:
        """
        Calculate pattern-based score for specific topics.
        
        Args:
            text: Input text to analyze
            topic_id: Bridge topic ID
            
        Returns:
            Pattern match score
        """
        if topic_id not in self.topic_patterns:
            return 0.0
        
        patterns = self.topic_patterns[topic_id]
        score = 0.0
        
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                score += len(matches) * 1.5  # Weight by number of matches
        
        return score
    
    def calculate_legal_reference_score(self, entities: Dict, topic_id: str) -> float:
        """
        Calculate score based on legal references that relate to the topic.
        
        Args:
            entities: Extracted entities from text
            topic_id: Bridge topic ID
            
        Returns:
            Legal reference score
        """
        legal_refs = entities.get("legal_refs", [])
        if not legal_refs:
            return 0.0
        
        score = 0.0
        
        # Topic-specific legal reference scoring
        if topic_id == "rte_section_12_1_c":
            for ref in legal_refs:
                if (ref.get("type") == "section" and 
                    ref.get("number") == "12" and
                    ref.get("act", "").lower().find("rte") != -1):
                    score += 3.0
        
        elif topic_id.startswith("rte_"):
            for ref in legal_refs:
                if ref.get("act", "").lower().find("rte") != -1:
                    score += 2.0
        
        elif topic_id in ["school_management_committees", "rte_compliance"]:
            for ref in legal_refs:
                if (ref.get("type") == "section" and 
                    ref.get("number") == "21"):
                    score += 2.5
        
        # General RTE Act references
        for ref in legal_refs:
            if ref.get("act", "").lower().find("rte") != -1:
                score += 1.0
        
        return score
    
    def match_chunk_to_topics(
        self, 
        text: str, 
        entities: Dict, 
        chunk_id: str
    ) -> List[Dict]:
        """
        Match a single chunk to relevant bridge topics.
        
        Args:
            text: Chunk text
            entities: Extracted entities from the chunk
            chunk_id: Chunk identifier
            
        Returns:
            List of topic matches with scores
        """
        matches = []
        
        for topic_id in self.bridge_topics:
            # Calculate different score components
            keyword_score = self.calculate_keyword_score(text, topic_id)
            entity_score = self.calculate_entity_score(entities, topic_id)
            pattern_score = self.calculate_pattern_score(text, topic_id)
            legal_score = self.calculate_legal_reference_score(entities, topic_id)
            
            # Combine scores with weights
            total_score = (
                keyword_score * 1.0 +
                entity_score * 1.2 +  # Entities slightly more important
                pattern_score * 1.3 +  # Patterns are quite reliable
                legal_score * 1.1
            )
            
            # Apply threshold
            if total_score >= self.min_score_threshold:
                confidence = "high" if total_score >= self.high_confidence_threshold else "medium"
                
                matches.append({
                    "topic_id": topic_id,
                    "topic_name": self.bridge_topics[topic_id].get("topic_name", topic_id),
                    "score": round(total_score, 2),
                    "confidence": confidence,
                    "score_breakdown": {
                        "keywords": round(keyword_score, 2),
                        "entities": round(entity_score, 2),
                        "patterns": round(pattern_score, 2),
                        "legal_refs": round(legal_score, 2)
                    }
                })
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top 5 matches to avoid noise
        matches = matches[:5]
        
        logger.debug(f"Chunk {chunk_id} matched to {len(matches)} topics")
        
        return matches
    
    def match_document_to_topics(
        self, 
        chunks: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Match all chunks in a document to topics and aggregate results.
        
        Args:
            chunks: List of chunk dictionaries with entities
            
        Returns:
            Dictionary mapping topic_ids to lists of matching chunk_ids
        """
        topic_to_chunks = defaultdict(list)
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            entities = chunk.get("entities", {})
            
            matches = self.match_chunk_to_topics(text, entities, chunk_id)
            
            # Store chunk-to-topic mappings
            chunk["bridge_topics"] = matches
            
            # Build reverse mapping for topic aggregation
            for match in matches:
                topic_id = match["topic_id"]
                topic_to_chunks[topic_id].append({
                    "chunk_id": chunk_id,
                    "score": match["score"],
                    "confidence": match["confidence"]
                })
        
        return dict(topic_to_chunks)
    
    def find_related_topics(self, topic_id: str) -> List[str]:
        """
        Find topics related to the given topic.
        
        Args:
            topic_id: Bridge topic ID
            
        Returns:
            List of related topic IDs
        """
        if topic_id not in self.bridge_topics:
            return []
        
        topic_data = self.bridge_topics[topic_id]
        return topic_data.get("related_topics", [])
    
    def get_topic_statistics(self, topic_matches: Dict) -> Dict:
        """
        Generate statistics about topic matching results.
        
        Args:
            topic_matches: Results from match_document_to_topics
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_topics_matched": len(topic_matches),
            "total_topic_bridge_coverage": 0,
            "top_topics": [],
            "topic_chunk_counts": {},
            "high_confidence_matches": 0,
            "medium_confidence_matches": 0
        }
        
        # Calculate coverage (percentage of bridge topics that had matches)
        total_bridge_topics = len(self.bridge_topics)
        if total_bridge_topics > 0:
            stats["total_topic_bridge_coverage"] = round(
                (len(topic_matches) / total_bridge_topics) * 100, 1
            )
        
        # Count chunks per topic and confidence levels
        topic_chunk_counts = {}
        confidence_counts = {"high": 0, "medium": 0}
        
        for topic_id, chunks in topic_matches.items():
            topic_chunk_counts[topic_id] = len(chunks)
            
            for chunk in chunks:
                confidence_counts[chunk["confidence"]] += 1
        
        stats["topic_chunk_counts"] = topic_chunk_counts
        stats["high_confidence_matches"] = confidence_counts["high"]
        stats["medium_confidence_matches"] = confidence_counts["medium"]
        
        # Top topics by chunk count
        top_topics = sorted(
            topic_chunk_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        stats["top_topics"] = [
            {
                "topic_id": topic_id,
                "topic_name": self.bridge_topics.get(topic_id, {}).get("topic_name", topic_id),
                "chunk_count": count
            }
            for topic_id, count in top_topics
        ]
        
        return stats
    
    def export_topic_matches(self, topic_matches: Dict, output_file: str):
        """
        Export topic matching results to file.
        
        Args:
            topic_matches: Results from match_document_to_topics
            output_file: Path to output file
        """
        export_data = {
            "metadata": {
                "total_topics": len(self.bridge_topics),
                "matched_topics": len(topic_matches),
                "generation_timestamp": "2025-10-29"  # You could use datetime.now()
            },
            "topic_matches": topic_matches,
            "statistics": self.get_topic_statistics(topic_matches)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported topic matches to {output_file}")


# Example usage and testing
if __name__ == "__main__":
    # Test the topic matcher
    matcher = TopicMatcher()
    
    # Test entities (would come from EntityExtractor)
    test_entities = {
        "legal_refs": [
            {"type": "section", "number": "12", "subsection": "1", "clause": "c", "act": "RTE Act"}
        ],
        "schemes": ["Nadu-Nedu"],
        "districts": ["Visakhapatnam"],
        "metrics": ["PTR"],
        "keywords": ["25% reservation", "private schools", "infrastructure"]
    }
    
    # Test text
    test_text = """
    Section 12(1)(c) of the RTE Act mandates 25% reservation for EWS and DG children 
    in private unaided schools. The Nadu-Nedu programme focuses on infrastructure 
    development in government schools across Visakhapatnam district.
    """
    
    matches = matcher.match_chunk_to_topics(test_text, test_entities, "test_chunk_001")
    
    print(f"Found {len(matches)} topic matches:")
    for match in matches:
        print(f"- {match['topic_name']} (score: {match['score']}, confidence: {match['confidence']})")
        print(f"  Score breakdown: {match['score_breakdown']}")
        print()