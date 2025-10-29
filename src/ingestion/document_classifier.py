"""
Document Classifier for education policy documents.

Content-based classification that improves upon folder-based classification
by analyzing document structure, language patterns, and content features.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentClassifier:
    """
    Content-based document classifier for education policy documents.
    
    Classifies documents into categories based on their content structure
    and language patterns rather than just folder location.
    """
    
    def __init__(self):
        """Initialize document classifier with rules and patterns."""
        self._init_classification_rules()
        self._init_feature_extractors()
        
        # Classification thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        
        logger.info("DocumentClassifier initialized")
    
    def _init_classification_rules(self):
        """Initialize rule-based classification patterns."""
        
        # Document type indicators
        self.type_indicators = {
            "act": {
                "structure_patterns": [
                    re.compile(r'\bchapter\s+\d+\b', re.IGNORECASE),
                    re.compile(r'\bsection\s+\d+', re.IGNORECASE),
                    re.compile(r'\bpreamble\b', re.IGNORECASE),
                    re.compile(r'\benactment\b', re.IGNORECASE)
                ],
                "language_patterns": [
                    re.compile(r'\bwhereas\b', re.IGNORECASE),
                    re.compile(r'\bbe it enacted\b', re.IGNORECASE),
                    re.compile(r'\bnothing in this act\b', re.IGNORECASE),
                    re.compile(r'\bsubject to\b', re.IGNORECASE)
                ],
                "title_patterns": [
                    re.compile(r'\bact\b', re.IGNORECASE),
                    re.compile(r'\beducation act\b', re.IGNORECASE),
                    re.compile(r'\bright.*education\b', re.IGNORECASE)
                ]
            },
            
            "rule": {
                "structure_patterns": [
                    re.compile(r'\brule\s+\d+', re.IGNORECASE),
                    re.compile(r'\bchapter\s+\d+', re.IGNORECASE),
                    re.compile(r'\bform\s+[IVX]+\b', re.IGNORECASE)
                ],
                "language_patterns": [
                    re.compile(r'\bin exercise of.*powers.*conferred\b', re.IGNORECASE),
                    re.compile(r'\bin pursuance of\b', re.IGNORECASE),
                    re.compile(r'\bthese rules may be called\b', re.IGNORECASE),
                    re.compile(r'\bimplementing\b', re.IGNORECASE)
                ],
                "title_patterns": [
                    re.compile(r'\brules?\b', re.IGNORECASE),
                    re.compile(r'\beducation rules\b', re.IGNORECASE)
                ]
            },
            
            "government_order": {
                "structure_patterns": [
                    re.compile(r'\bG\.?O\.?\s*(?:Ms\.?|MS\.?)', re.IGNORECASE),
                    re.compile(r'\border\b.*\bdated?\b', re.IGNORECASE),
                    re.compile(r'\bpresent.*orders?\b', re.IGNORECASE)
                ],
                "language_patterns": [
                    re.compile(r'\borders?\b', re.IGNORECASE),
                    re.compile(r'\bdirected to\b', re.IGNORECASE),
                    re.compile(r'\bhere?by orders?\b', re.IGNORECASE),
                    re.compile(r'\bwith immediate effect\b', re.IGNORECASE)
                ],
                "title_patterns": [
                    re.compile(r'\bgovernment order\b', re.IGNORECASE),
                    re.compile(r'\bG\.?O\b', re.IGNORECASE)
                ]
            },
            
            "judicial": {
                "structure_patterns": [
                    re.compile(r'\bvs?\.?\b.*\bon\s+\d+', re.IGNORECASE),
                    re.compile(r'\bpetitioner\b', re.IGNORECASE),
                    re.compile(r'\brespondent\b', re.IGNORECASE),
                    re.compile(r'\bjudgment\b', re.IGNORECASE)
                ],
                "language_patterns": [
                    re.compile(r'\bheld that\b', re.IGNORECASE),
                    re.compile(r'\bit is.*ordered\b', re.IGNORECASE),
                    re.compile(r'\bthe court\b', re.IGNORECASE),
                    re.compile(r'\bhonourable.*court\b', re.IGNORECASE)
                ],
                "title_patterns": [
                    re.compile(r'\bvs?\.\b', re.IGNORECASE),
                    re.compile(r'\bsupreme court\b', re.IGNORECASE),
                    re.compile(r'\bhigh court\b', re.IGNORECASE)
                ]
            },
            
            "data_report": {
                "structure_patterns": [
                    re.compile(r'\btable\s+\d+', re.IGNORECASE),
                    re.compile(r'\bfigure\s+\d+', re.IGNORECASE),
                    re.compile(r'\bappendix\b', re.IGNORECASE),
                    re.compile(r'\bstatistics\b', re.IGNORECASE)
                ],
                "language_patterns": [
                    re.compile(r'\bdata shows?\b', re.IGNORECASE),
                    re.compile(r'\bpercentage\b', re.IGNORECASE),
                    re.compile(r'\benrollment\b', re.IGNORECASE),
                    re.compile(r'\banalysis\b', re.IGNORECASE)
                ],
                "title_patterns": [
                    re.compile(r'\breport\b', re.IGNORECASE),
                    re.compile(r'\bdata\b', re.IGNORECASE),
                    re.compile(r'\bstatistics\b', re.IGNORECASE),
                    re.compile(r'\budise\b', re.IGNORECASE)
                ]
            },
            
            "budget_finance": {
                "structure_patterns": [
                    re.compile(r'\bvolume\s+[IVX]+', re.IGNORECASE),
                    re.compile(r'\ballocation\b', re.IGNORECASE),
                    re.compile(r'\bexpenditure\b', re.IGNORECASE),
                    re.compile(r'\bbudget\b', re.IGNORECASE)
                ],
                "language_patterns": [
                    re.compile(r'\bcrores?\b', re.IGNORECASE),
                    re.compile(r'\blakhs?\b', re.IGNORECASE),
                    re.compile(r'\bfinancial year\b', re.IGNORECASE),
                    re.compile(r'\bresource allocation\b', re.IGNORECASE)
                ],
                "title_patterns": [
                    re.compile(r'\bbudget\b', re.IGNORECASE),
                    re.compile(r'\bfinance\b', re.IGNORECASE),
                    re.compile(r'\bexpenditure\b', re.IGNORECASE)
                ]
            },
            
            "framework": {
                "structure_patterns": [
                    re.compile(r'\bguidelines?\b', re.IGNORECASE),
                    re.compile(r'\bframework\b', re.IGNORECASE),
                    re.compile(r'\bpolicy\b', re.IGNORECASE),
                    re.compile(r'\bprinciples?\b', re.IGNORECASE)
                ],
                "language_patterns": [
                    re.compile(r'\brecommends?\b', re.IGNORECASE),
                    re.compile(r'\bshould\b', re.IGNORECASE),
                    re.compile(r'\bpromote\b', re.IGNORECASE),
                    re.compile(r'\bensure\b', re.IGNORECASE)
                ],
                "title_patterns": [
                    re.compile(r'\bpolicy\b', re.IGNORECASE),
                    re.compile(r'\bframework\b', re.IGNORECASE),
                    re.compile(r'\bguidelines?\b', re.IGNORECASE)
                ]
            },
            
            "circular": {
                "structure_patterns": [
                    re.compile(r'\bcircular\b', re.IGNORECASE),
                    re.compile(r'\bto all\b', re.IGNORECASE),
                    re.compile(r'\bmemorandum\b', re.IGNORECASE)
                ],
                "language_patterns": [
                    re.compile(r'\binformed that\b', re.IGNORECASE),
                    re.compile(r'\bbrought to.*notice\b', re.IGNORECASE),
                    re.compile(r'\bkindly\b', re.IGNORECASE),
                    re.compile(r'\bfurther action\b', re.IGNORECASE)
                ],
                "title_patterns": [
                    re.compile(r'\bcircular\b', re.IGNORECASE),
                    re.compile(r'\bmemo\b', re.IGNORECASE),
                    re.compile(r'\bletter\b', re.IGNORECASE)
                ]
            }
        }
    
    def _init_feature_extractors(self):
        """Initialize feature extraction tools."""
        # TF-IDF vectorizer for text features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        # ML classifier (will be trained if data is available)
        self.ml_classifier = LogisticRegression(random_state=42)
        self.is_ml_trained = False
    
    def extract_structural_features(self, text: str, title: str = "") -> Dict:
        """
        Extract structural features from document text.
        
        Args:
            text: Document text
            title: Document title
            
        Returns:
            Dictionary of structural features
        """
        features = {
            "has_sections": bool(re.search(r'\bsection\s+\d+', text, re.IGNORECASE)),
            "has_chapters": bool(re.search(r'\bchapter\s+\d+', text, re.IGNORECASE)),
            "has_rules": bool(re.search(r'\brule\s+\d+', text, re.IGNORECASE)),
            "has_go_number": bool(re.search(r'\bG\.?O\.?\s*(?:Ms\.?|MS\.?)', text, re.IGNORECASE)),
            "has_case_citation": bool(re.search(r'\bvs?\.?\b', text, re.IGNORECASE)),
            "has_tables": bool(re.search(r'\btable\s+\d+', text, re.IGNORECASE)),
            "has_financial_terms": bool(re.search(r'\b(?:crores?|lakhs?|budget|allocation)\b', text, re.IGNORECASE)),
            "has_legal_language": bool(re.search(r'\b(?:whereas|enacted|hereby|subject to)\b', text, re.IGNORECASE)),
            "has_administrative_language": bool(re.search(r'\b(?:orders?|directed|immediate effect)\b', text, re.IGNORECASE)),
            "has_policy_language": bool(re.search(r'\b(?:guidelines?|framework|policy|recommends?)\b', text, re.IGNORECASE)),
            "word_count": len(text.split()),
            "avg_sentence_length": self._calculate_avg_sentence_length(text),
            "title_length": len(title.split()) if title else 0
        }
        
        return features
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def calculate_rule_based_scores(self, text: str, title: str = "") -> Dict[str, float]:
        """
        Calculate rule-based classification scores for each document type.
        
        Args:
            text: Document text
            title: Document title
            
        Returns:
            Dictionary of scores for each document type
        """
        scores = {}
        full_text = f"{title} {text}".lower()
        
        for doc_type, indicators in self.type_indicators.items():
            score = 0.0
            
            # Check structure patterns
            for pattern in indicators["structure_patterns"]:
                matches = len(pattern.findall(full_text))
                score += matches * 2.0  # Structure patterns are strong indicators
            
            # Check language patterns
            for pattern in indicators["language_patterns"]:
                matches = len(pattern.findall(full_text))
                score += matches * 1.5  # Language patterns are good indicators
            
            # Check title patterns
            for pattern in indicators["title_patterns"]:
                matches = len(pattern.findall(title.lower()))
                score += matches * 3.0  # Title patterns are very strong indicators
            
            scores[doc_type] = score
        
        return scores
    
    def classify_document(
        self, 
        text: str, 
        title: str = "", 
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Classify a document using multiple approaches.
        
        Args:
            text: Document text
            title: Document title
            metadata: Additional metadata (folder path, etc.)
            
        Returns:
            Classification result with confidence scores
        """
        if not text or not text.strip():
            return {
                "predicted_type": "unknown",
                "confidence": 0.0,
                "method": "insufficient_data",
                "scores": {}
            }
        
        # Extract features
        structural_features = self.extract_structural_features(text, title)
        
        # Get rule-based scores
        rule_scores = self.calculate_rule_based_scores(text, title)
        
        # Normalize scores
        max_score = max(rule_scores.values()) if rule_scores.values() else 0
        if max_score > 0:
            normalized_scores = {k: v/max_score for k, v in rule_scores.items()}
        else:
            normalized_scores = rule_scores
        
        # Find best classification
        if normalized_scores:
            best_type = max(normalized_scores, key=normalized_scores.get)
            best_score = normalized_scores[best_type]
            
            # Determine confidence level
            if best_score >= self.high_confidence_threshold:
                confidence_level = "high"
            elif best_score >= self.medium_confidence_threshold:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            # Consider folder-based classification as fallback
            folder_type = self._infer_from_folder(metadata)
            if folder_type and best_score < self.medium_confidence_threshold:
                # Low confidence in content-based classification, use folder
                return {
                    "predicted_type": folder_type,
                    "confidence": 0.5,  # Medium confidence for folder-based
                    "confidence_level": "medium",
                    "method": "folder_fallback",
                    "content_scores": normalized_scores,
                    "structural_features": structural_features
                }
            
            return {
                "predicted_type": best_type,
                "confidence": best_score,
                "confidence_level": confidence_level,
                "method": "content_based",
                "scores": normalized_scores,
                "structural_features": structural_features,
                "folder_type": folder_type
            }
        
        # Fallback to folder-based classification
        folder_type = self._infer_from_folder(metadata)
        if folder_type:
            return {
                "predicted_type": folder_type,
                "confidence": 0.4,
                "confidence_level": "low",
                "method": "folder_only",
                "scores": {},
                "structural_features": structural_features
            }
        
        return {
            "predicted_type": "unknown",
            "confidence": 0.0,
            "confidence_level": "none",
            "method": "no_classification",
            "scores": normalized_scores,
            "structural_features": structural_features
        }
    
    def _infer_from_folder(self, metadata: Optional[Dict]) -> Optional[str]:
        """Infer document type from folder structure."""
        if not metadata:
            return None
        
        parent_folders = metadata.get("parent_folders", [])
        if not parent_folders:
            return None
        
        # Convert folder names to lower case for matching
        folder_names = [folder.lower() for folder in parent_folders]
        
        # Folder to document type mapping
        folder_mappings = {
            "acts": "act",
            "statutory": "act",
            "rules": "rule",
            "government_orders": "government_order",
            "go": "government_order",
            "judicial": "judicial",
            "judiciary": "judicial",
            "data_reports": "data_report",
            "achievement": "data_report",
            "budget": "budget_finance",
            "financial": "budget_finance",
            "policy": "framework",
            "frameworks": "framework",
            "guidelines": "framework"
        }
        
        # Check for exact matches first
        for folder in folder_names:
            if folder in folder_mappings:
                return folder_mappings[folder]
        
        # Check for partial matches
        for folder in folder_names:
            for key, doc_type in folder_mappings.items():
                if key in folder:
                    return doc_type
        
        return None
    
    def classify_batch(self, documents: List[Dict]) -> List[Dict]:
        """
        Classify multiple documents in batch.
        
        Args:
            documents: List of document dictionaries with text, title, metadata
            
        Returns:
            List of classification results
        """
        results = []
        
        for doc in documents:
            text = doc.get("text", "")
            title = doc.get("title", "")
            metadata = doc.get("metadata", {})
            doc_id = doc.get("doc_id", "unknown")
            
            classification = self.classify_document(text, title, metadata)
            classification["doc_id"] = doc_id
            
            results.append(classification)
        
        logger.info(f"Classified {len(results)} documents")
        
        return results
    
    def get_classification_statistics(self, classifications: List[Dict]) -> Dict:
        """
        Generate statistics about classification results.
        
        Args:
            classifications: List of classification results
            
        Returns:
            Statistics dictionary
        """
        if not classifications:
            return {}
        
        # Count by document type
        type_counts = Counter(c.get("predicted_type", "unknown") for c in classifications)
        
        # Count by confidence level
        confidence_counts = Counter(c.get("confidence_level", "none") for c in classifications)
        
        # Count by method
        method_counts = Counter(c.get("method", "unknown") for c in classifications)
        
        # Calculate average confidence
        confidences = [c.get("confidence", 0) for c in classifications]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_documents": len(classifications),
            "type_distribution": dict(type_counts),
            "confidence_distribution": dict(confidence_counts),
            "method_distribution": dict(method_counts),
            "average_confidence": round(avg_confidence, 3),
            "high_confidence_count": confidence_counts.get("high", 0),
            "medium_confidence_count": confidence_counts.get("medium", 0),
            "low_confidence_count": confidence_counts.get("low", 0)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the document classifier
    classifier = DocumentClassifier()
    
    # Test documents
    test_docs = [
        {
            "doc_id": "rte_act",
            "title": "Right of Children to Free and Compulsory Education Act 2009",
            "text": "Whereas the Constitution of India under Article 21A provides that every child has a right to free and compulsory education. Be it enacted by Parliament in the Sixtieth Year of the Republic of India as follows: Chapter I - Preliminary. Section 1. Short title and commencement.",
            "metadata": {"parent_folders": ["acts"]}
        },
        {
            "doc_id": "go_sample",
            "title": "Government Order MS No. 67/2023",
            "text": "Government of Andhra Pradesh. G.O.MS.No. 67 dated 15.04.2023. Subject: Education - Implementation of Nadu-Nedu Programme. Orders: The Government hereby orders that Nadu-Nedu programme shall be implemented with immediate effect.",
            "metadata": {"parent_folders": ["government_orders"]}
        },
        {
            "doc_id": "court_case",
            "title": "Society for Unaided Schools vs Union of India",
            "text": "The petitioner challenges the constitutional validity of Section 12(1)(c) of the RTE Act. The respondent is Union of India. The Court held that the provision is constitutional and does not violate any fundamental rights.",
            "metadata": {"parent_folders": ["judicial"]}
        }
    ]
    
    # Classify documents
    results = classifier.classify_batch(test_docs)
    
    print("Classification Results:")
    for result in results:
        print(f"Document: {result['doc_id']}")
        print(f"Predicted Type: {result['predicted_type']}")
        print(f"Confidence: {result['confidence']} ({result['confidence_level']})")
        print(f"Method: {result['method']}")
        print()
    
    # Get statistics
    stats = classifier.get_classification_statistics(results)
    print("Classification Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")