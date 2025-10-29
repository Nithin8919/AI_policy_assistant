"""
Quality Checker for education policy documents.

Performs quality control checks during ingestion to identify and flag
issues with text extraction, metadata completeness, entity extraction,
and overall document processing quality.
"""

import re
import string
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import logging
from collections import Counter
import numpy as np

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityChecker:
    """
    Comprehensive quality control for document ingestion pipeline.
    
    Checks multiple aspects of document processing quality:
    - Text extraction quality
    - Metadata completeness  
    - Entity extraction effectiveness
    - Relation extraction coverage
    - Overall processing success
    """
    
    def __init__(self):
        """Initialize quality checker with thresholds and patterns."""
        self._init_quality_thresholds()
        self._init_language_patterns()
        self._init_validation_rules()
        
        logger.info("QualityChecker initialized")
    
    def _init_quality_thresholds(self):
        """Initialize quality scoring thresholds."""
        self.thresholds = {
            # Text quality
            "min_text_length": 100,  # Minimum characters for valid text
            "max_special_char_ratio": 0.3,  # Max ratio of special chars
            "min_word_ratio": 0.7,  # Min ratio of real words
            "min_sentence_ratio": 0.05,  # Min ratio of complete sentences
            
            # Metadata quality
            "required_metadata_fields": ["title", "doc_type", "year"],
            "min_metadata_completeness": 0.7,
            
            # Entity extraction quality
            "min_entities_per_1000_chars": 1,  # Expected entity density
            "expected_go_pattern_in_orders": True,
            "expected_sections_in_acts": True,
            
            # Overall quality
            "excellent_score": 90,
            "good_score": 75,
            "acceptable_score": 60,
            "poor_score": 40
        }
    
    def _init_language_patterns(self):
        """Initialize patterns for language detection and quality."""
        # Common English words for language detection
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
            'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
            'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
            'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
            'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us'
        }
        
        # Education domain words
        self.education_words = {
            'education', 'school', 'student', 'teacher', 'learning', 'curriculum',
            'academic', 'classroom', 'enrollment', 'admission', 'policy',
            'act', 'rule', 'section', 'government', 'right', 'child',
            'primary', 'secondary', 'training', 'development', 'quality',
            'infrastructure', 'management', 'committee', 'district'
        }
        
        # Gibberish indicators
        self.gibberish_patterns = [
            re.compile(r'[^\w\s]{4,}'),  # Multiple consecutive special chars
            re.compile(r'\w{20,}'),      # Very long words (likely OCR errors)
            re.compile(r'(.)\1{5,}'),    # Repeated characters
            re.compile(r'[A-Z]{10,}'),   # Long sequences of capitals
        ]
    
    def _init_validation_rules(self):
        """Initialize document type specific validation rules."""
        self.validation_rules = {
            "act": {
                "should_have_sections": True,
                "should_have_chapters": True,
                "should_have_legal_language": True,
                "min_length": 1000
            },
            "rule": {
                "should_have_rules": True,
                "should_have_implementing_language": True,
                "min_length": 500
            },
            "government_order": {
                "should_have_go_number": True,
                "should_have_order_language": True,
                "should_have_dates": True,
                "min_length": 200
            },
            "judicial": {
                "should_have_case_parties": True,
                "should_have_judgment_language": True,
                "min_length": 500
            },
            "data_report": {
                "should_have_data_elements": True,
                "should_have_statistics": True,
                "min_length": 300
            }
        }
    
    def check_text_extraction_quality(self, text: str) -> Dict:
        """
        Check the quality of text extraction.
        
        Args:
            text: Extracted text to analyze
            
        Returns:
            Dictionary with text quality metrics
        """
        if not text or not text.strip():
            return {
                "score": 0,
                "issues": ["empty_text"],
                "metrics": {"length": 0}
            }
        
        text = text.strip()
        length = len(text)
        issues = []
        
        # Check minimum length
        if length < self.thresholds["min_text_length"]:
            issues.append("too_short")
        
        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / length if length > 0 else 0
        
        if special_ratio > self.thresholds["max_special_char_ratio"]:
            issues.append("too_many_special_chars")
        
        # Check for real words vs gibberish
        words = text.split()
        if words:
            real_words = sum(1 for word in words if self._is_real_word(word))
            word_ratio = real_words / len(words)
            
            if word_ratio < self.thresholds["min_word_ratio"]:
                issues.append("low_word_quality")
        else:
            word_ratio = 0
            issues.append("no_words")
        
        # Check for complete sentences
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = sum(1 for s in sentences if len(s.strip().split()) >= 3)
        sentence_ratio = complete_sentences / len(sentences) if sentences else 0
        
        if sentence_ratio < self.thresholds["min_sentence_ratio"]:
            issues.append("few_complete_sentences")
        
        # Check for gibberish patterns
        for pattern in self.gibberish_patterns:
            if pattern.search(text):
                issues.append("gibberish_detected")
                break
        
        # Check language (basic English detection)
        english_score = self._calculate_english_score(text)
        if english_score < 0.3:
            issues.append("non_english_text")
        
        # Calculate overall text quality score
        score = self._calculate_text_score(
            length, special_ratio, word_ratio, sentence_ratio, english_score, issues
        )
        
        return {
            "score": score,
            "issues": issues,
            "metrics": {
                "length": length,
                "special_char_ratio": round(special_ratio, 3),
                "word_ratio": round(word_ratio, 3),
                "sentence_ratio": round(sentence_ratio, 3),
                "english_score": round(english_score, 3),
                "word_count": len(words)
            }
        }
    
    def _is_real_word(self, word: str) -> bool:
        """Check if a word appears to be a real English word."""
        word_clean = re.sub(r'[^\w]', '', word.lower())
        
        if len(word_clean) < 2:
            return False
        
        # Check if it's a common word
        if word_clean in self.common_words:
            return True
        
        # Check if it's an education domain word
        if word_clean in self.education_words:
            return True
        
        # Check basic word patterns
        if re.match(r'^[a-z]+$', word_clean) and len(word_clean) >= 3:
            # Basic heuristics for real words
            vowels = sum(1 for c in word_clean if c in 'aeiou')
            consonants = len(word_clean) - vowels
            
            # Reasonable vowel to consonant ratio
            if vowels > 0 and consonants > 0:
                ratio = vowels / len(word_clean)
                if 0.1 <= ratio <= 0.7:
                    return True
        
        return False
    
    def _calculate_english_score(self, text: str) -> float:
        """Calculate how likely the text is to be English."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return 0.0
        
        common_word_count = sum(1 for word in words if word in self.common_words)
        return common_word_count / len(words)
    
    def _calculate_text_score(
        self, 
        length: int, 
        special_ratio: float, 
        word_ratio: float, 
        sentence_ratio: float, 
        english_score: float, 
        issues: List[str]
    ) -> int:
        """Calculate overall text quality score (0-100)."""
        score = 100
        
        # Deduct points for issues
        if "empty_text" in issues:
            return 0
        
        if "too_short" in issues:
            score -= 30
        
        if "too_many_special_chars" in issues:
            score -= 25
        
        if "low_word_quality" in issues:
            score -= 30
        
        if "few_complete_sentences" in issues:
            score -= 20
        
        if "gibberish_detected" in issues:
            score -= 40
        
        if "non_english_text" in issues:
            score -= 25
        
        if "no_words" in issues:
            score -= 50
        
        # Bonus for good metrics
        if english_score > 0.5:
            score += 10
        
        if word_ratio > 0.9:
            score += 10
        
        return max(0, min(100, score))
    
    def check_metadata_completeness(self, metadata: Dict, doc_type: str = None) -> Dict:
        """
        Check metadata completeness and validity.
        
        Args:
            metadata: Document metadata
            doc_type: Document type (if known)
            
        Returns:
            Dictionary with metadata quality assessment
        """
        if not metadata:
            return {
                "score": 0,
                "issues": ["no_metadata"],
                "completeness": 0.0
            }
        
        issues = []
        required_fields = self.thresholds["required_metadata_fields"].copy()
        
        # Add document type specific requirements
        if doc_type == "government_order":
            required_fields.extend(["go_number", "date"])
        elif doc_type == "act":
            required_fields.extend(["enactment_year"])
        elif doc_type == "judicial":
            required_fields.extend(["court", "parties"])
        
        # Check field presence and validity
        present_fields = 0
        total_fields = len(required_fields)
        
        for field in required_fields:
            if field in metadata and metadata[field]:
                # Check if field has meaningful value
                value = metadata[field]
                if isinstance(value, str) and value.strip():
                    present_fields += 1
                elif isinstance(value, (int, float)) and value > 0:
                    present_fields += 1
                elif isinstance(value, list) and value:
                    present_fields += 1
                else:
                    issues.append(f"invalid_{field}")
            else:
                issues.append(f"missing_{field}")
        
        completeness = present_fields / total_fields if total_fields > 0 else 0
        
        # Check for year validity
        if "year" in metadata:
            year = metadata["year"]
            if isinstance(year, (int, float)):
                if year < 1900 or year > 2030:
                    issues.append("invalid_year")
            else:
                issues.append("non_numeric_year")
        
        # Calculate score
        score = int(completeness * 100)
        
        if completeness < self.thresholds["min_metadata_completeness"]:
            issues.append("low_completeness")
        
        return {
            "score": score,
            "issues": issues,
            "completeness": round(completeness, 3),
            "present_fields": present_fields,
            "total_required": total_fields
        }
    
    def check_entity_extraction_quality(self, entities: Dict, text: str, doc_type: str = None) -> Dict:
        """
        Check quality of entity extraction.
        
        Args:
            entities: Extracted entities
            text: Source text
            doc_type: Document type
            
        Returns:
            Dictionary with entity extraction quality assessment
        """
        if not entities:
            return {
                "score": 0,
                "issues": ["no_entities"],
                "entity_density": 0.0
            }
        
        issues = []
        text_length = len(text) if text else 0
        
        # Count total entities extracted
        total_entities = 0
        for entity_type, entity_list in entities.items():
            if isinstance(entity_list, list):
                total_entities += len(entity_list)
        
        # Calculate entity density
        entity_density = (total_entities / text_length * 1000) if text_length > 0 else 0
        
        if entity_density < self.thresholds["min_entities_per_1000_chars"]:
            issues.append("low_entity_density")
        
        # Document type specific checks
        if doc_type == "government_order":
            if not entities.get("go_refs"):
                issues.append("missing_go_references")
            if not entities.get("schemes") and not entities.get("districts"):
                issues.append("missing_administrative_entities")
        
        elif doc_type == "act":
            if not entities.get("legal_refs"):
                issues.append("missing_legal_references")
            sections = [ref for ref in entities.get("legal_refs", []) 
                       if ref.get("type") == "section"]
            if not sections:
                issues.append("missing_sections")
        
        elif doc_type == "judicial":
            if not entities.get("legal_refs"):
                issues.append("missing_legal_citations")
        
        # Check for empty entity categories
        empty_categories = sum(1 for v in entities.values() 
                              if isinstance(v, list) and len(v) == 0)
        total_categories = len(entities)
        
        if empty_categories / total_categories > 0.7:
            issues.append("mostly_empty_categories")
        
        # Calculate score
        base_score = min(100, int(entity_density * 50))  # Scale entity density
        
        # Deduct for issues
        if "no_entities" in issues:
            return {"score": 0, "issues": issues, "entity_density": 0.0}
        
        if "low_entity_density" in issues:
            base_score -= 30
        
        if "missing_go_references" in issues:
            base_score -= 25
        
        if "missing_legal_references" in issues:
            base_score -= 25
        
        if "mostly_empty_categories" in issues:
            base_score -= 20
        
        score = max(0, base_score)
        
        return {
            "score": score,
            "issues": issues,
            "entity_density": round(entity_density, 2),
            "total_entities": total_entities,
            "entity_breakdown": {k: len(v) if isinstance(v, list) else 0 
                               for k, v in entities.items()}
        }
    
    def check_relation_extraction_quality(self, relations: List[Dict], doc_type: str = None) -> Dict:
        """
        Check quality of relation extraction.
        
        Args:
            relations: Extracted relations
            doc_type: Document type
            
        Returns:
            Dictionary with relation extraction quality assessment
        """
        if not relations:
            return {
                "score": 30,  # Not critical, so don't completely fail
                "issues": ["no_relations"],
                "relation_count": 0
            }
        
        issues = []
        relation_count = len(relations)
        
        # Check for expected relation types by document type
        relation_types = set(rel.get("relation_type") for rel in relations)
        
        if doc_type == "government_order":
            if "implements" not in relation_types and "cites" not in relation_types:
                issues.append("missing_implementation_relations")
            
            # Check for supersession relations in superseding orders
            supersedes_count = sum(1 for rel in relations 
                                 if rel.get("relation_type") == "supersedes")
            if supersedes_count == 0:
                # Not always required, so just note it
                pass
        
        elif doc_type == "act":
            if "defines" not in relation_types:
                issues.append("missing_definitions")
        
        # Check relation quality
        high_confidence_relations = sum(1 for rel in relations 
                                      if rel.get("confidence", 0) > 0.8)
        
        if relation_count > 0:
            confidence_ratio = high_confidence_relations / relation_count
            if confidence_ratio < 0.5:
                issues.append("low_confidence_relations")
        
        # Calculate score
        base_score = min(100, relation_count * 20)  # 20 points per relation, max 100
        
        if "missing_implementation_relations" in issues:
            base_score -= 15
        
        if "low_confidence_relations" in issues:
            base_score -= 20
        
        score = max(30, base_score)  # Minimum score since relations are not critical
        
        return {
            "score": score,
            "issues": issues,
            "relation_count": relation_count,
            "relation_types": list(relation_types),
            "high_confidence_count": high_confidence_relations
        }
    
    def check_document_type_consistency(self, predicted_type: str, classification_result: Dict) -> Dict:
        """Check consistency of document type classification."""
        if not predicted_type or predicted_type == "unknown":
            return {
                "score": 0,
                "issues": ["unknown_document_type"],
                "consistency": False
            }
        
        issues = []
        confidence = classification_result.get("confidence", 0)
        confidence_level = classification_result.get("confidence_level", "none")
        
        if confidence_level == "low":
            issues.append("low_classification_confidence")
        
        if confidence < 0.5:
            issues.append("very_low_confidence")
        
        # Check method used
        method = classification_result.get("method", "unknown")
        if method == "folder_only":
            issues.append("folder_based_only")
        
        score = int(confidence * 100)
        
        return {
            "score": score,
            "issues": issues,
            "consistency": confidence > 0.6,
            "method": method
        }
    
    def calculate_overall_quality_score(self, quality_checks: Dict) -> Dict:
        """
        Calculate overall quality score from individual check results.
        
        Args:
            quality_checks: Dictionary of individual quality check results
            
        Returns:
            Overall quality assessment
        """
        # Weight different quality aspects
        weights = {
            "text_extraction": 0.3,
            "metadata_completeness": 0.2,
            "entity_extraction": 0.25,
            "relation_extraction": 0.15,
            "document_classification": 0.1
        }
        
        weighted_score = 0
        total_weight = 0
        all_issues = []
        
        for check_name, weight in weights.items():
            if check_name in quality_checks:
                check_result = quality_checks[check_name]
                score = check_result.get("score", 0)
                weighted_score += score * weight
                total_weight += weight
                all_issues.extend(check_result.get("issues", []))
        
        # Normalize score
        overall_score = int(weighted_score / total_weight) if total_weight > 0 else 0
        
        # Determine quality level
        if overall_score >= self.thresholds["excellent_score"]:
            quality_level = "excellent"
        elif overall_score >= self.thresholds["good_score"]:
            quality_level = "good"
        elif overall_score >= self.thresholds["acceptable_score"]:
            quality_level = "acceptable"
        elif overall_score >= self.thresholds["poor_score"]:
            quality_level = "poor"
        else:
            quality_level = "critical"
        
        # Count critical issues
        critical_issues = [issue for issue in all_issues 
                          if issue in ["empty_text", "no_metadata", "unknown_document_type"]]
        
        return {
            "overall_score": overall_score,
            "quality_level": quality_level,
            "all_issues": list(set(all_issues)),  # Deduplicate
            "critical_issues": critical_issues,
            "issue_count": len(set(all_issues)),
            "needs_review": quality_level in ["poor", "critical"] or len(critical_issues) > 0,
            "component_scores": {k: v.get("score", 0) for k, v in quality_checks.items()}
        }
    
    def check_document_quality(
        self, 
        text: str, 
        metadata: Dict, 
        entities: Dict, 
        relations: List[Dict], 
        classification_result: Dict
    ) -> Dict:
        """
        Perform comprehensive quality check on a processed document.
        
        Args:
            text: Extracted document text
            metadata: Document metadata
            entities: Extracted entities
            relations: Extracted relations
            classification_result: Document classification result
            
        Returns:
            Comprehensive quality assessment
        """
        doc_type = classification_result.get("predicted_type")
        
        # Perform individual quality checks
        quality_checks = {
            "text_extraction": self.check_text_extraction_quality(text),
            "metadata_completeness": self.check_metadata_completeness(metadata, doc_type),
            "entity_extraction": self.check_entity_extraction_quality(entities, text, doc_type),
            "relation_extraction": self.check_relation_extraction_quality(relations, doc_type),
            "document_classification": self.check_document_type_consistency(doc_type, classification_result)
        }
        
        # Calculate overall quality
        overall_quality = self.calculate_overall_quality_score(quality_checks)
        
        # Combine results
        quality_report = {
            "overall": overall_quality,
            "components": quality_checks,
            "recommendations": self._generate_recommendations(quality_checks, overall_quality)
        }
        
        logger.debug(f"Document quality score: {overall_quality['overall_score']} ({overall_quality['quality_level']})")
        
        return quality_report
    
    def _generate_recommendations(self, quality_checks: Dict, overall_quality: Dict) -> List[str]:
        """Generate recommendations for improving document quality."""
        recommendations = []
        
        # Text extraction recommendations
        text_issues = quality_checks.get("text_extraction", {}).get("issues", [])
        if "too_short" in text_issues:
            recommendations.append("Consider using alternative PDF extraction methods")
        if "gibberish_detected" in text_issues:
            recommendations.append("OCR quality is poor - consider manual review or re-scanning")
        if "non_english_text" in text_issues:
            recommendations.append("Document may not be in English - verify language")
        
        # Metadata recommendations
        metadata_issues = quality_checks.get("metadata_completeness", {}).get("issues", [])
        if "low_completeness" in metadata_issues:
            recommendations.append("Improve metadata extraction or add manual metadata")
        
        # Entity extraction recommendations
        entity_issues = quality_checks.get("entity_extraction", {}).get("issues", [])
        if "low_entity_density" in entity_issues:
            recommendations.append("Review entity extraction patterns and dictionaries")
        
        # Overall recommendations
        if overall_quality.get("quality_level") == "critical":
            recommendations.append("Document requires manual review before use")
        elif overall_quality.get("quality_level") == "poor":
            recommendations.append("Consider re-processing with different extraction settings")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Test the quality checker
    checker = QualityChecker()
    
    # Test with good quality document
    good_text = """
    Section 12(1)(c) of the Right to Education Act mandates that private schools
    shall reserve 25% of their seats for children from economically weaker sections
    and disadvantaged groups. This provision ensures inclusive education and
    promotes social equity in the education system.
    """
    
    good_metadata = {
        "title": "RTE Act Section 12",
        "doc_type": "act",
        "year": 2009,
        "file_format": "pdf"
    }
    
    good_entities = {
        "legal_refs": [{"type": "section", "number": "12", "act": "RTE Act"}],
        "schemes": [],
        "keywords": ["25% reservation", "economically weaker sections"]
    }
    
    good_relations = [
        {"relation_type": "mandates", "confidence": 0.9}
    ]
    
    good_classification = {
        "predicted_type": "act",
        "confidence": 0.95,
        "confidence_level": "high",
        "method": "content_based"
    }
    
    # Check quality
    quality_report = checker.check_document_quality(
        good_text, good_metadata, good_entities, good_relations, good_classification
    )
    
    print("Quality Report:")
    print(f"Overall Score: {quality_report['overall']['overall_score']}")
    print(f"Quality Level: {quality_report['overall']['quality_level']}")
    print(f"Issues: {quality_report['overall']['all_issues']}")
    print(f"Needs Review: {quality_report['overall']['needs_review']}")
    print(f"Recommendations: {quality_report['recommendations']}")