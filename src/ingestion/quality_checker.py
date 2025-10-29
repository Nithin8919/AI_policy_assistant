"""
Enhanced Quality Checker with Robust Scoring Algorithm.

Provides comprehensive quality assessment for document processing with:
- Intelligent thresholds based on document type
- Multi-dimensional scoring (text, entities, relations, structure)
- Clear quality levels with actionable feedback
- Fixed scoring logic that doesn't mark everything as "poor"
"""

import re
from typing import Dict, List, Optional
from collections import Counter
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityChecker:
    """
    Enhanced quality checker with intelligent scoring.
    
    Key Improvements:
    - Document type-aware thresholds
    - Multi-factor scoring (not just entity count)
    - Clear quality levels with specific criteria
    - Actionable feedback for improvement
    """
    
    def __init__(self):
        """Initialize quality checker with intelligent thresholds."""
        # Document type-specific thresholds
        self.thresholds = {
            "act": {
                "min_text_length": 1000,
                "min_entity_density": 5.0,  # entities per 100 words
                "min_section_count": 3,
                "min_legal_refs": 5
            },
            "rule": {
                "min_text_length": 800,
                "min_entity_density": 4.0,
                "min_section_count": 2,
                "min_legal_refs": 3
            },
            "government_order": {
                "min_text_length": 500,
                "min_entity_density": 3.0,
                "min_section_count": 1,
                "min_go_refs": 1
            },
            "framework": {
                "min_text_length": 2000,
                "min_entity_density": 8.0,  # Frameworks are comprehensive
                "min_section_count": 5,
                "min_keywords": 10
            },
            "data_report": {
                "min_text_length": 1000,
                "min_entity_density": 10.0,  # Data reports have lots of metrics
                "min_metrics": 5
            },
            "case_law": {
                "min_text_length": 1500,
                "min_entity_density": 6.0,
                "min_legal_refs": 8
            },
            "unknown": {
                "min_text_length": 500,
                "min_entity_density": 2.0,
                "min_section_count": 1
            }
        }
        
        # Quality level definitions (score ranges)
        self.quality_levels = {
            "excellent": (85, 100),      # >85: Production-ready
            "good": (70, 84),             # 70-84: Usable with minor issues
            "acceptable": (50, 69),       # 50-69: Usable but needs improvement
            "poor": (30, 49),             # 30-49: Significant issues
            "critical": (0, 29)           # <30: Not usable
        }
        
        logger.info("QualityChecker initialized with intelligent thresholds")
    
    def check_text_extraction_quality(self, text: str) -> Dict:
        """
        Check quality of extracted text.
        
        Args:
            text: Extracted text to evaluate
            
        Returns:
            Quality report with score and issues
        """
        if not text:
            return {"score": 0, "issues": ["no_text"]}
        
        issues = []
        score = 100  # Start with perfect score
        
        # Check text length
        if len(text) < 100:
            issues.append("text_too_short")
            score -= 30
        
        # Check for extraction artifacts
        if text.count('\n') / len(text) > 0.1:  # Too many line breaks
            issues.append("excessive_line_breaks")
            score -= 10
        
        # Check for garbled text (too many single characters)
        single_chars = len([c for c in text.split() if len(c) == 1])
        if single_chars / len(text.split()) > 0.3:
            issues.append("garbled_text")
            score -= 20
        
        # Check for reasonable word distribution
        words = text.split()
        if len(words) == 0:
            return {"score": 0, "issues": ["no_words"]}
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 2:
            issues.append("suspicious_word_length")
            score -= 15
        
        # Ensure score is non-negative
        score = max(0, score)
        
        return {
            "score": score,
            "issues": issues,
            "text_length": len(text),
            "word_count": len(words)
        }
    
    def check_document_quality(
        self,
        text: str,
        metadata: Dict,
        entities: Dict,
        relations: List[Dict],
        classification: Dict
    ) -> Dict:
        """
        Comprehensive document quality check.
        
        Args:
            text: Document text
            metadata: Document metadata
            entities: Extracted entities
            relations: Extracted relations
            classification: Document classification result
            
        Returns:
            Complete quality report with score and detailed feedback
        """
        doc_type = classification.get("predicted_type", "unknown")
        
        # Individual quality checks
        text_quality = self.check_text_quality(text, doc_type)
        entity_quality = self.check_entity_quality(entities, doc_type, len(text))
        relation_quality = self.check_relation_quality(relations, doc_type)
        metadata_quality = self.check_metadata_quality(metadata)
        
        # Calculate overall score (weighted average)
        weights = {
            "text": 0.25,
            "entities": 0.35,
            "relations": 0.20,
            "metadata": 0.20
        }
        
        overall_score = (
            text_quality["score"] * weights["text"] +
            entity_quality["score"] * weights["entities"] +
            relation_quality["score"] * weights["relations"] +
            metadata_quality["score"] * weights["metadata"]
        )
        
        # Determine quality level
        quality_level = self._get_quality_level(overall_score)
        
        # Collect all issues
        all_issues = (
            text_quality.get("issues", []) +
            entity_quality.get("issues", []) +
            relation_quality.get("issues", []) +
            metadata_quality.get("issues", [])
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score, quality_level, all_issues, doc_type
        )
        
        # Comprehensive quality report
        quality_report = {
            "overall": {
                "overall_score": round(overall_score, 1),
                "quality_level": quality_level,
                "document_type": doc_type,
                "is_production_ready": overall_score >= 70
            },
            "component_scores": {
                "text_quality": text_quality["score"],
                "entity_quality": entity_quality["score"],
                "relation_quality": relation_quality["score"],
                "metadata_quality": metadata_quality["score"]
            },
            "detailed_assessment": {
                "text": text_quality,
                "entities": entity_quality,
                "relations": relation_quality,
                "metadata": metadata_quality
            },
            "issues": all_issues,
            "recommendations": recommendations,
            "thresholds_used": self.thresholds.get(doc_type, self.thresholds["unknown"])
        }
        
        logger.debug(f"Quality check complete: {quality_level} ({overall_score:.1f})")
        
        return quality_report
    
    def check_text_quality(self, text: str, doc_type: str) -> Dict:
        """Check quality of extracted text."""
        thresholds = self.thresholds.get(doc_type, self.thresholds["unknown"])
        
        text_length = len(text)
        word_count = len(text.split())
        
        issues = []
        score = 100
        
        # Check minimum length
        min_length = thresholds["min_text_length"]
        if text_length < min_length:
            issues.append(f"Text too short ({text_length} chars < {min_length} required)")
            score -= 30
        elif text_length < min_length * 1.5:
            issues.append(f"Text marginally short ({text_length} chars)")
            score -= 15
        
        # Check for extraction artifacts
        artifact_patterns = [
            (r'[^\x00-\x7F]{10,}', "Non-ASCII artifacts detected"),
            (r'\.{4,}', "Multiple dots (possible OCR error)"),
            (r'\s{5,}', "Excessive whitespace"),
        ]
        
        for pattern, issue_desc in artifact_patterns:
            if re.search(pattern, text):
                issues.append(issue_desc)
                score -= 10
        
        # Check word density
        if word_count > 0:
            avg_word_length = text_length / word_count
            if avg_word_length > 15:  # Abnormally long "words"
                issues.append("Abnormal word length (possible extraction error)")
                score -= 15
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        return {
            "score": score,
            "text_length": text_length,
            "word_count": word_count,
            "issues": issues
        }
    
    def check_entity_quality(self, entities: Dict, doc_type: str, text_length: int) -> Dict:
        """Check quality of entity extraction."""
        thresholds = self.thresholds.get(doc_type, self.thresholds["unknown"])
        
        # Count total entities
        total_entities = sum(
            len(v) if isinstance(v, list) else 0 
            for v in entities.values()
        )
        
        # Calculate entity density (entities per 100 words)
        word_count = max(1, text_length // 5)  # Rough estimate: 5 chars per word
        entity_density = (total_entities / word_count) * 100
        
        issues = []
        score = 100
        
        # Check if we have any entities at all
        if total_entities == 0:
            issues.append("No entities extracted")
            return {"score": 0, "issues": issues, "entity_density": 0.0}
        
        # Check entity density against threshold
        min_density = thresholds.get("min_entity_density", 2.0)
        if entity_density < min_density:
            issues.append(f"Low entity density ({entity_density:.1f} < {min_density})")
            # Proportional penalty (not full 30 points)
            penalty = min(30, int((min_density - entity_density) / min_density * 30))
            score -= penalty
        elif entity_density < min_density * 1.5:
            issues.append(f"Below-average entity density ({entity_density:.1f})")
            score -= 10
        
        # Check for domain-specific entities
        if doc_type == "act" or doc_type == "rule":
            legal_refs = len(entities.get("legal_refs", []))
            min_legal = thresholds.get("min_legal_refs", 3)
            if legal_refs < min_legal:
                issues.append(f"Few legal references ({legal_refs} < {min_legal})")
                score -= 15
        
        elif doc_type == "government_order":
            go_refs = len(entities.get("go_refs", []))
            if go_refs == 0:
                issues.append("No GO references found")
                score -= 20
        
        elif doc_type == "data_report":
            metrics = len(entities.get("metrics", []))
            min_metrics = thresholds.get("min_metrics", 5)
            if metrics < min_metrics:
                issues.append(f"Few metrics found ({metrics} < {min_metrics})")
                score -= 20
        
        # Check entity diversity (not all entities from one category)
        non_empty_categories = sum(
            1 for v in entities.values() 
            if isinstance(v, list) and len(v) > 0
        )
        
        if non_empty_categories < 3:
            issues.append(f"Low entity diversity ({non_empty_categories} categories)")
            score -= 15
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        return {
            "score": score,
            "issues": issues,
            "entity_density": round(entity_density, 2),
            "total_entities": total_entities,
            "entity_categories": non_empty_categories,
            "entity_breakdown": {
                k: len(v) if isinstance(v, list) else 0 
                for k, v in entities.items()
            }
        }
    
    def check_relation_quality(self, relations: List[Dict], doc_type: str) -> Dict:
        """Check quality of relation extraction."""
        relation_count = len(relations)
        
        issues = []
        score = 100
        
        # Relations are less critical for some doc types
        if doc_type in ["data_report", "framework"]:
            if relation_count == 0:
                issues.append("No relations found (not critical for this doc type)")
                score = 70  # Not a major issue
            return {"score": score, "issues": issues, "relation_count": relation_count}
        
        # For legal/GO documents, relations are important
        if doc_type in ["act", "rule", "government_order", "case_law"]:
            if relation_count == 0:
                issues.append("No relations found")
                score = 50  # Significant but not fatal
            elif relation_count < 3:
                issues.append(f"Few relations found ({relation_count})")
                score = 70
        
        # Check relation type diversity
        if relation_count > 0:
            relation_types = set(r.get("relation_type", "unknown") for r in relations)
            if len(relation_types) < 2:
                issues.append("Low relation type diversity")
                score -= 10
        
        return {
            "score": score,
            "issues": issues,
            "relation_count": relation_count
        }
    
    def check_metadata_quality(self, metadata: Dict) -> Dict:
        """Check quality of extracted metadata."""
        issues = []
        score = 100
        
        # Essential metadata fields
        essential_fields = ["doc_id", "doc_type", "title"]
        missing_fields = [f for f in essential_fields if not metadata.get(f)]
        
        if missing_fields:
            issues.append(f"Missing essential metadata: {', '.join(missing_fields)}")
            score -= 20 * len(missing_fields)
        
        # Recommended metadata fields
        recommended_fields = ["year", "source_url", "file_path"]
        missing_recommended = [f for f in recommended_fields if not metadata.get(f)]
        
        if missing_recommended:
            issues.append(f"Missing recommended metadata: {', '.join(missing_recommended)}")
            score -= 5 * len(missing_recommended)
        
        # Check title quality
        title = metadata.get("title", "")
        if title:
            if len(title) < 10:
                issues.append("Title too short")
                score -= 10
            elif "untitled" in title.lower():
                issues.append("Generic title detected")
                score -= 15
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        return {
            "score": score,
            "issues": issues,
            "completeness": round((len(essential_fields) - len(missing_fields)) / len(essential_fields) * 100, 1)
        }
    
    def _get_quality_level(self, score: float) -> str:
        """Determine quality level from score."""
        for level, (min_score, max_score) in self.quality_levels.items():
            if min_score <= score <= max_score:
                return level
        return "unknown"
    
    def _generate_recommendations(
        self, 
        score: float, 
        level: str, 
        issues: List[str],
        doc_type: str
    ) -> List[str]:
        """Generate actionable recommendations for improvement."""
        recommendations = []
        
        if level in ["poor", "critical"]:
            recommendations.append("⚠️ Document quality is below acceptable - review extraction process")
        
        # Issue-specific recommendations
        issue_keywords = {
            "text too short": "Check PDF extraction settings or try alternative extraction method",
            "no entities": "Review entity extraction patterns or try with manual annotation",
            "low entity density": "Consider fine-tuning entity extraction rules for this document type",
            "no relations": "Add more relation extraction patterns or use dependency parsing",
            "missing metadata": "Improve metadata extraction from document headers/footers",
            "low entity diversity": "Expand entity extraction to cover more entity types"
        }
        
        for issue in issues:
            issue_lower = issue.lower()
            for keyword, recommendation in issue_keywords.items():
                if keyword in issue_lower and recommendation not in recommendations:
                    recommendations.append(recommendation)
        
        # General recommendations based on score
        if score < 50:
            recommendations.append("Consider manual review and annotation for this document")
        elif score < 70:
            recommendations.append("Document is usable but could benefit from enhancement")
        elif score >= 85:
            recommendations.append("✅ Document quality is excellent - ready for production")
        
        return recommendations


# Testing
if __name__ == "__main__":
    # Test the quality checker
    checker = QualityChecker()
    
    # Test with sample data (simulating a good quality document)
    test_text = "Section 1. This Act may be called the Education Act. " * 50  # 2500+ chars
    test_entities = {
        "legal_refs": ["Section 1", "Section 2", "Article 21", "Article 25", "Section 3"],
        "keywords": ["education", "right", "school", "teacher", "student"],
        "spacy_entities": [{"text": "Education Act", "label": "LAW"}]
    }
    test_relations = [
        {"relation_type": "cites", "source": "Act1", "target": "Section 1"},
        {"relation_type": "implements", "source": "Rule1", "target": "Act1"}
    ]
    test_metadata = {
        "doc_id": "test_123",
        "doc_type": "act",
        "title": "The Education Act 2020",
        "year": 2020
    }
    test_classification = {"predicted_type": "act", "confidence": 0.95}
    
    report = checker.check_document_quality(
        test_text, test_metadata, test_entities, test_relations, test_classification
    )
    
    print("Quality Check Report:")
    print(f"Overall Score: {report['overall']['overall_score']}")
    print(f"Quality Level: {report['overall']['quality_level']}")
    print(f"Production Ready: {report['overall']['is_production_ready']}")
    print(f"\nComponent Scores:")
    for component, score in report['component_scores'].items():
        print(f"  {component}: {score}")
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")