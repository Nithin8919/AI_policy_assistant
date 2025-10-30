"""
Entity Validation Module

Validates extracted entities against known valid values to prevent:
- Hallucinations (extracting non-existent entities)
- Typos (corrects with fuzzy matching)
- Invalid combinations (metrics not available for district/year)
- Out-of-range dates
"""

import copy
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from fuzzywuzzy import process, fuzz
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Single validation issue"""
    type: str  # 'invalid_entity', 'typo', 'unavailable', 'out_of_range'
    entity_type: str
    value: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ValidationResult:
    """Complete validation result"""
    is_valid: bool
    validated_entities: Dict[str, List]
    issues: List[ValidationIssue]
    corrections_applied: List[Dict]
    query: str


class EntityValidator:
    """
    Validates extracted entities against knowledge base.
    Provides corrections via fuzzy matching.
    """
    
    def __init__(self, 
                 valid_districts: Optional[Set[str]] = None,
                 valid_schemes: Optional[Set[str]] = None,
                 valid_metrics: Optional[Set[str]] = None,
                 valid_years: Optional[Set[int]] = None,
                 fuzzy_threshold: int = 80):
        """
        Initialize validator with valid entity sets.
        
        Args:
            valid_districts: Set of valid district names
            valid_schemes: Set of valid scheme names
            valid_metrics: Set of valid metric names
            valid_years: Set of years with available data
            fuzzy_threshold: Minimum score for fuzzy matches (0-100)
        """
        self.valid_districts = valid_districts or self._get_default_districts()
        self.valid_schemes = valid_schemes or self._get_default_schemes()
        self.valid_metrics = valid_metrics or self._get_default_metrics()
        self.valid_years = valid_years or self._get_default_years()
        self.fuzzy_threshold = fuzzy_threshold
        
        logger.info(f"EntityValidator initialized with {len(self.valid_districts)} districts, "
                   f"{len(self.valid_schemes)} schemes, {len(self.valid_metrics)} metrics")
    
    def _get_default_districts(self) -> Set[str]:
        """Default AP districts"""
        return {
            "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Rajahmundry",
            "Anantapur", "Chittoor", "Kurnool", "Nellore", "Kadapa",
            "Prakasam", "Srikakulam", "Vizianagaram"
        }
    
    def _get_default_schemes(self) -> Set[str]:
        """Default schemes"""
        return {
            "Nadu-Nedu", "Jagananna Amma Vodi", "Jagananna Gorumudda",
            "Vidya Deevena", "Vasathi Deevena", "Sarva Shiksha Abhiyan",
            "Rashtriya Madhyamik Shiksha Abhiyan"
        }
    
    def _get_default_metrics(self) -> Set[str]:
        """Default metrics"""
        return {
            "Pupil-Teacher Ratio", "PTR", "Enrollment", "Gross Enrollment Ratio",
            "Net Enrollment Ratio", "Dropout Rate", "Budget Allocation",
            "Infrastructure", "Teacher Strength", "Student Strength"
        }
    
    def _get_default_years(self) -> Set[int]:
        """Default available years"""
        return set(range(2010, 2026))  # 2010-2025
    
    def validate(self, entities: Dict[str, List], query: str) -> ValidationResult:
        """
        Validate all extracted entities.
        
        Args:
            entities: Extracted entities from entity extractor
            query: Original query for context
            
        Returns:
            ValidationResult with validated entities and issues
        """
        validated = copy.deepcopy(entities)
        issues = []
        corrections = []
        
        # Validate each entity type
        if 'districts' in entities:
            district_issues, district_corrections = self._validate_districts(
                validated['districts']
            )
            issues.extend(district_issues)
            corrections.extend(district_corrections)
        
        if 'schemes' in entities:
            scheme_issues, scheme_corrections = self._validate_schemes(
                validated['schemes']
            )
            issues.extend(scheme_issues)
            corrections.extend(scheme_corrections)
        
        if 'metrics' in entities:
            metric_issues, metric_corrections = self._validate_metrics(
                validated['metrics']
            )
            issues.extend(metric_issues)
            corrections.extend(metric_corrections)
        
        if 'dates' in entities:
            date_issues = self._validate_dates(validated['dates'])
            issues.extend(date_issues)
        
        # Check for hallucinations (too many entities extracted)
        hallucination_issues = self._check_hallucinations(validated, query)
        issues.extend(hallucination_issues)
        
        # Determine overall validity
        error_issues = [i for i in issues if i.severity == 'error']
        is_valid = len(error_issues) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            validated_entities=validated,
            issues=issues,
            corrections_applied=corrections,
            query=query
        )
        
        logger.debug(f"Validation result: {'VALID' if is_valid else 'INVALID'} "
                    f"({len(issues)} issues, {len(corrections)} corrections)")
        
        return result
    
    def _validate_districts(self, districts: List[Dict]) -> Tuple[List[ValidationIssue], List[Dict]]:
        """Validate district entities"""
        issues = []
        corrections = []
        
        for i, district in enumerate(districts):
            if isinstance(district, dict):
                district_name = district.get('canonical', district.get('text', ''))
            else:
                district_name = str(district)
            
            if district_name not in self.valid_districts:
                # Try fuzzy match
                match, score = process.extractOne(
                    district_name,
                    self.valid_districts,
                    scorer=fuzz.ratio
                )
                
                if score >= self.fuzzy_threshold:
                    # High confidence correction
                    if isinstance(districts[i], dict):
                        districts[i]['canonical'] = match
                        districts[i]['corrected'] = True
                        districts[i]['original'] = district_name
                        districts[i]['fuzzy_score'] = score
                    
                    corrections.append({
                        'entity_type': 'district',
                        'original': district_name,
                        'corrected': match,
                        'confidence': score / 100
                    })
                    
                    issues.append(ValidationIssue(
                        type='typo_corrected',
                        entity_type='district',
                        value=district_name,
                        message=f"'{district_name}' corrected to '{match}'",
                        severity='info',
                        suggestion=match,
                        confidence=score / 100
                    ))
                else:
                    # Invalid district
                    if isinstance(districts[i], dict):
                        districts[i]['valid'] = False
                        districts[i]['suggestion'] = match
                    
                    issues.append(ValidationIssue(
                        type='invalid_entity',
                        entity_type='district',
                        value=district_name,
                        message=f"'{district_name}' is not a recognized AP district",
                        severity='error',
                        suggestion=match,
                        confidence=score / 100
                    ))
        
        return issues, corrections
    
    def _validate_schemes(self, schemes: List[Dict]) -> Tuple[List[ValidationIssue], List[Dict]]:
        """Validate scheme entities"""
        issues = []
        corrections = []
        
        for i, scheme in enumerate(schemes):
            if isinstance(scheme, dict):
                scheme_name = scheme.get('canonical', scheme.get('text', ''))
            else:
                scheme_name = str(scheme)
            
            if scheme_name not in self.valid_schemes:
                # Try fuzzy match
                match, score = process.extractOne(
                    scheme_name,
                    self.valid_schemes,
                    scorer=fuzz.token_set_ratio  # Better for multi-word names
                )
                
                if score >= self.fuzzy_threshold:
                    # Correction
                    if isinstance(schemes[i], dict):
                        schemes[i]['canonical'] = match
                        schemes[i]['corrected'] = True
                        schemes[i]['original'] = scheme_name
                    
                    corrections.append({
                        'entity_type': 'scheme',
                        'original': scheme_name,
                        'corrected': match,
                        'confidence': score / 100
                    })
                    
                    issues.append(ValidationIssue(
                        type='typo_corrected',
                        entity_type='scheme',
                        value=scheme_name,
                        message=f"'{scheme_name}' corrected to '{match}'",
                        severity='info',
                        suggestion=match,
                        confidence=score / 100
                    ))
                else:
                    # Invalid scheme
                    if isinstance(schemes[i], dict):
                        schemes[i]['valid'] = False
                    
                    issues.append(ValidationIssue(
                        type='invalid_entity',
                        entity_type='scheme',
                        value=scheme_name,
                        message=f"Scheme '{scheme_name}' not found in database",
                        severity='warning',
                        suggestion=match,
                        confidence=score / 100
                    ))
        
        return issues, corrections
    
    def _validate_metrics(self, metrics: List[Dict]) -> Tuple[List[ValidationIssue], List[Dict]]:
        """Validate metric entities"""
        issues = []
        corrections = []
        
        for i, metric in enumerate(metrics):
            if isinstance(metric, dict):
                metric_name = metric.get('canonical', metric.get('text', ''))
            else:
                metric_name = str(metric)
            
            if metric_name not in self.valid_metrics:
                # Try fuzzy match
                match, score = process.extractOne(
                    metric_name,
                    self.valid_metrics,
                    scorer=fuzz.token_set_ratio
                )
                
                if score >= self.fuzzy_threshold:
                    # Correction
                    if isinstance(metrics[i], dict):
                        metrics[i]['canonical'] = match
                        metrics[i]['corrected'] = True
                        metrics[i]['original'] = metric_name
                    
                    corrections.append({
                        'entity_type': 'metric',
                        'original': metric_name,
                        'corrected': match,
                        'confidence': score / 100
                    })
        
        return issues, corrections
    
    def _validate_dates(self, dates: List[Dict]) -> List[ValidationIssue]:
        """Validate date entities"""
        issues = []
        
        for date in dates:
            if isinstance(date, dict) and date.get('type') == 'academic_year':
                year_text = date.get('text', '')
                try:
                    # Extract year from "2023-24" format
                    year = int(year_text.split('-')[0])
                    
                    if year not in self.valid_years:
                        issues.append(ValidationIssue(
                            type='out_of_range',
                            entity_type='date',
                            value=year_text,
                            message=f"No data available for academic year {year_text}",
                            severity='warning',
                            suggestion=f"Available years: {min(self.valid_years)}-{max(self.valid_years)}"
                        ))
                except (ValueError, IndexError):
                    issues.append(ValidationIssue(
                        type='invalid_format',
                        entity_type='date',
                        value=year_text,
                        message=f"Invalid date format: {year_text}",
                        severity='error'
                    ))
        
        return issues
    
    def _check_hallucinations(self, entities: Dict[str, List], query: str) -> List[ValidationIssue]:
        """Check for potential hallucinations"""
        issues = []
        query_lower = query.lower()
        
        # Check if district was actually mentioned
        for district in entities.get('districts', []):
            if isinstance(district, dict):
                district_name = district.get('canonical', '')
                if district_name.lower() not in query_lower:
                    issues.append(ValidationIssue(
                        type='possible_hallucination',
                        entity_type='district',
                        value=district_name,
                        message=f"District '{district_name}' not mentioned in query",
                        severity='warning'
                    ))
        
        # Check for excessive entities (likely extraction errors)
        if len(entities.get('districts', [])) > 3:
            issues.append(ValidationIssue(
                type='excessive_entities',
                entity_type='district',
                value=str(len(entities['districts'])),
                message=f"Extracted {len(entities['districts'])} districts - may be too many",
                severity='warning'
            ))
        
        return issues
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get human-readable validation summary"""
        if result.is_valid:
            return "✅ All entities validated successfully"
        
        error_count = sum(1 for i in result.issues if i.severity == 'error')
        warning_count = sum(1 for i in result.issues if i.severity == 'warning')
        correction_count = len(result.corrections_applied)
        
        summary_parts = []
        if error_count > 0:
            summary_parts.append(f"❌ {error_count} errors")
        if warning_count > 0:
            summary_parts.append(f"⚠️  {warning_count} warnings")
        if correction_count > 0:
            summary_parts.append(f"🔧 {correction_count} corrections applied")
        
        return " | ".join(summary_parts)


# Main API function
def validate_entities(entities: Dict[str, List], 
                     query: str,
                     valid_districts: Optional[Set[str]] = None,
                     valid_schemes: Optional[Set[str]] = None) -> ValidationResult:
    """
    Validate extracted entities.
    
    Args:
        entities: Entities from entity extractor
        query: Original query
        valid_districts: Optional custom district set
        valid_schemes: Optional custom scheme set
        
    Returns:
        ValidationResult
    """
    validator = EntityValidator(
        valid_districts=valid_districts,
        valid_schemes=valid_schemes
    )
    return validator.validate(entities, query)

