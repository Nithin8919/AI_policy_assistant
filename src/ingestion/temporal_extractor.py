"""
Temporal Extractor for education policy documents.

Extracts and normalizes temporal information (dates, years, periods) from text.
This enables temporal queries like "show me policy as of date X" and tracking
of when documents were issued, when data was collected, effective dates, etc.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, date
from dateutil import parser as date_parser
import calendar
import logging

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalExtractor:
    """
    Extract and normalize temporal information from policy documents.
    
    Handles various date formats, academic years, financial years, and
    temporal references commonly found in government documents.
    """
    
    def __init__(self):
        """Initialize temporal extractor with patterns and rules."""
        self._compile_patterns()
        self._init_month_mappings()
        
        logger.info("TemporalExtractor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for different temporal formats."""
        
        # Standard date patterns (DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY)
        self.date_patterns = [
            # DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY
            re.compile(r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b'),
            # YYYY.MM.DD, YYYY/MM/DD, YYYY-MM-DD  
            re.compile(r'\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b'),
            # DD Month YYYY (e.g., "15 April 2023")
            re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE),
            # Month DD, YYYY (e.g., "April 15, 2023")
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b', re.IGNORECASE),
            # YYYY Month DD (e.g., "2023 April 15")
            re.compile(r'\b(\d{4})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?\b', re.IGNORECASE)
        ]
        
        # Academic year patterns (2022-23, 2022-2023, FY 2022-23)
        self.academic_year_patterns = [
            re.compile(r'\b(?:AY|Academic Year|FY|Financial Year)?\s*(\d{4})-(\d{2,4})\b', re.IGNORECASE),
            re.compile(r'\b(\d{4})-(\d{2})\s+(?:academic year|session)\b', re.IGNORECASE),
            re.compile(r'\bacademic year\s+(\d{4})-(\d{2,4})\b', re.IGNORECASE)
        ]
        
        # Relative date patterns
        self.relative_patterns = [
            # "with effect from", "w.e.f."
            re.compile(r'\b(?:with effect from|w\.?e\.?f\.?)\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE),
            # "effective from", "from the date of"
            re.compile(r'\b(?:effective from|from the date of)\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE),
            # "as on", "as of"
            re.compile(r'\b(?:as on|as of)\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE),
            # "dated", "dt."
            re.compile(r'\b(?:dated?|dt\.?)\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE),
            # "on or before", "not later than"
            re.compile(r'\b(?:on or before|not later than|before)\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE)
        ]
        
        # Range patterns
        self.range_patterns = [
            # "from X to Y", "between X and Y"
            re.compile(r'\b(?:from|between)\s+([^.]+?)\s+(?:to|and)\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE),
            # "during the period X to Y"
            re.compile(r'\bduring (?:the )?period\s+([^.]+?)\s+to\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE),
            # "in the years X-Y"
            re.compile(r'\bin the years?\s+(\d{4})-(\d{4})\b', re.IGNORECASE)
        ]
        
        # Period patterns (quarters, semesters, etc.)
        self.period_patterns = [
            # Quarters
            re.compile(r'\b(?:Q[1-4]|(?:first|second|third|fourth)\s+quarter)\s+(?:of\s+)?(\d{4})\b', re.IGNORECASE),
            # Semesters
            re.compile(r'\b(?:(?:first|second)\s+semester|sem\s*[12])\s+(?:of\s+)?(\d{4})\b', re.IGNORECASE),
            # Half-yearly
            re.compile(r'\b(?:first|second)\s+half\s+(?:of\s+)?(\d{4})\b', re.IGNORECASE)
        ]
        
        # Special patterns for government documents
        self.govt_patterns = [
            # GO issue date pattern
            re.compile(r'\bG\.?O\.?\s*(?:Ms\.?|MS\.?)\s*(?:No\.?)?\s*\d+[^.]*?dated?\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE),
            # Notification date
            re.compile(r'\bnotification\s+(?:no\.?\s*[^.]*?)?dated?\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE),
            # Order date
            re.compile(r'\border\s+(?:no\.?\s*[^.]*?)?dated?\s+([^.,;]+?)(?:[.,;]|$)', re.IGNORECASE)
        ]
    
    def _init_month_mappings(self):
        """Initialize month name to number mappings."""
        self.month_names = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
    
    def parse_date_string(self, date_str: str) -> Optional[str]:
        """
        Parse a date string to ISO format (YYYY-MM-DD).
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            ISO formatted date string or None if parsing fails
        """
        if not date_str or not date_str.strip():
            return None
        
        date_str = date_str.strip()
        
        # Try standard date patterns first
        for pattern in self.date_patterns:
            match = pattern.search(date_str)
            if match:
                groups = match.groups()
                
                # Handle different group arrangements
                if len(groups) == 3:
                    if pattern == self.date_patterns[0]:  # DD.MM.YYYY format
                        day, month, year = groups
                        try:
                            return f"{year}-{int(month):02d}-{int(day):02d}"
                        except (ValueError, TypeError):
                            continue
                    
                    elif pattern == self.date_patterns[1]:  # YYYY.MM.DD format
                        year, month, day = groups
                        try:
                            return f"{year}-{int(month):02d}-{int(day):02d}"
                        except (ValueError, TypeError):
                            continue
                    
                    elif pattern == self.date_patterns[2]:  # DD Month YYYY
                        day, month_name, year = groups
                        month_num = self.month_names.get(month_name.lower())
                        if month_num:
                            try:
                                return f"{year}-{month_num:02d}-{int(day):02d}"
                            except (ValueError, TypeError):
                                continue
                    
                    elif pattern == self.date_patterns[3]:  # Month DD, YYYY
                        month_name, day, year = groups
                        month_num = self.month_names.get(month_name.lower())
                        if month_num:
                            try:
                                return f"{year}-{month_num:02d}-{int(day):02d}"
                            except (ValueError, TypeError):
                                continue
                    
                    elif pattern == self.date_patterns[4]:  # YYYY Month DD
                        year, month_name, day = groups
                        month_num = self.month_names.get(month_name.lower())
                        if month_num:
                            try:
                                return f"{year}-{month_num:02d}-{int(day):02d}"
                            except (ValueError, TypeError):
                                continue
        
        # Try using dateutil as fallback
        try:
            parsed_date = date_parser.parse(date_str, fuzzy=True)
            return parsed_date.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
        
        # Try to extract just year if full date parsing fails
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return f"{year_match.group(0)}-01-01"  # Default to January 1st
        
        return None
    
    def parse_academic_year(self, year_str: str) -> Optional[Tuple[int, int]]:
        """
        Parse academic year string to tuple (start_year, end_year).
        
        Args:
            year_str: Academic year string like "2022-23" or "2022-2023"
            
        Returns:
            Tuple of (start_year, end_year) or None
        """
        for pattern in self.academic_year_patterns:
            match = pattern.search(year_str)
            if match:
                start_year_str, end_year_str = match.groups()
                
                try:
                    start_year = int(start_year_str)
                    
                    # Handle 2-digit end year (e.g., "22" in "2022-23")
                    if len(end_year_str) == 2:
                        end_year = int(f"{start_year_str[:2]}{end_year_str}")
                    else:
                        end_year = int(end_year_str)
                    
                    return (start_year, end_year)
                    
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def extract_dates(self, text: str) -> List[Dict]:
        """
        Extract all date references from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of date dictionaries with metadata
        """
        dates = []
        
        # Extract standard dates
        for pattern in self.date_patterns:
            for match in pattern.finditer(text):
                date_str = match.group(0)
                parsed_date = self.parse_date_string(date_str)
                
                if parsed_date:
                    dates.append({
                        "type": "absolute_date",
                        "raw_text": date_str,
                        "normalized_date": parsed_date,
                        "position": match.span(),
                        "context": self._get_context(text, match.start()),
                        "confidence": 0.9
                    })
        
        # Extract relative dates
        for pattern in self.relative_patterns:
            for match in pattern.finditer(text):
                date_phrase = match.group(1).strip()
                parsed_date = self.parse_date_string(date_phrase)
                
                if parsed_date:
                    # Determine the type based on the pattern
                    full_match = match.group(0)
                    if "effect from" in full_match.lower() or "w.e.f" in full_match.lower():
                        date_type = "effective_date"
                    elif "as on" in full_match.lower() or "as of" in full_match.lower():
                        date_type = "reference_date"
                    elif "dated" in full_match.lower():
                        date_type = "issue_date"
                    else:
                        date_type = "relative_date"
                    
                    dates.append({
                        "type": date_type,
                        "raw_text": match.group(0),
                        "normalized_date": parsed_date,
                        "position": match.span(),
                        "context": self._get_context(text, match.start()),
                        "confidence": 0.85
                    })
        
        # Extract government document dates
        for pattern in self.govt_patterns:
            for match in pattern.finditer(text):
                date_phrase = match.group(1).strip()
                parsed_date = self.parse_date_string(date_phrase)
                
                if parsed_date:
                    dates.append({
                        "type": "document_date",
                        "raw_text": match.group(0),
                        "normalized_date": parsed_date,
                        "position": match.span(),
                        "context": self._get_context(text, match.start()),
                        "confidence": 0.95
                    })
        
        return self._deduplicate_dates(dates)
    
    def extract_academic_years(self, text: str) -> List[Dict]:
        """
        Extract academic year references from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of academic year dictionaries
        """
        academic_years = []
        
        for pattern in self.academic_year_patterns:
            for match in pattern.finditer(text):
                year_tuple = self.parse_academic_year(match.group(0))
                
                if year_tuple:
                    start_year, end_year = year_tuple
                    
                    academic_years.append({
                        "type": "academic_year",
                        "raw_text": match.group(0),
                        "start_year": start_year,
                        "end_year": end_year,
                        "year_range": f"{start_year}-{end_year}",
                        "position": match.span(),
                        "context": self._get_context(text, match.start()),
                        "confidence": 0.9
                    })
        
        return academic_years
    
    def extract_date_ranges(self, text: str) -> List[Dict]:
        """
        Extract date range references from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of date range dictionaries
        """
        ranges = []
        
        for pattern in self.range_patterns:
            for match in pattern.finditer(text):
                if len(match.groups()) >= 2:
                    start_date_str = match.group(1).strip()
                    end_date_str = match.group(2).strip()
                    
                    start_date = self.parse_date_string(start_date_str)
                    end_date = self.parse_date_string(end_date_str)
                    
                    if start_date and end_date:
                        ranges.append({
                            "type": "date_range",
                            "raw_text": match.group(0),
                            "start_date": start_date,
                            "end_date": end_date,
                            "position": match.span(),
                            "context": self._get_context(text, match.start()),
                            "confidence": 0.85
                        })
        
        return ranges
    
    def extract_periods(self, text: str) -> List[Dict]:
        """
        Extract period references (quarters, semesters) from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of period dictionaries
        """
        periods = []
        
        for pattern in self.period_patterns:
            for match in pattern.finditer(text):
                year = int(match.groups()[-1])  # Last group is always the year
                period_text = match.group(0)
                
                # Determine period type
                if "quarter" in period_text.lower() or "q" in period_text.lower():
                    period_type = "quarter"
                elif "semester" in period_text.lower() or "sem" in period_text.lower():
                    period_type = "semester"
                elif "half" in period_text.lower():
                    period_type = "half_year"
                else:
                    period_type = "period"
                
                periods.append({
                    "type": period_type,
                    "raw_text": period_text,
                    "year": year,
                    "position": match.span(),
                    "context": self._get_context(text, match.start()),
                    "confidence": 0.8
                })
        
        return periods
    
    def _get_context(self, text: str, position: int, window: int = 50) -> str:
        """Get surrounding context for a temporal reference."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()
    
    def _deduplicate_dates(self, dates: List[Dict]) -> List[Dict]:
        """Remove duplicate dates based on normalized date and position."""
        seen = set()
        unique_dates = []
        
        for date_item in dates:
            # Create a key for deduplication
            key = (
                date_item.get("normalized_date"),
                date_item.get("position", (0, 0))[0] // 10  # Group nearby positions
            )
            
            if key not in seen:
                seen.add(key)
                unique_dates.append(date_item)
        
        return unique_dates
    
    def extract_all_temporal(self, text: str) -> Dict:
        """
        Extract all temporal information from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing all temporal extractions
        """
        if not text or not text.strip():
            return {
                "dates": [],
                "academic_years": [],
                "date_ranges": [],
                "periods": [],
                "summary": {
                    "total_temporal_refs": 0,
                    "earliest_date": None,
                    "latest_date": None,
                    "primary_years": []
                }
            }
        
        try:
            # Extract all types of temporal information
            dates = self.extract_dates(text)
            academic_years = self.extract_academic_years(text)
            date_ranges = self.extract_date_ranges(text)
            periods = self.extract_periods(text)
            
            # Generate summary
            summary = self._generate_summary(dates, academic_years, date_ranges, periods)
            
            return {
                "dates": dates,
                "academic_years": academic_years,
                "date_ranges": date_ranges,
                "periods": periods,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error extracting temporal information: {e}")
            return {
                "dates": [],
                "academic_years": [],
                "date_ranges": [],
                "periods": [],
                "summary": {"total_temporal_refs": 0}
            }
    
    def _generate_summary(self, dates, academic_years, date_ranges, periods) -> Dict:
        """Generate summary statistics for temporal extractions."""
        all_dates = []
        all_years = set()
        
        # Collect all dates
        for date_item in dates:
            if date_item.get("normalized_date"):
                all_dates.append(date_item["normalized_date"])
                year = date_item["normalized_date"][:4]
                all_years.add(int(year))
        
        # Collect years from academic years
        for ay in academic_years:
            all_years.add(ay.get("start_year"))
            all_years.add(ay.get("end_year"))
        
        # Collect years from periods
        for period in periods:
            if period.get("year"):
                all_years.add(period["year"])
        
        # Remove None values
        all_years.discard(None)
        
        summary = {
            "total_temporal_refs": len(dates) + len(academic_years) + len(date_ranges) + len(periods),
            "total_dates": len(dates),
            "total_academic_years": len(academic_years),
            "total_ranges": len(date_ranges),
            "total_periods": len(periods)
        }
        
        if all_dates:
            summary["earliest_date"] = min(all_dates)
            summary["latest_date"] = max(all_dates)
        
        if all_years:
            summary["year_span"] = (min(all_years), max(all_years))
            # Most frequently mentioned years
            summary["primary_years"] = sorted(all_years, reverse=True)[:5]
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Test the temporal extractor
    extractor = TemporalExtractor()
    
    # Test text with various temporal references
    test_text = """
    G.O.MS.No. 67 dated 15.04.2023 is issued for the academic year 2023-24.
    With effect from 01.06.2023, new norms shall apply. As per the notification
    dated 10th March 2023, schools must submit data as on 31st March 2024.
    
    During the period from 2020 to 2023, significant changes were made.
    The policy is effective from January 1, 2023 to December 31, 2025.
    First quarter of 2024 shows improvement in learning outcomes.
    """
    
    temporal_info = extractor.extract_all_temporal(test_text)
    
    print("Temporal Information Extracted:")
    print(f"Dates: {len(temporal_info['dates'])}")
    print(f"Academic Years: {len(temporal_info['academic_years'])}")
    print(f"Date Ranges: {len(temporal_info['date_ranges'])}")
    print(f"Periods: {len(temporal_info['periods'])}")
    print(f"Summary: {temporal_info['summary']}")
    
    print("\nDetailed Extractions:")
    for date_item in temporal_info['dates']:
        print(f"- {date_item['type']}: {date_item['normalized_date']} ({date_item['raw_text']})")