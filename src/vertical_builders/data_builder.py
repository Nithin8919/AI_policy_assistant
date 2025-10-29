"""
Data Report Database Builder for Metrics and Statistics.

Creates a structured data database with:
- Metrics catalog (all available metrics)
- Time series tracking
- District-wise breakdowns
- Category-wise breakdowns (SC/ST/OBC)
- Trend analysis preparation
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime

from src.vertical_builders.base_builder import BaseVerticalBuilder, extract_number_from_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataDatabaseBuilder(BaseVerticalBuilder):
    """
    Build specialized database for data reports and metrics.
    
    Creates structured access to statistical data with time series,
    district breakdowns, and category-wise analysis.
    """
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize data database builder."""
        super().__init__(data_dir, output_dir)
        
        # Initialize metric patterns
        self._init_metric_patterns()
        
        # Initialize districts and categories
        self._init_dimensions()
        
        # Statistics
        self.data_stats = {
            "metrics_cataloged": 0,
            "data_points_extracted": 0,
            "time_periods_covered": 0,
            "districts_covered": 0
        }
    
    def _init_metric_patterns(self):
        """Initialize patterns for metric extraction."""
        # Metric name patterns
        self.metric_patterns = {
            "enrollment": [
                re.compile(r'(?:total\s+)?enrollment', re.IGNORECASE),
                re.compile(r'number of students', re.IGNORECASE)
            ],
            "ptr": [
                re.compile(r'pupil[- ]teacher ratio', re.IGNORECASE),
                re.compile(r'PTR', re.IGNORECASE),
                re.compile(r'student[- ]teacher ratio', re.IGNORECASE)
            ],
            "dropout_rate": [
                re.compile(r'dropout rate', re.IGNORECASE),
                re.compile(r'drop[- ]out', re.IGNORECASE)
            ],
            "retention_rate": [
                re.compile(r'retention rate', re.IGNORECASE)
            ],
            "transition_rate": [
                re.compile(r'transition rate', re.IGNORECASE)
            ],
            "pass_percentage": [
                re.compile(r'pass percentage', re.IGNORECASE),
                re.compile(r'pass rate', re.IGNORECASE)
            ],
            "ger": [
                re.compile(r'gross enrollment ratio', re.IGNORECASE),
                re.compile(r'GER', re.IGNORECASE)
            ],
            "ner": [
                re.compile(r'net enrollment ratio', re.IGNORECASE),
                re.compile(r'NER', re.IGNORECASE)
            ]
        }
        
        # Number extraction patterns
        self.number_patterns = [
            re.compile(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|lakh|thousand|%)?'),
            re.compile(r'(\d+(?:\.\d+)?):(\d+)'),  # Ratio format 28:1
        ]
    
    def _init_dimensions(self):
        """Initialize dimension lists (districts, categories)."""
        self.ap_districts = [
            "Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna",
            "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam",
            "Vizianagaram", "West Godavari", "YSR Kadapa"
        ]
        
        self.social_categories = [
            "SC", "ST", "OBC", "BC", "EWS", "Minority", "General"
        ]
        
        self.gender_categories = ["Male", "Female", "Transgender", "Total"]
        
        self.location_categories = ["Rural", "Urban", "Total"]
    
    def get_vertical_name(self) -> str:
        """Get vertical name for output directory."""
        return "data"
    
    def extract_metric_name(self, text: str) -> List[Tuple[str, str]]:
        """Extract metric names from text with their IDs."""
        found_metrics = []
        
        for metric_id, patterns in self.metric_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    # Find the full phrase
                    match = pattern.search(text)
                    found_metrics.append((metric_id, match.group(0)))
                    break
        
        return found_metrics
    
    def extract_year(self, text: str) -> Optional[str]:
        """Extract academic year from text."""
        # Pattern: 2022-23, 2016-17, etc.
        year_pattern = re.compile(r'(\d{4})-?(\d{2})')
        match = year_pattern.search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        # Pattern: 2022, 2016, etc.
        year_pattern = re.compile(r'\b(20\d{2})\b')
        match = year_pattern.search(text)
        if match:
            return match.group(1)
        
        return None
    
    def extract_district(self, text: str) -> Optional[str]:
        """Extract district name from text."""
        for district in self.ap_districts:
            if district.lower() in text.lower():
                return district
        return None
    
    def extract_category(self, text: str) -> Optional[str]:
        """Extract social/gender category from text."""
        # Check social categories
        for category in self.social_categories:
            if re.search(r'\b' + re.escape(category) + r'\b', text, re.IGNORECASE):
                return category
        
        # Check gender categories
        for category in self.gender_categories:
            if category.lower() in text.lower():
                return f"Gender:{category}"
        
        # Check location categories
        for category in self.location_categories:
            if category.lower() in text.lower():
                return f"Location:{category}"
        
        return None
    
    def extract_numeric_value(self, text: str, context: str = "") -> Optional[Dict]:
        """Extract numeric value with unit from text."""
        for pattern in self.number_patterns:
            match = pattern.search(text)
            if match:
                # Check if it's a ratio (28:1 format)
                if ':' in match.group(0):
                    return {
                        "value": match.group(0),
                        "type": "ratio"
                    }
                
                # Regular number
                num_str = match.group(1).replace(',', '')
                try:
                    value = float(num_str)
                    
                    # Detect unit
                    unit = None
                    if 'crore' in text.lower():
                        unit = "crore"
                        value = value * 10000000
                    elif 'lakh' in text.lower():
                        unit = "lakh"
                        value = value * 100000
                    elif '%' in match.group(0):
                        unit = "percentage"
                    elif ':' in context:
                        unit = "ratio"
                    
                    return {
                        "raw_value": num_str,
                        "value": value,
                        "unit": unit
                    }
                except ValueError:
                    continue
        
        return None
    
    def extract_data_points(self, chunks: List[Dict]) -> List[Dict]:
        """Extract data points from chunks."""
        data_points = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id")
            
            # Extract metrics mentioned
            metrics = self.extract_metric_name(text)
            
            # Extract year
            year = self.extract_year(text)
            
            # Extract district
            district = self.extract_district(text)
            
            # Extract category
            category = self.extract_category(text)
            
            # Extract numeric values
            # Split text into sentences to get context
            sentences = re.split(r'[.!?\n]+', text)
            
            for sentence in sentences:
                # Check if sentence contains any metric
                for metric_id, metric_name in metrics:
                    if metric_name.lower() in sentence.lower():
                        # Extract number from this sentence
                        value_info = self.extract_numeric_value(sentence, text)
                        
                        if value_info:
                            data_point = {
                                "metric_id": metric_id,
                                "metric_name": metric_name,
                                "value": value_info.get("value"),
                                "raw_value": value_info.get("raw_value", ""),
                                "unit": value_info.get("unit"),
                                "year": year,
                                "district": district or "All",
                                "category": category or "Total",
                                "source_chunk": chunk_id,
                                "context": sentence.strip()
                            }
                            data_points.append(data_point)
        
        return data_points
    
    def build_metrics_catalog(self, data_points: List[Dict]) -> Dict:
        """Build catalog of all metrics with their data points."""
        metrics_catalog = defaultdict(lambda: {
            "metric_id": "",
            "full_name": "",
            "unit": "number",
            "data_points": []
        })
        
        for point in data_points:
            metric_id = point["metric_id"]
            
            if not metrics_catalog[metric_id]["metric_id"]:
                metrics_catalog[metric_id]["metric_id"] = metric_id
                metrics_catalog[metric_id]["full_name"] = point["metric_name"]
                metrics_catalog[metric_id]["unit"] = point.get("unit", "number")
            
            metrics_catalog[metric_id]["data_points"].append(point)
        
        return dict(metrics_catalog)
    
    def build_time_series(self, metrics_catalog: Dict) -> Dict:
        """Build time series for each metric."""
        time_series = {}
        
        for metric_id, metric_data in metrics_catalog.items():
            series = defaultdict(float)
            
            for point in metric_data["data_points"]:
                year = point.get("year")
                value = point.get("value")
                
                if year and value:
                    # Aggregate by year (sum or average depending on metric)
                    if metric_id in ["enrollment"]:
                        series[year] += value
                    else:
                        # For rates/ratios, take average
                        if year not in series:
                            series[year] = []
                        series[year].append(value)
            
            # Calculate averages for rate metrics
            for year, values in series.items():
                if isinstance(values, list):
                    series[year] = sum(values) / len(values)
            
            time_series[metric_id] = dict(sorted(series.items()))
        
        return time_series
    
    def build_district_breakdown(self, data_points: List[Dict]) -> Dict:
        """Build district-wise breakdown of metrics."""
        district_breakdown = defaultdict(lambda: defaultdict(float))
        
        for point in data_points:
            district = point.get("district", "All")
            metric_id = point.get("metric_id")
            value = point.get("value")
            
            if district != "All" and value:
                district_breakdown[district][metric_id] = value
        
        return dict(district_breakdown)
    
    def build_category_breakdown(self, data_points: List[Dict]) -> Dict:
        """Build category-wise breakdown of metrics."""
        category_breakdown = defaultdict(lambda: defaultdict(float))
        
        for point in data_points:
            category = point.get("category", "Total")
            metric_id = point.get("metric_id")
            value = point.get("value")
            
            if category != "Total" and value:
                category_breakdown[category][metric_id] = value
        
        return dict(category_breakdown)
    
    def build_database(self) -> Dict:
        """
        Build complete data/metrics database.
        
        Returns:
            Complete data database structure
        """
        # Load processed data
        chunks = self.load_processed_chunks()
        
        if not chunks:
            logger.error("No chunks loaded for data database building")
            return {}
        
        # Filter for data report documents
        data_chunks = self.filter_chunks_by_doc_type(chunks, ["data_report", "statistics"])
        
        if not data_chunks:
            logger.warning("No data report chunks found")
            return {}
        
        # Extract data points
        data_points = self.extract_data_points(data_chunks)
        
        if not data_points:
            logger.warning("No data points extracted")
            return {}
        
        # Build metrics catalog
        metrics_catalog = self.build_metrics_catalog(data_points)
        
        # Build time series
        time_series = self.build_time_series(metrics_catalog)
        
        # Build district breakdown
        district_breakdown = self.build_district_breakdown(data_points)
        
        # Build category breakdown
        category_breakdown = self.build_category_breakdown(data_points)
        
        # Update stats
        self.data_stats["metrics_cataloged"] = len(metrics_catalog)
        self.data_stats["data_points_extracted"] = len(data_points)
        self.data_stats["time_periods_covered"] = len(set(p.get("year") for p in data_points if p.get("year")))
        self.data_stats["districts_covered"] = len(district_breakdown)
        
        # Create comprehensive data database
        data_database = {
            "metadata": {
                "database_type": "data_metrics",
                "creation_date": datetime.now().isoformat(),
                "builder_version": "1.0.0",
                "total_metrics": len(metrics_catalog),
                "total_data_points": len(data_points),
                "statistics": self.data_stats
            },
            "metrics_catalog": metrics_catalog,
            "time_series": time_series,
            "district_breakdown": district_breakdown,
            "category_breakdown": category_breakdown,
            "all_data_points": data_points
        }
        
        # Update stats
        self.stats["database_entries_created"] = len(data_points)
        
        return data_database


# Testing
if __name__ == "__main__":
    builder = DataDatabaseBuilder()
    
    print("Testing Data Database Builder...")
    
    # Test metric extraction
    test_text = "The pupil-teacher ratio (PTR) in 2022-23 was 28:1."
    metrics = builder.extract_metric_name(test_text)
    print(f"Metrics found: {metrics}")
    
    # Test year extraction
    year = builder.extract_year(test_text)
    print(f"Year: {year}")
    
    # Test value extraction
    value = builder.extract_numeric_value("28:1", test_text)
    print(f"Value: {value}")
    
    print("\nData Database Builder ready!")