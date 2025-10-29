"""
Scheme Database Builder for Implementation Tracking.

Creates a structured scheme database with:
- Scheme metadata (objectives, eligibility)
- Budget allocation tracking
- Beneficiary statistics
- Implementation timeline
- District-wise rollout
- Governance structure
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime

from src.vertical_builders.base_builder import BaseVerticalBuilder, extract_number_from_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SchemeDatabaseBuilder(BaseVerticalBuilder):
    """
    Build specialized database for scheme implementation.
    
    Creates structured access to scheme information with budget tracking,
    beneficiary statistics, and implementation monitoring.
    """
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize scheme database builder."""
        super().__init__(data_dir, output_dir)
        
        # Initialize scheme patterns
        self._init_scheme_patterns()
        
        # Initialize AP schemes list
        self._init_ap_schemes()
        
        # Statistics
        self.scheme_stats = {
            "schemes_processed": 0,
            "budget_entries": 0,
            "beneficiary_records": 0,
            "districts_covered": 0
        }
    
    def _init_scheme_patterns(self):
        """Initialize patterns for scheme extraction."""
        # Budget amount patterns
        self.budget_patterns = [
            re.compile(
                r'(?:Rs|INR|₹)\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|thousand)',
                re.IGNORECASE
            ),
            re.compile(
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|thousand)\s*(?:rupees)?',
                re.IGNORECASE
            )
        ]
        
        # Beneficiary patterns
        self.beneficiary_patterns = [
            re.compile(r'(\d+(?:,\d+)*)\s*(?:students|children|schools|teachers)', re.IGNORECASE),
            re.compile(r'benefiting\s+(\d+(?:,\d+)*)', re.IGNORECASE)
        ]
        
        # Status indicators
        self.status_keywords = {
            "ongoing": ["ongoing", "in progress", "being implemented"],
            "completed": ["completed", "finished", "concluded"],
            "planned": ["planned", "proposed", "upcoming"],
            "sanctioned": ["sanctioned", "approved"]
        }
        
        # Component indicators
        self.component_keywords = [
            "component", "phase", "pillar", "module", "aspect", "feature"
        ]
    
    def _init_ap_schemes(self):
        """Initialize list of AP education schemes."""
        self.ap_schemes = {
            "nadu_nedu": ["Nadu-Nedu", "Nadu Nedu", "Naadu-Nedu"],
            "amma_vodi": ["Amma Vodi", "Ammavodi"],
            "gorumudda": ["Jagananna Gorumudda", "Gorumudda", "Mid Day Meal"],
            "vidya_deevena": ["Jagananna Vidya Deevena", "Vidya Deevena"],
            "vidya_kanuka": ["Jagananna Vidya Kanuka", "Vidya Kanuka"],
            "vasathi_deevena": ["Vasathi Deevena"],
            "smart_schools": ["Smart School Programme", "Smart Schools"]
        }
        
        # Flatten to list for matching
        self.scheme_names = []
        for scheme_id, variations in self.ap_schemes.items():
            self.scheme_names.extend(variations)
    
    def get_vertical_name(self) -> str:
        """Get vertical name for output directory."""
        return "schemes"
    
    def identify_scheme(self, text: str) -> Optional[Tuple[str, str]]:
        """Identify scheme from text. Returns (scheme_id, scheme_name)."""
        for scheme_id, variations in self.ap_schemes.items():
            for variation in variations:
                if variation.lower() in text.lower():
                    return (scheme_id, variation)
        return None
    
    def extract_objective(self, text: str) -> Optional[str]:
        """Extract scheme objective from text."""
        # Look for objective indicators
        objective_patterns = [
            r'objective[s]?\s+(?:is|are|of)\s+([^.]+)',
            r'aims?\s+(?:to|at)\s+([^.]+)',
            r'intended\s+to\s+([^.]+)'
        ]
        
        for pattern in objective_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_eligibility(self, text: str) -> List[str]:
        """Extract eligibility criteria from text."""
        eligibility = []
        
        # Look for eligibility indicators
        eligibility_patterns = [
            r'eligible\s+(?:for|students?|schools?)[:\s]+([^.]+)',
            r'applicable\s+to\s+([^.]+)',
            r'benefits?\s+(?:to|for)\s+([^.]+)'
        ]
        
        for pattern in eligibility_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                eligibility.append(match.group(1).strip())
        
        return eligibility
    
    def extract_budget_info(self, text: str) -> Dict:
        """Extract budget information from text."""
        budget_info = {
            "total_allocation": None,
            "year_wise": {},
            "expenditure": {}
        }
        
        # Extract budget amounts
        for pattern in self.budget_patterns:
            for match in pattern.finditer(text):
                amount_str = match.group(1).replace(',', '')
                unit = match.group(2).lower()
                
                try:
                    amount = float(amount_str)
                    
                    # Convert to crores
                    if unit == "lakh":
                        amount = amount / 100
                    elif unit == "thousand":
                        amount = amount / 10000
                    
                    # Try to determine if it's allocation or expenditure
                    context_before = text[max(0, match.start()-100):match.start()]
                    context_after = text[match.end():min(len(text), match.end()+100)]
                    context = context_before + match.group(0) + context_after
                    
                    # Look for year in context
                    year_match = re.search(r'(\d{4})-?(\d{2})?', context)
                    if year_match:
                        year = year_match.group(1) + "-" + year_match.group(2) if year_match.group(2) else year_match.group(1)
                        
                        if any(word in context.lower() for word in ["allocated", "allocation", "sanctioned"]):
                            budget_info["year_wise"][year] = amount
                        elif any(word in context.lower() for word in ["expenditure", "spent", "utilized"]):
                            budget_info["expenditure"][year] = amount
                    else:
                        # No year specified, might be total
                        if "total" in context.lower():
                            budget_info["total_allocation"] = amount
                
                except ValueError:
                    continue
        
        return budget_info
    
    def extract_beneficiary_info(self, text: str) -> Dict:
        """Extract beneficiary information from text."""
        beneficiary_info = {
            "schools_covered": None,
            "students_benefited": None,
            "teachers_covered": None
        }
        
        for pattern in self.beneficiary_patterns:
            for match in pattern.finditer(text):
                count_str = match.group(1).replace(',', '')
                try:
                    count = int(count_str)
                    
                    # Determine type from context
                    context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    
                    if "school" in context.lower():
                        beneficiary_info["schools_covered"] = count
                    elif "student" in context.lower() or "children" in context.lower():
                        beneficiary_info["students_benefited"] = count
                    elif "teacher" in context.lower():
                        beneficiary_info["teachers_covered"] = count
                
                except ValueError:
                    continue
        
        return beneficiary_info
    
    def extract_implementing_agency(self, text: str) -> Optional[str]:
        """Extract implementing agency from text."""
        agency_patterns = [
            r'implemented by\s+([^.]+)',
            r'implementing agency[:\s]+([^.]+)',
            r'(?:under|by)\s+(?:the\s+)?(School Education Department|Finance Department|[A-Z][A-Za-z\s]+Department)'
        ]
        
        for pattern in agency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def determine_status(self, text: str) -> str:
        """Determine scheme status from text."""
        text_lower = text.lower()
        
        for status, keywords in self.status_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return status
        
        return "unknown"
    
    def extract_components(self, text: str) -> List[str]:
        """Extract scheme components from text."""
        components = []
        
        # Look for numbered lists or bullet points
        lines = text.split('\n')
        for line in lines:
            # Check if line starts with number or bullet
            if re.match(r'^\s*(?:\d+[\.\)]\s*|[-•]\s*)', line):
                # Remove numbering/bullets
                component = re.sub(r'^\s*(?:\d+[\.\)]\s*|[-•]\s*)', '', line).strip()
                if len(component) > 10 and len(component) < 200:  # Reasonable length
                    components.append(component)
        
        return components
    
    def extract_district_rollout(self, chunks: List[Dict]) -> Dict:
        """Extract district-wise implementation info."""
        district_rollout = {}
        
        districts = [
            "Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna",
            "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam",
            "Vizianagaram", "West Godavari", "YSR Kadapa"
        ]
        
        for chunk in chunks:
            text = chunk.get("text", "")
            
            for district in districts:
                if district.lower() in text.lower():
                    if district not in district_rollout:
                        district_rollout[district] = {
                            "mentioned": True,
                            "details": []
                        }
                    
                    # Extract context about this district
                    # Find sentences mentioning the district
                    sentences = re.split(r'[.!?\n]+', text)
                    for sentence in sentences:
                        if district.lower() in sentence.lower():
                            district_rollout[district]["details"].append(sentence.strip())
        
        return district_rollout
    
    def build_scheme_entry(self, doc_chunks: List[Dict], doc_id: str, scheme_id: str, scheme_name: str) -> Dict:
        """Build a single scheme entry from chunks."""
        # Combine text from all chunks
        combined_text = " ".join(chunk.get("text", "") for chunk in doc_chunks)
        
        # Extract scheme information
        objective = self.extract_objective(combined_text)
        eligibility = self.extract_eligibility(combined_text)
        budget_info = self.extract_budget_info(combined_text)
        beneficiary_info = self.extract_beneficiary_info(combined_text)
        implementing_agency = self.extract_implementing_agency(combined_text)
        status = self.determine_status(combined_text)
        components = self.extract_components(combined_text)
        district_rollout = self.extract_district_rollout(doc_chunks)
        
        # Extract metadata
        metadata = self.extract_metadata_from_chunks(doc_chunks)
        
        # Find related GOs and Acts from relations
        related_gos = []
        legal_basis = []
        
        for chunk in doc_chunks:
            entities = chunk.get("entities", {})
            related_gos.extend(entities.get("go_refs", []))
            legal_basis.extend(entities.get("legal_refs", []))
        
        # Build scheme entry
        scheme_entry = {
            "scheme_id": scheme_id,
            "full_name": scheme_name,
            "doc_id": doc_id,
            "launch_date": metadata.get("year"),
            "objective": objective,
            "eligibility": eligibility,
            "budget": budget_info,
            "beneficiaries": beneficiary_info,
            "implementing_agency": implementing_agency,
            "governing_orders": list(set(related_gos)),
            "legal_basis": list(set(legal_basis)),
            "district_rollout": district_rollout,
            "components": components,
            "status": status,
            "chunks": [chunk.get("chunk_id") for chunk in doc_chunks],
            "metadata": metadata
        }
        
        # Update stats
        self.scheme_stats["schemes_processed"] += 1
        if budget_info["year_wise"]:
            self.scheme_stats["budget_entries"] += len(budget_info["year_wise"])
        if beneficiary_info["students_benefited"]:
            self.scheme_stats["beneficiary_records"] += 1
        self.scheme_stats["districts_covered"] += len(district_rollout)
        
        return scheme_entry
    
    def build_budget_summary(self, schemes: Dict) -> Dict:
        """Build overall budget summary across schemes."""
        budget_summary = {
            "total_education_budget": 0,
            "scheme_wise": {},
            "year_wise": defaultdict(float)
        }
        
        for scheme_id, scheme_data in schemes.items():
            budget = scheme_data.get("budget", {})
            
            # Scheme-wise total
            if budget.get("total_allocation"):
                budget_summary["scheme_wise"][scheme_id] = budget["total_allocation"]
                budget_summary["total_education_budget"] += budget["total_allocation"]
            
            # Year-wise aggregation
            for year, amount in budget.get("year_wise", {}).items():
                budget_summary["year_wise"][year] += amount
        
        budget_summary["year_wise"] = dict(budget_summary["year_wise"])
        
        return budget_summary
    
    def build_database(self) -> Dict:
        """
        Build complete scheme database.
        
        Returns:
            Complete scheme database structure
        """
        # Load processed data
        chunks = self.load_processed_chunks()
        
        if not chunks:
            logger.error("No chunks loaded for scheme database building")
            return {}
        
        # Filter for scheme-related documents
        # Schemes can be in GOs, frameworks, or dedicated scheme documents
        scheme_chunks = self.filter_chunks_by_doc_type(
            chunks,
            ["government_order", "framework", "scheme", "policy"]
        )
        
        if not scheme_chunks:
            logger.warning("No scheme-related chunks found")
            return {}
        
        # Group by document and identify schemes
        doc_chunks = self.group_chunks_by_document(scheme_chunks)
        
        # Build scheme database
        schemes = {}
        
        for doc_id, doc_chunk_list in doc_chunks.items():
            # Check if document mentions any scheme
            combined_text = " ".join(chunk.get("text", "") for chunk in doc_chunk_list)
            
            scheme_info = self.identify_scheme(combined_text)
            if scheme_info:
                scheme_id, scheme_name = scheme_info
                
                # Build scheme entry (or update if already exists)
                if scheme_id not in schemes:
                    scheme_entry = self.build_scheme_entry(doc_chunk_list, doc_id, scheme_id, scheme_name)
                    schemes[scheme_id] = scheme_entry
                else:
                    # Merge information from multiple documents
                    # (same scheme mentioned in multiple GOs)
                    existing = schemes[scheme_id]
                    new_entry = self.build_scheme_entry(doc_chunk_list, doc_id, scheme_id, scheme_name)
                    
                    # Merge GOs and legal basis
                    existing["governing_orders"].extend(new_entry["governing_orders"])
                    existing["governing_orders"] = list(set(existing["governing_orders"]))
                    
                    existing["legal_basis"].extend(new_entry["legal_basis"])
                    existing["legal_basis"] = list(set(existing["legal_basis"]))
        
        # Build budget summary
        budget_summary = self.build_budget_summary(schemes)
        
        # Create comprehensive scheme database
        scheme_database = {
            "metadata": {
                "database_type": "schemes",
                "creation_date": datetime.now().isoformat(),
                "builder_version": "1.0.0",
                "total_schemes": len(schemes),
                "statistics": self.scheme_stats
            },
            "schemes": schemes,
            "budget_summary": budget_summary,
            "status_summary": self._build_status_summary(schemes)
        }
        
        # Update stats
        self.stats["database_entries_created"] = len(schemes)
        
        return scheme_database
    
    def _build_status_summary(self, schemes: Dict) -> Dict:
        """Build summary of scheme statuses."""
        status_summary = Counter()
        for scheme_data in schemes.values():
            status = scheme_data.get("status", "unknown")
            status_summary[status] += 1
        
        return dict(status_summary)


# Testing
if __name__ == "__main__":
    builder = SchemeDatabaseBuilder()
    
    print("Testing Scheme Database Builder...")
    
    # Test scheme identification
    test_text = "The Nadu-Nedu programme was launched with a budget of Rs. 5000 crore."
    scheme_info = builder.identify_scheme(test_text)
    print(f"Scheme identified: {scheme_info}")
    
    # Test budget extraction
    budget = builder.extract_budget_info(test_text)
    print(f"Budget info: {budget}")
    
    # Test beneficiary extraction
    test_text2 = "The scheme benefited 45000 schools and 4.5 lakh students."
    beneficiary = builder.extract_beneficiary_info(test_text2)
    print(f"Beneficiary info: {beneficiary}")
    
    print("\nScheme Database Builder ready!")