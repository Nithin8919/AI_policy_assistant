"""
Vertical Database Builders for AP Education Policy System.

Transforms processed chunks and entities into specialized vertical databases
optimized for domain-specific agents and queries.

Vertical Databases:
- Legal: Acts, Rules, Sections with cross-references
- Government Orders: Active GOs with supersession chains  
- Judicial: Case law with precedent relationships
- Data Reports: Metrics catalog with time series
- Schemes: Implementation tracking and budget allocation

Each vertical provides structured access optimized for its domain,
enabling specialized agents to answer precise queries.
"""

from .base_builder import BaseVerticalBuilder
from .legal_builder import LegalDatabaseBuilder
from .go_builder import GODatabaseBuilder
from .judicial_builder import JudicialDatabaseBuilder
from .data_builder import DataDatabaseBuilder
from .schema_builder import SchemeDatabaseBuilder

__version__ = "1.0.0"

__all__ = [
    "BaseVerticalBuilder",
    "LegalDatabaseBuilder", 
    "GODatabaseBuilder",
    "JudicialDatabaseBuilder",
    "DataDatabaseBuilder", 
    "SchemeDatabaseBuilder"
]