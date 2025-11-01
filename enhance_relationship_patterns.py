#!/usr/bin/env python3
"""
Enhance Relationship Extraction Patterns

This script improves the bridge table relationship patterns
to better match real document content formats.
"""

import os
import sys
import json
import re
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.knowledge_graph.bridge_table_builder import BridgeTableBuilder, RelationshipType, EntityType

def create_enhanced_patterns():
    """Create enhanced patterns for better relationship extraction"""
    
    # More flexible GO reference patterns
    enhanced_go_patterns = [
        r'G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'Government\s+Order.*?No\.?\s*(\d+)',
        r'GO\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'Order\s+No\.?\s*(\d+)',
        r'(?:Ms\.?|Rt\.?)\s*No\.?\s*(\d+)'  # Shorter form
    ]
    
    # Enhanced supersession patterns
    enhanced_supersession_patterns = [
        # Direct supersession
        r'supersedes?\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'in\s+supersession\s+of\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'replaces?\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'cancels?\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        # Order context
        r'(?:this\s+order\s+)?supersedes?\s+(?:the\s+)?(?:order\s+)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'(?:hereby\s+)?supersedes?\s+(?:all\s+)?(?:previous\s+)?(?:orders?\s+)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        # Reference context
        r'(?:vide|ref|reference)\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+).*?(?:is|stands?)\s+(?:superseded|cancelled|replaced)',
        r'(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+).*?(?:is|stands?)\s+(?:hereby\s+)?(?:superseded|cancelled|replaced)'
    ]
    
    # Enhanced reference patterns with more flexible GO matching
    enhanced_reference_patterns = [
        r'(?:refer|see|vide|ref)\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'as\s+per\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'in\s+accordance\s+with\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'pursuant\s+to\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'under\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'subject\s+to\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)'
    ]
    
    # Enhanced section patterns
    enhanced_section_patterns = [
        r'Section\s+(\d+(?:\([a-z0-9]\))*(?:\([ivx]+\))*)',
        r'Sec\.\s*(\d+(?:\([a-z0-9]\))*)',
        r'Article\s+(\d+[A-Z]*)',
        r'Rule\s+(\d+(?:\([a-z0-9]\))*)',
        r'Chapter\s+(\d+)',
        r'Para\s+(\d+(?:\.\d+)*)',
        r'Clause\s+(\d+(?:\([a-z]\))*)'
    ]
    
    # Enhanced scheme patterns for AP schemes
    enhanced_scheme_patterns = [
        r'(Nadu[\\-\\s]?Nedu)',
        r'(Amma\\s+Vodi)',
        r'(Jagananna\\s+[A-Za-z\\s]+)',
        r'(Mid[\\-\\s]?Day[\\-\\s]?Meal)',
        r'(Sarva\\s+Shiksha\\s+Abhiyan)',
        r'(SSA)',
        r'(RMSA)',
        r'(Samagra\\s+Shiksha)',
        r'(PM\\s+POSHAN)',
        r'(Gorumudda)',
        r'(Vidya\\s+Volunteers?)',
        r'(SMC)'  # School Management Committee
    ]
    
    return {
        'entity_patterns': {
            EntityType.GOVERNMENT_ORDER: enhanced_go_patterns,
            EntityType.LEGAL_SECTION: enhanced_section_patterns,
            EntityType.SCHEME: enhanced_scheme_patterns,
        },
        'relationship_patterns': {
            RelationshipType.SUPERSEDES: enhanced_supersession_patterns,
            RelationshipType.REFERENCES: enhanced_reference_patterns,
        }
    }

def patch_bridge_table_builder():
    """Monkey patch the BridgeTableBuilder with enhanced patterns"""
    
    enhanced = create_enhanced_patterns()
    
    # Update entity patterns
    for entity_type, patterns in enhanced['entity_patterns'].items():
        BridgeTableBuilder.ENTITY_PATTERNS[entity_type] = patterns
    
    # Update relationship patterns  
    for rel_type, patterns in enhanced['relationship_patterns'].items():
        BridgeTableBuilder.RELATIONSHIP_PATTERNS[rel_type] = patterns
    
    print("✅ Enhanced patterns applied to BridgeTableBuilder")

def test_pattern_matching():
    """Test the enhanced patterns against sample content"""
    
    print("🧪 Testing enhanced patterns...")
    
    # Sample content for testing
    test_content = [
        "This order supersedes G.O.Ms.No.54 dated 15/03/2023.",
        "In supersession of GO Ms No 123, this order is issued.",
        "As per G.O.Rt.No.456 dated 10/01/2022, the following is ordered.",
        "Vide GO Ms No 789, the Nadu-Nedu scheme is implemented.", 
        "Section 12(1)(c) of the RTE Act read with Section 15.",
        "This replaces Order No.321 and cancels Ms.No.654.",
        "Under the Amma Vodi scheme as per Rt.No.999.",
        "Reference GO Ms No 111 stands superseded by this order."
    ]
    
    enhanced = create_enhanced_patterns()
    
    for content in test_content:
        print(f"\nTesting: {content}")
        
        # Test GO patterns
        for pattern in enhanced['entity_patterns'][EntityType.GOVERNMENT_ORDER]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"  GO matches: {matches} (pattern: {pattern[:50]}...)")
        
        # Test supersession patterns
        for pattern in enhanced['relationship_patterns'][RelationshipType.SUPERSEDES]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"  Supersession matches: {matches} (pattern: {pattern[:50]}...)")
        
        # Test reference patterns
        for pattern in enhanced['relationship_patterns'][RelationshipType.REFERENCES]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"  Reference matches: {matches} (pattern: {pattern[:50]}...)")

def main():
    """Main function"""
    
    print("=" * 80)
    print("🔧 ENHANCING RELATIONSHIP EXTRACTION PATTERNS")
    print("=" * 80)
    
    # Test patterns first
    test_pattern_matching()
    
    # Apply enhancements
    patch_bridge_table_builder()
    
    print("\n✅ Pattern enhancement complete!")
    print("Now run rebuild_bridge_table.py to rebuild with enhanced patterns.")

if __name__ == "__main__":
    main()