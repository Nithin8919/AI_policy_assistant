"""Fast path via bridge table"""
import json
from typing import Dict

def lookup_bridge_table(topic: str, bridge_file: str) -> List[str]:
    """Lookup sources for a topic in bridge table"""
    with open(bridge_file, 'r') as f:
        bridge_data = json.load(f)
    
    return bridge_data.get("topics", {}).get(topic, [])






