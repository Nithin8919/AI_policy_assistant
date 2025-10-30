"""Populate bridge table"""
import sys
sys.path.append('src')

from src.knowledge_graph.bridge_builder import build_bridge_table
from pathlib import Path

def main():
    """Build bridge table"""
    data_dir = Path("data/processed")
    bridge_dir = Path("data/knowledge_graph")
    
    # Build bridge table
    bridge_data = build_bridge_table(list(data_dir.rglob("*.json")))
    
    # Save
    import json
    with open(bridge_dir / "bridge_table.json", 'w') as f:
        json.dump(bridge_data, f, indent=2)

if __name__ == "__main__":
    main()




