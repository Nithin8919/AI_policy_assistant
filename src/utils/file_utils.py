"""Read/write helpers"""
import json

def read_json(filepath: str) -> dict:
    """Read JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(data: dict, filepath: str):
    """Write JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


