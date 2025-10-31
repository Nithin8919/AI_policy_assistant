"""Parse dates, academic years"""
from datetime import datetime

def parse_date(date_str: str) -> datetime:
    """Parse date string"""
    try:
        return datetime.fromisoformat(date_str)
    except:
        return None

def parse_academic_year(year_str: str) -> tuple:
    """Parse academic year like '2022-23'"""
    # Implementation
    return (2022, 2023)






