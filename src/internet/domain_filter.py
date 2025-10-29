"""Whitelist enforcement"""

ALLOWED_DOMAINS = [
    "unesco.org",
    "oecd.org",
    "edu-gov.in",
    "ncert.nic.in",
    "ncte.gov.in"
]

def filter_by_domain(urls: list) -> list:
    """Filter URLs by allowed domains"""
    return [url for url in urls if any(domain in url for domain in ALLOWED_DOMAINS)]


