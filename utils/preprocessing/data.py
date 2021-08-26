from ast import literal_eval
from typing import List, Any


# trim the number from the ICPC code and return the alphabetical categories
def extract_icpc_categories(codes: Any, remove_admin=True) -> List[str]:
    codes = literal_eval(codes)
    codes = [c[0].strip().upper() for c in codes]
    if remove_admin:
        codes = [c for c in codes if c != '-']
    return codes
