# features.py
from rapidfuzz import fuzz
import re

def extract_pair_features(s1_name, s1_addr, s2_name, s2_addr):
    # Handle Nulls
    s1_name = s1_name or ""
    s1_addr = s1_addr or ""
    s2_name = s2_name or ""
    s2_addr = s2_addr or ""
    
    # Name Similarities
    name_ratio = fuzz.ratio(s1_name, s2_name)
    name_partial = fuzz.partial_ratio(s1_name, s2_name)
    name_token_sort = fuzz.token_sort_ratio(s1_name, s2_name)
    name_token_set = fuzz.token_set_ratio(s1_name, s2_name)
    
    # Address Similarities
    addr_ratio = fuzz.ratio(s1_addr, s2_addr)
    addr_token_set = fuzz.token_set_ratio(s1_addr, s2_addr)
    
    # Numeric / House Number Match (Extract digits)
    s1_nums = set(re.findall(r'\d+', s1_addr))
    s2_nums = set(re.findall(r'\d+', s2_addr))
    num_match = 1.0 if s1_nums and s2_nums and (s1_nums & s2_nums) else 0.0
    
    # Length Difference Features
    len_diff_name = abs(len(s1_name) - len(s2_name))
    len_diff_addr = abs(len(s1_addr) - len(s2_addr))
    
    # Missing Flags
    missing_addr = 1.0 if not s1_addr or not s2_addr else 0.0
    
    return [
        name_ratio,
        name_partial,
        name_token_sort,
        name_token_set,
        addr_ratio,
        addr_token_set,
        num_match,
        len_diff_name,
        len_diff_addr,
        missing_addr
    ]