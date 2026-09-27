# # features.py
# from rapidfuzz import fuzz
# import re

# def extract_pair_features(s1_name, s1_addr, s2_name, s2_addr):
#     # Handle Nulls
#     s1_name = s1_name or ""
#     s1_addr = s1_addr or ""
#     s2_name = s2_name or ""
#     s2_addr = s2_addr or ""
    
#     # Name Similarities
#     name_ratio = fuzz.ratio(s1_name, s2_name)
#     name_partial = fuzz.partial_ratio(s1_name, s2_name)
#     name_token_sort = fuzz.token_sort_ratio(s1_name, s2_name)
#     name_token_set = fuzz.token_set_ratio(s1_name, s2_name)
    
#     # Address Similarities
#     addr_ratio = fuzz.ratio(s1_addr, s2_addr)
#     addr_token_set = fuzz.token_set_ratio(s1_addr, s2_addr)
    
#     # Numeric / House Number Match (Extract digits)
#     s1_nums = set(re.findall(r'\d+', s1_addr))
#     s2_nums = set(re.findall(r'\d+', s2_addr))
#     num_match = 1.0 if s1_nums and s2_nums and (s1_nums & s2_nums) else 0.0
    
#     # Length Difference Features
#     len_diff_name = abs(len(s1_name) - len(s2_name))
#     len_diff_addr = abs(len(s1_addr) - len(s2_addr))
    
#     # Missing Flags
#     missing_addr = 1.0 if not s1_addr or not s2_addr else 0.0
    
#     return [
#         name_ratio,
#         name_partial,
#         name_token_sort,
#         name_token_set,
#         addr_ratio,
#         addr_token_set,
#         num_match,
#         len_diff_name,
#         len_diff_addr,
#         missing_addr
#     ]














# features.py
import re
import numpy as np
from rapidfuzz import fuzz, distance

def extract_house_number(address_str):
    """Extracts leading or primary numerical sequence from address string."""
    if not address_str:
        return None
    match = re.search(r'\b\d+\b', address_str)
    return match.group(0) if match else None

def extract_digits(text):
    """Extracts all digits from a string as a single joined string."""
    if not text:
        return ""
    return "".join(re.findall(r'\d+', text))

def get_token_jaccard(str1, str2):
    """Computes Jaccard similarity on word tokens."""
    if not str1 or not str2:
        return 0.0
    set1, set2 = set(str1.split()), set(str2.split())
    union = len(set1.union(set2))
    return len(set1.intersection(set2)) / union if union > 0 else 0.0

def get_char_ngram_jaccard(str1, str2, n=3):
    """Computes character n-gram Jaccard similarity."""
    if not str1 or not str2 or len(str1) < n or len(str2) < n:
        return 0.0
    ngrams1 = set([str1[i:i+n] for i in range(len(str1) - n + 1)])
    ngrams2 = set([str2[i:i+n] for i in range(len(str2) - n + 1)])
    union = len(ngrams1.union(ngrams2))
    return len(ngrams1.intersection(ngrams2)) / union if union > 0 else 0.0

def build_21_features(name1, name2, addr1, addr2, country1, country2, source_id="s2"):
    """
    Computes a 21-feature similarity vector for a single pair of records.
    Returns a numpy array of shape (21,).
    """
    # Safe handling for missing or None strings
    n1 = (name1 or "").lower().strip()
    n2 = (name2 or "").lower().strip()
    a1 = (addr1 or "").lower().strip()
    a2 = (addr2 or "").lower().strip()
    c1 = (country1 or "").lower().strip()
    c2 = (country2 or "").lower().strip()

    # --- [ Name Similarity ] ---
    f1_n_exact = 1.0 if (n1 and n1 == n2) else 0.0
    f2_n_jw = distance.JaroWinkler.similarity(n1, n2)
    f3_n_lev = fuzz.ratio(n1, n2) / 100.0
    f4_n_jaccard = get_token_jaccard(n1, n2)
    
    # Token Overlap Ratio
    tokens1, tokens2 = set(n1.split()), set(n2.split())
    min_tokens = min(len(tokens1), len(tokens2))
    f5_n_overlap = (len(tokens1.intersection(tokens2)) / min_tokens) if min_tokens > 0 else 0.0
    
    f6_n_ngram = get_char_ngram_jaccard(n1, n2, n=3)
    f7_n_len_diff = abs(len(n1) - len(n2))
    f8_n_prefix = 1.0 if (len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4]) else 0.0

    # --- [ Address Similarity ] ---
    f9_a_exact = 1.0 if (a1 and a1 == a2) else 0.0
    f10_a_jaccard = get_token_jaccard(a1, a2)
    f11_a_lev = fuzz.ratio(a1, a2) / 100.0
    
    # House Number Match
    h1, h2 = extract_house_number(a1), extract_house_number(a2)
    f12_a_house_match = 1.0 if (h1 and h2 and h1 == h2) else 0.0
    
    # Postal Code Match (simple regex extraction)
    p1 = re.search(r'\b\d{5}(?:-\d{4})?\b|\b[a-z]\d[a-z]\s?\d[a-z]\d\b', a1)
    p2 = re.search(r'\b\d{5}(?:-\d{4})?\b|\b[a-z]\d[a-z]\s?\d[a-z]\d\b', a2)
    f13_a_postal_match = 1.0 if (p1 and p2 and p1.group(0) == p2.group(0)) else 0.0
    
    f14_a_city_overlap = get_token_jaccard(a1, a2)  # Token overlap on full address
    f15_a_len_diff = abs(len(a1) - len(a2))

    # --- [ Cross-Fields & Metadata ] ---
    f16_country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    f17_is_s3 = 1.0 if source_id.lower() == "s3" else 0.0
    f18_n_missing = 1.0 if (not n1 or not n2) else 0.0
    f19_a_missing = 1.0 if (not a1 or not a2) else 0.0
    
    # Digit ratio matching (Numerical accuracy across fields)
    d1, d2 = extract_digits(n1 + a1), extract_digits(n2 + a2)
    f20_digit_ratio = fuzz.ratio(d1, d2) / 100.0 if (d1 or d2) else 1.0
    
    # Combined Token Set Ratio
    f21_comb_token_set = fuzz.token_set_ratio(f"{n1} {a1}", f"{n2} {a2}") / 100.0

    return np.array([
        f1_n_exact, f2_n_jw, f3_n_lev, f4_n_jaccard, f5_n_overlap, f6_n_ngram, f7_n_len_diff, f8_n_prefix,
        f9_a_exact, f10_a_jaccard, f11_a_lev, f12_a_house_match, f13_a_postal_match, f14_a_city_overlap, f15_a_len_diff,
        f16_country_match, f17_is_s3, f18_n_missing, f19_a_missing, f20_digit_ratio, f21_comb_token_set
    ], dtype=np.float32)