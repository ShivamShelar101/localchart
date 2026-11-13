import re
from difflib import SequenceMatcher

def item_name(text):
    name = text.split(":", 1)[-1].rsplit(" (status", 1)[0]
    name = re.sub(r"\s*\((disorder|finding|substance|situation|morphologic abnormality|procedure)\)\s*$", "", name)
    return name.strip()

def sim(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b: return 0.0
    if len(a) > 3 and len(b) > 3 and (a in b or b in a): return 1.0
    return SequenceMatcher(None, a, b).ratio()
