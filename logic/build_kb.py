import os
from nltk.sem import Expression
import sys

KB_PATH = os.path.join(os.path.dirname(__file__), "logical-kb.csv")
read_expr = Expression.fromstring

def main():
    if not os.path.exists(KB_PATH):
        print(f"ERROR: {KB_PATH} not found.")
        sys.exit(1)

    seen = set()
    duplicates = []
    failed = []
    facts = []
    with open(KB_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            line_clean = raw.strip()
            if not line_clean or line_clean.startswith("#"):
                continue
            try:
                expr = read_expr(line_clean)
                if line_clean in seen:
                    duplicates.append((i, raw))
                else:
                    seen.add(line_clean)
                    facts.append(expr)
            except Exception as e:
                failed.append((i, raw, str(e)))

    print(f"KB integrity check for: {KB_PATH}")
    print("=" * 50)
    print(f"Total lines (non-empty, non-comment): {len(seen) + len(duplicates) + len(failed)}")
    print(f"  Loaded FOL facts/rules: {len(facts)}")
    print(f"  Duplicates: {len(duplicates)}")
    print(f"  Failed to parse: {len(failed)}")
    print()

    if duplicates:
        print("Duplicate lines:")
        for i, raw in duplicates:
            print(f"  Line {i}: {raw}")
        print()

    if failed:
        print("Failed to parse:")
        for i, raw, err in failed:
            print(f"  Line {i}: {raw}")
            print(f"    Error: {err}")
        print()

    print("Done.")

if __name__ == "__main__":
    main() 