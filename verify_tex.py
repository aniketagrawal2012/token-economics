"""
verify_tex.py - Automated LaTeX & Scientific Paper Validator
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Verifies environment balancing, math delimiter parity, citation integrity,
cross-reference resolution, and absence of placeholder tokens.
"""

import re
import sys
import os


def verify_latex(file_path):
    if not os.path.exists(file_path):
        print(f"[FAIL] Target file does not exist: {file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []
    warnings = []

    print(f"Auditing LaTeX Manuscript: {file_path}")
    print("=" * 70)

    # 1. Environment Balancing
    begins = re.findall(r'\\begin\{([a-zA-Z0-9*]+)\}', content)
    ends = re.findall(r'\\end\{([a-zA-Z0-9*]+)\}', content)

    begin_counts = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1
    end_counts = {}
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1

    all_envs = set(begin_counts.keys()).union(set(end_counts.keys()))
    for env in sorted(all_envs):
        bc = begin_counts.get(env, 0)
        ec = end_counts.get(env, 0)
        if bc != ec:
            errors.append(f"Mismatched environment '{env}': {bc} \\begin vs {ec} \\end")
        else:
            print(f"  [PASS] Environment '{env}': {bc} balanced pairs")

    # 2. Inline Math Parity ($)
    clean_dollars = re.sub(r'\\\$', '', content)
    dollar_count = clean_dollars.count('$')
    if dollar_count % 2 != 0:
        errors.append(f"Unbalanced inline math ($): found {dollar_count} occurrences (odd number)")
    else:
        print(f"  [PASS] Inline math ($): {dollar_count} occurrences ({dollar_count // 2} balanced pairs)")

    # 3. Citations and Bibliography Integrity
    citations = set(re.findall(r'\\cite\{([^}]+)\}', content))
    flat_citations = set()
    for c in citations:
        for single in c.split(','):
            flat_citations.add(single.strip())

    bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', content))

    missing_bib = flat_citations - bibitems
    uncited_bib = bibitems - flat_citations

    if missing_bib:
        errors.append(f"Missing bibliography entries for citations: {sorted(missing_bib)}")
    else:
        print(f"  [PASS] Citations matched: all {len(flat_citations)} citations have bibliography entries")

    if uncited_bib:
        warnings.append(f"Uncited bibliography entries: {sorted(uncited_bib)}")
    else:
        print(f"  [PASS] Bibliography clean: all {len(bibitems)} entries cited in text")

    # 4. Cross-Reference Resolution
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', content))
    missing_labels = refs - labels
    if missing_labels:
        errors.append(f"Unresolved \\ref references: {sorted(missing_labels)}")
    else:
        print(f"  [PASS] Cross-references: all {len(refs)} \\ref matched across {len(labels)} labels")

    # 5. Placeholder & Banned Terms Check
    banned_tokens = ["TODO", "TBD", "PLACEHOLDER", "[?]", "FIXME", "XXX"]
    found_banned = []
    for token in banned_tokens:
        if token in content:
            found_banned.append(token)

    if found_banned:
        errors.append(f"Found placeholder tokens: {found_banned}")
    else:
        print("  [PASS] Placeholder check: zero draft tokens detected")

    # 6. Author and Affiliation Check
    if "Aniket Agrawal" in content and "Google" in content:
        print("  [PASS] Author block certified: Aniket Agrawal, Google")
    else:
        errors.append("Author block missing certified inventor details")

    print("-" * 70)
    if warnings:
        print(f"AUDIT WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"AUDIT FAILED WITH {len(errors)} ERRORS:")
        for err in errors:
            print(f"  [ERROR] {err}")
        return False
    else:
        print("AUDIT SUCCESS: 100% DIAMOND QUALITY LATEX MANUSCRIPT CERTIFIED!")
        return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tokenomics_compiler_paper.tex"
    success = verify_latex(target)
    sys.exit(0 if success else 1)
