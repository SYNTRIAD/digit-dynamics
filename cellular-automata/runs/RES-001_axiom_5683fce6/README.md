# RES-001 — Original CA Research Session (axiom_research_5683fce6)

**Status:** Superseded by RES-003 (axiom_research_baf19226)

This was the original session that first identified Rule 4 → A000204 and
Rule 76 → A001644. It was run with a sparse k-range [3,4,5,6,7,8,10,12]
and contained three bugs that prevented automated OEIS detection:

- OEISChecker crash on `.terms` attribute
- Gap artifacts from non-continuous k-range
- Flawed exponential detection tolerance

OEIS matches were identified manually in this session.
RES-003 is the corrected rerun with continuous k=1..14 and all fixes applied.

Session artifacts are stored in the AXIOM research repository (internal).
Certificate signature: (see AXIOM experiment store, session axiom_research_5683fce6)
