#!/usr/bin/env python3
"""
Verify that the vendored prompts in cabot_public_lib/cabot_prompts.py still match, byte-for-byte,
the source they were extracted from in the parent NEJMBench repository.

cabot_prompts.py is the hand-maintained source of truth — edit it directly. This script is just a
one-way provenance check: it confirms the prompts that were vendored from git history still match
their pinned commits. It only checks the prompts listed in the mapping below; any custom prompts you
add later are simply ignored, so adding them never breaks this check.

It must be run from inside a checkout of the NEJMBench git repo (it uses `git show <commit>:<path>`).
Once CaBot-Public is split into its own repo this check no longer applies.

Mapping (version -> commit):
  v1        text DDx + standard presentation   -> f5a08a76
  v1.1      presentation (missing-info, Nov)   -> c09c6952   (text DDx prompt is AUTHORED, not vendored)
  vr1       text DDx (composed) + presentation -> 4fe60742
  vs1/vs1.1 simple QA literature system prompt -> f404f0dd   (cabot2_old.py LIT_SEARCH_SYSTEM)
"""
import os
import re
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from cabot_public_lib import cabot_prompts as P  # noqa: E402


def show(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"]).decode()


def grab(src, name):
    # Return the original RUNTIME value (escapes interpreted as the source module did), so we
    # compare what the model actually received — not raw source text. Comparing source text would
    # mask backslash-doubling bugs, since "\\textwidth" reads identically in both representations.
    m = re.search(rf'{name} = """(.*?)"""', src, re.S)
    if not m:
        raise SystemExit(f"could not find {name}")
    ns = {}
    exec(m.group(0), ns)
    return ns[name]


def main():
    mono = show("f5a08a76", "cabot/cabot2.py")
    modu = show("4fe60742", "cabot/cabot2.py")
    vnov = show("c09c6952", "cabot/cabot_video.py")
    vudn = show("4fe60742", "cabot/cabot_video.py")
    vv1 = show("f5a08a76", "cabot/cabot_video.py")
    vqa = show("f404f0dd", "cabot/cabot2_old.py")

    # vr1 text DDx is composed exactly as cabot2.py@4fe60742 does for agent=True, use_similar_cases=False
    vr1 = (grab(modu, "PROMPT_INTRO") + grab(modu, "PROMPT_LITERATURE_TOOL")
           + grab(modu, "PROMPT_AGENT_INSTRUCTIONS")
           + grab(modu, "PROMPT_FORMATTING").replace("{examples_ref}", "").replace("{examples_mirror}", "")
           + grab(modu, "PROMPT_CASE"))

    checks = [
        ("v1 text DDx (verbatim)", P.V1_DDX_PROMPT == grab(mono, "CABOT_PROMPT_LARGE")),
        ("v1 text DDx non-agent (verbatim)", P.V1_DDX_PROMPT_NONAGENT == grab(mono, "CABOT_PROMPT_NON_AGENT")),
        ("vr1 text DDx (composed)", P.VR1_DDX_PROMPT == vr1),
        ("v1.1 is v1 + missing-info block (authored)", "HANDLING MISSING OR INCOMPLETE INFORMATION" in P.V1_1_DDX_PROMPT),
        ("v1 standard presentation (verbatim)", P.V1_PRESENTATION_PROMPT == grab(vv1, "PRESENTATION_PROMPT")),
        ("with-DDX prefix (verbatim)", P.PRESENTATION_WITH_DDX_PREFIX == grab(vnov, "PRESENTATION_PROMPT_WITH_DDX")),
        ("without-DDX prefix (verbatim)", P.PRESENTATION_WITHOUT_DDX_PREFIX == grab(vnov, "PRESENTATION_PROMPT_WITHOUT_DDX")),
        ("v1.1 presentation body (verbatim)", P.PRESENTATION_BODY_MISSING_INFO_NOV == grab(vnov, "PRESENTATION_PROMPT_BODY")),
        ("vr1 presentation body (verbatim)", P.PRESENTATION_BODY_MISSING_INFO_UDN == grab(vudn, "PRESENTATION_PROMPT_BODY")),
        ("vs1/vs1.1 simple QA system prompt (verbatim)", P.SIMPLE_QA_SYSTEM_PROMPT == grab(vqa, "LIT_SEARCH_SYSTEM")),
    ]
    ok = True
    for name, passed in checks:
        print(("PASS" if passed else "FAIL"), name)
        ok = ok and passed
    if not ok:
        sys.exit("FIDELITY CHECK FAILED")
    print("\nAll prompts match their pinned source commits.")


if __name__ == "__main__":
    main()
