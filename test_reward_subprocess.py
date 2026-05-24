"""Test if Lean verification works in a subprocess (simulating Ray worker)."""
import os
os.environ["PATH"] = os.path.expanduser("~/.elan/bin") + ":" + os.environ.get("PATH", "")

try:
    from lean_dojo import LeanGitRepo, Theorem, Dojo, TacticState, ProofFinished, LeanError
    repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", "29dcec074de168ac2bf835a77ef68bbe069194c5")
    thm = Theorem(repo, "Mathlib/GroupTheory/PGroup.lean", "IsPGroup.powEquiv_symm_apply")

    with Dojo(thm, timeout=30) as (dojo, init_state):
        r = dojo.run_tac(init_state, "simp [powEquiv]")
        print(f"Result: {type(r).__name__}")
        if isinstance(r, ProofFinished):
            print("REWARD: 1.0")
        elif isinstance(r, TacticState):
            print("REWARD: 1.0 (progress)")
        else:
            print("REWARD: 0.0")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
