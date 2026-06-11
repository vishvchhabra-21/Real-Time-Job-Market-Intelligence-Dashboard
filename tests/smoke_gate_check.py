"""Throwaway smoke check: gate renders first, profile unlocks the dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

APP_PATH = str(ROOT / "dashboard" / "app.py")

# 1. First load: no profile -> the calibration gate must render, no tabs/sidebar widgets.
at = AppTest.from_file(APP_PATH, default_timeout=180)
at.run()
assert not at.exception, f"Gate run raised: {at.exception}"
assert len(at.tabs) == 0, "Tabs must stay locked before a profile exists"
gate_buttons = [b.label for b in at.button]
assert any("Unlock" in label for label in gate_buttons), f"Gate buttons missing: {gate_buttons}"

# 2. Unlock via the skills path.
at.multiselect("gate_skill_select").set_value(["Python", "SQL"])
unlock = next(b for b in at.button if b.label == "Unlock with skills")
unlock.click()
at.run()
assert not at.exception, f"Unlocked run raised: {at.exception}"
assert len(at.tabs) == 4, f"Expected 4 tabs after unlock, got {len(at.tabs)}"
assert at.session_state["candidate_profile"]["skills"] == ["Python", "SQL"]
assert "Python" in at.session_state["resume_text_input"]

# 3. Recalibrate returns to the gate.
recal = next(b for b in at.button if b.label == "Recalibrate")
recal.click()
at.run()
assert not at.exception, f"Recalibrate run raised: {at.exception}"
assert len(at.tabs) == 0, "Recalibrate must re-lock the dashboard"

print("Gate smoke check passed: locked -> unlocked -> re-locked.")
