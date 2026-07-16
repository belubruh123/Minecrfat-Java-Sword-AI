#!/bin/bash
# ============================================================================
#  Train the full DRL sword-PvP bot FROM ZERO — one command.
# ============================================================================
# Runs the 4-stage curriculum end to end, manages the Minecraft server for you
# (corruption-safe reboots between segments), warm-starts each stage from the
# previous one, and ships the finished weights into models/:
#
#     1. aim  (horizontal)   ─┐
#     2a. aim (vertical intro)│
#     2b. aim (full vertical) ├─► models/aim.pt
#     3. combo school (fighter, learns to combo vs a walker)
#     4. mortal fight (fighter vs a real, fighting opponent) ─► models/fighter2.pt
#
# The pilot then auto-uses your models/fighter2.pt + models/aim.pt (press G
# in game — see SETUP.md). No manual checkpoint juggling.
#
# ── USAGE ───────────────────────────────────────────────────────────────────
#   scripts/train_all.sh              # 'full' preset (best model, ~12h on CPU)
#   scripts/train_all.sh quick        # decent model, ~4-5h
#   scripts/train_all.sh smoke        # tiny — just proves the pipeline runs
#
#   # or set your own budgets (steps; 1 step = 1 tick x 8 arenas):
#   AIM_STEPS=1000000 VERT_STEPS=1500000 COMBO_STEPS=1500000 FIGHT_STEPS=250000 \
#       scripts/train_all.sh
#
#   SEG=40000   # steps per server reboot (leave as-is unless you know why)
#
# Interrupted? Just run the same command again — each stage resumes from its
# own runs/fz_*/latest.pt. To start a stage over, delete its runs/fz_* dir.
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

PRESET="${1:-full}"
case "$PRESET" in
  smoke) : "${AIM_STEPS:=40000}";   : "${VERT_STEPS:=40000}"
         : "${COMBO_STEPS:=40000}"; : "${FIGHT_STEPS:=40000}"; : "${THEO_STEPS:=40000}" ;;
  quick) : "${AIM_STEPS:=800000}";  : "${VERT_INTRO_STEPS:=800000}"; : "${VERT_STEPS:=800000}"
         : "${COMBO_STEPS:=1000000}"; : "${FIGHT_STEPS:=150000}"; : "${THEO_STEPS:=150000}" ;;
  full)  : "${AIM_STEPS:=3000000}"; : "${VERT_INTRO_STEPS:=800000}"; : "${VERT_STEPS:=3000000}"
         : "${COMBO_STEPS:=2000000}"; : "${FIGHT_STEPS:=300000}"; : "${THEO_STEPS:=400000}" ;;
  *) echo "unknown preset '$PRESET' (use: smoke | quick | full)" >&2; exit 1 ;;
esac
SEG="${SEG:-40000}"

# ── preflight ────────────────────────────────────────────────────────────────
[ -x .venv/bin/python ] || { echo "!! no .venv — see SETUP.md (create the venv first)" >&2; exit 1; }
for c in stage1_aim stage2a_vertical_intro stage2b_vertical_full stage7a_comboschool stage7b_chain stage8_theobald; do
  [ -f "trainer/configs/$c.yaml" ] || { echo "!! missing trainer/configs/$c.yaml" >&2; exit 1; }
done

best_or_final() { [ -f "runs/$1/best.pt" ] && echo "runs/$1/best.pt" || echo "runs/$1/final.pt"; }
ship() {  # ship <src> <models/dst.pt>  — backs up any existing model first
  [ -f "$1" ] || { echo "!! missing $1 — cannot ship $2" >&2; return 1; }
  [ -f "$2" ] && cp "$2" "${2%.pt}_prev.pt" && echo "   (backed up old $2 -> ${2%.pt}_prev.pt)"
  cp "$1" "$2" && echo ">> SHIPPED  $1  ->  $2"
}
stage() { echo; echo "############ $* ############"; }
# Flip a stage to "running" in runs/ladder.json so the dashboard reflects the
# real pipeline position (it only auto-marks "done" when a run hits its target).
mark_running() {
  .venv/bin/python - "$1" <<'PY' 2>/dev/null || true
import json, os, sys
p = "runs/ladder.json"
if os.path.exists(p):
    d = json.load(open(p))
    for s in d.get("stages", []):
        if s.get("run") == sys.argv[1] and s.get("status") != "done":
            s["status"] = "running"
    json.dump(d, open(p, "w"), indent=2)
PY
}

echo "=================================================================="
echo "  TRAIN FROM ZERO   preset=$PRESET   (seg=$SEG steps/reboot)"
echo "  budgets: aim=$AIM_STEPS vert=$VERT_STEPS combo=$COMBO_STEPS"
echo "           fight=$FIGHT_STEPS theobald=$THEO_STEPS"
echo "=================================================================="

stage "1/6  AIM — horizontal, pitch LOCKED (yaw only), from scratch"
mark_running fz_aim
INIT_CKPT=none \
  scripts/train_segmented.sh configs/stage1_aim.yaml fz_aim "$AIM_STEPS" "$SEG" || exit 1

stage "2a/6 AIM — vertical INTRO, tiny elevation, target always in view"
mark_running fz_aim_vert_intro
INIT_CKPT="../$(best_or_final fz_aim)" \
  scripts/train_segmented.sh configs/stage2a_vertical_intro.yaml fz_aim_vert_intro "$VERT_INTRO_STEPS" "$SEG" || exit 1

stage "2b/6 AIM — full vertical, player in front, warm from stage 2a"
mark_running fz_aim_vert
INIT_CKPT="../$(best_or_final fz_aim_vert_intro)" \
  scripts/train_segmented.sh configs/stage2b_vertical_full.yaml fz_aim_vert "$VERT_STEPS" "$SEG" || exit 1
ship "$(best_or_final fz_aim_vert)" models/aim.pt || exit 1

stage "3/6  COMBO SCHOOL — fighter learns to combo (fresh, uses new aim.pt)"
mark_running fz_combo
INIT_CKPT=none AIM_CKPT=../models/aim.pt \
  scripts/train_segmented.sh configs/stage7a_comboschool.yaml fz_combo "$COMBO_STEPS" "$SEG" || exit 1

stage "4/6  MORTAL FIGHT — vs a fighting opponent (warm from combo school)"
mark_running fz_fight
INIT_CKPT="../$(best_or_final fz_combo)" AIM_CKPT=../models/aim.pt \
  scripts/train_segmented.sh configs/stage7b_chain.yaml fz_fight "$FIGHT_STEPS" "$SEG" || exit 1
ship "$(best_or_final fz_fight)" models/fighter2.pt || exit 1

stage "5/6  THEOBALD — fine-tune vs the TheoBaldTheBird-style bot (FINAL bot)"
mark_running fz_theobald
INIT_CKPT="../$(best_or_final fz_fight)" AIM_CKPT=../models/aim.pt \
  scripts/train_segmented.sh configs/stage8_theobald.yaml fz_theobald "$THEO_STEPS" "$SEG" || exit 1
ship "$(best_or_final fz_theobald)" models/fighter2.pt || exit 1

echo
echo "=================================================================="
echo "  DONE ✓   shipped  models/aim.pt  +  models/fighter2.pt"
echo "  (final fighter2 = hardened vs the TheoBaldTheBird-style bot)"
echo "  The pilot auto-uses them — press G in game (SETUP.md)."
echo
echo "  Check your bot:"
echo "    cd trainer && ../.venv/bin/python eval.py ../runs/fz_theobald/best.pt \\"
echo "        --episodes 30 --curriculum configs/stage8_theobald.yaml \\"
echo "        --aim ../models/aim.pt --stochastic"
echo "=================================================================="
