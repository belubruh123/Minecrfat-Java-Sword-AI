#!/bin/bash
# ============================================================================
#  Train the full DRL sword-PvP bot FROM ZERO — one command.
# ============================================================================
# Runs the 4-stage curriculum end to end, manages the Minecraft server for you
# (corruption-safe reboots between segments), warm-starts each stage from the
# previous one, and ships the finished weights into models/:
#
#     1. aim  (horizontal)   ─┐
#     2. aim  (vertical)      ├─► models/aim.pt
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
         : "${COMBO_STEPS:=40000}"; : "${FIGHT_STEPS:=40000}" ;;
  quick) : "${AIM_STEPS:=800000}";  : "${VERT_STEPS:=800000}"
         : "${COMBO_STEPS:=1000000}"; : "${FIGHT_STEPS:=150000}" ;;
  full)  : "${AIM_STEPS:=3000000}"; : "${VERT_STEPS:=4000000}"
         : "${COMBO_STEPS:=2000000}"; : "${FIGHT_STEPS:=300000}" ;;
  *) echo "unknown preset '$PRESET' (use: smoke | quick | full)" >&2; exit 1 ;;
esac
SEG="${SEG:-40000}"

# ── preflight ────────────────────────────────────────────────────────────────
[ -x .venv/bin/python ] || { echo "!! no .venv — see SETUP.md (create the venv first)" >&2; exit 1; }
for c in stage1_aim stage2_vertical stage7a_comboschool stage7b_chain; do
  [ -f "trainer/configs/$c.yaml" ] || { echo "!! missing trainer/configs/$c.yaml" >&2; exit 1; }
done

best_or_final() { [ -f "runs/$1/best.pt" ] && echo "runs/$1/best.pt" || echo "runs/$1/final.pt"; }
ship() {  # ship <src> <models/dst.pt>  — backs up any existing model first
  [ -f "$1" ] || { echo "!! missing $1 — cannot ship $2" >&2; return 1; }
  [ -f "$2" ] && cp "$2" "${2%.pt}_prev.pt" && echo "   (backed up old $2 -> ${2%.pt}_prev.pt)"
  cp "$1" "$2" && echo ">> SHIPPED  $1  ->  $2"
}
stage() { echo; echo "############ $* ############"; }

echo "=================================================================="
echo "  TRAIN FROM ZERO   preset=$PRESET   (seg=$SEG steps/reboot)"
echo "  budgets: aim=$AIM_STEPS vert=$VERT_STEPS combo=$COMBO_STEPS fight=$FIGHT_STEPS"
echo "=================================================================="

stage "1/4  AIM — horizontal, from scratch"
INIT_CKPT=none \
  scripts/train_segmented.sh configs/stage1_aim.yaml fz_aim "$AIM_STEPS" "$SEG" || exit 1

stage "2/4  AIM — vertical, warm-started from stage 1"
INIT_CKPT="../$(best_or_final fz_aim)" \
  scripts/train_segmented.sh configs/stage2_vertical.yaml fz_aim_vert "$VERT_STEPS" "$SEG" || exit 1
ship "$(best_or_final fz_aim_vert)" models/aim.pt || exit 1

stage "3/4  COMBO SCHOOL — fighter learns to combo (fresh, uses new aim.pt)"
INIT_CKPT=none AIM_CKPT=../models/aim.pt \
  scripts/train_segmented.sh configs/stage7a_comboschool.yaml fz_combo "$COMBO_STEPS" "$SEG" || exit 1

stage "4/4  MORTAL FIGHT — vs a fighting opponent (warm from combo school)"
INIT_CKPT="../$(best_or_final fz_combo)" AIM_CKPT=../models/aim.pt \
  scripts/train_segmented.sh configs/stage7b_chain.yaml fz_fight "$FIGHT_STEPS" "$SEG" || exit 1
ship "$(best_or_final fz_fight)" models/fighter2.pt || exit 1

echo
echo "=================================================================="
echo "  DONE ✓   shipped  models/aim.pt  +  models/fighter2.pt"
echo "  The pilot auto-uses them — press G in game (SETUP.md)."
echo
echo "  Check your bot:"
echo "    cd trainer && ../.venv/bin/python eval.py ../runs/fz_fight/best.pt \\"
echo "        --episodes 30 --curriculum configs/stage7b_chain.yaml \\"
echo "        --aim ../models/aim.pt --stochastic"
echo "=================================================================="
