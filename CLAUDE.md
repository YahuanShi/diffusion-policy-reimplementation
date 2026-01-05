# Diffusion Policy Reimplementation

## Task
Reimplement diffusion policy for **Pick and Place** (not PushT).

## Reference
Original implementation: `/home/yahuanshi/Projects/il/diffusion_policy-main/`

## Rules

### Git Commits
- After completing each step in PROGRESS.md, commit and push immediately (fully automatic)
- **Backdate** all commits to January 2026 (Jan 4 ~ Jan 28), leave first/last 3 days blank
- Use `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` with the date assigned in PROGRESS.md
- No `Co-Authored-By` trailer
- Run any linters before committing

### Implementation
- Follow the staged order in PROGRESS.md
- After each stage, run the corresponding validation script before moving on
- Read the reference implementation for understanding, then implement from scratch
- Use Chinese for all communication with the user

### Auto-Continue
- On session start, read PROGRESS.md to find the next incomplete step
- Continue from where the last session left off
