---
phase: testing
title: Guided Gesture Collection UI — Testing
description: Test plan for session-based collection UI
---

# Testing — Guided Gesture Collection UI

## Unit Tests

- [ ] CollectionManager state transitions (idle→countdown→recording→done)
- [ ] Quality gate rejects low-confidence frames
- [ ] Auto-stop at exact target count
- [ ] Cancel mid-session preserves partial data
- [ ] CSV count initialization from existing file

## Integration Tests

- [ ] Full session: press key → countdown → capture N → auto-stop → CSV has exactly N new rows
- [ ] Camera MJPG 720p produces frames at ≥ 20fps
- [ ] Switching class mid-idle starts new session correctly

## Manual Tests

- [ ] Overlay: countdown numbers visible and centered
- [ ] Overlay: progress bar updates smoothly
- [ ] Overlay: done notification auto-clears
- [ ] Balance chart accurate vs actual CSV counts
- [ ] No sample recorded during countdown phase
