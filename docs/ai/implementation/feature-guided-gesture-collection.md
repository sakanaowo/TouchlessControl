---
phase: implementation
title: Guided Gesture Collection UI — Implementation
description: Implementation tracking for session-based collection UI
---

# Implementation — Guided Gesture Collection UI

## Status: Not Started

## Completed Tasks

_(none yet)_

## Implementation Notes

- Camera MJPG confirmed via v4l2-ctl: 1280×720@30fps supported
- Existing CSV has 6797 samples (42-dim legacy format, classes 0-6 only)
- Current `logging_csv()` function will be called by CollectionManager instead of direct main loop
