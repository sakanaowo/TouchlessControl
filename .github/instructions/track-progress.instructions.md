---
applyTo: "**"
description: "Auto-trigger progress tracking after significant agent actions. Ensures docs/ai/timeline.md is always up-to-date."
---

## Progress Tracking (Auto-trigger)

After completing any **significant action** (code changes, bug fixes, new features, config changes, architecture decisions), you MUST follow the `track-progress` skill to update the project timeline.

**Workflow:**

1. Complete the implementation work
2. Run tests and verify results
3. **Before calling `task_complete`**, update progress tracking:
   - Always: append row to `docs/ai/timeline.md`
   - If complex (≥3 files, debugging, design decisions): also update `docs/timeline/DD-MM-YYYY.md`
4. Then call `task_complete`

**Quick reference — summary row format:**

```
| YYYY-MM-DD | HH:MM | Tiêu đề | Kết quả ngắn | [chi tiết](../timeline/DD-MM-YYYY.md) hoặc — |
```

**At session start:** Read `docs/ai/timeline.md` to understand recent project context before working on substantial tasks.
