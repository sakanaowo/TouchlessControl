---
name: track-progress
description: "Track project progress timeline after significant actions. Use when: completing code changes, bug fixes, new features, config changes, architecture decisions. Updates docs/ai/timeline.md summary table and optionally docs/timeline/DD-MM-YYYY.md daily detail files. Agent MUST invoke this after every significant action."
---

# Track Progress

## Purpose

Maintain a living project timeline that the agent can read at the start of any session to understand the full history of changes, decisions, and their context. GitHub manages source code; this skill tracks progress, rationale, and impact.

## When to Trigger

**MUST track** after:

- Code changes affecting core logic or architecture
- Bug fixes that required investigation
- New features or modules created
- Configuration or infrastructure changes
- Architecture or design decisions

**Do NOT track:**

- File reads, searches, simple Q&A
- Trivial formatting or typo fixes
- Pure research without code changes

## Procedure

### Step 1 — Read current timeline

Read `docs/ai/timeline.md` to understand the latest entries and avoid duplicates.

### Step 2 — Update summary table

Append a row to the table in `docs/ai/timeline.md`:

```
| YYYY-MM-DD | HH:MM | Tiêu đề ngắn | Kết quả ngắn | Chi tiết |
```

| Field    | Description                                                                  |
| -------- | ---------------------------------------------------------------------------- |
| Date     | `YYYY-MM-DD` format                                                          |
| Time     | `HH:MM` 24h format — hour precision required                                 |
| Tiêu đề  | Concise action title in Vietnamese (3-10 words)                              |
| Kết quả  | One-line result: what changed, metrics (e.g., "111 tests pass")              |
| Chi tiết | `[chi tiết](../timeline/DD-MM-YYYY.md)` if detail file exists, otherwise `—` |

### Step 3 — Decide if daily detail file is needed

**Create** `docs/timeline/DD-MM-YYYY.md` when:

- Action involves multiple steps or files (≥3 files changed)
- Debugging or investigation was required
- Architecture or design decisions were made
- Impact spans multiple modules
- There are important references to other files or timeline entries

**Skip** detail file when:

- Single-file fix with obvious purpose
- Simple config value change
- Routine dependency update
- The summary row already captures everything meaningful

### Step 4 — Write daily detail entry (if needed)

Append to `docs/timeline/DD-MM-YYYY.md` (create file if it doesn't exist for that day):

```markdown
### HH:MM — [Tiêu đề]

**Vấn đề:** Mô tả ngắn vấn đề hoặc yêu cầu
**Nguyên nhân:** Tại sao cần thực hiện (context, trigger, user request)
**Hành động:**

- Bước 1: ...
- Bước 2: ...

**Kết quả:** Kết quả cụ thể (tests pass count, metrics, behavioral change)
**Ảnh hưởng:** Module/file nào bị ảnh hưởng, breaking changes nếu có
**References:**

- `path/to/modified/file.py` (modified/new: mô tả thay đổi)
- `docs/ai/timeline.md` entry YYYY-MM-DD HH:MM (nếu liên quan)
```

**Rules for detail entries:**

- Use Vietnamese for natural descriptions
- List ALL files changed with short description of each change
- Reference previous timeline entries if this is a continuation
- Include test results (pass count, coverage) when applicable

### Step 5 — Verify

- [ ] Summary table row has correct `|` separators and all 5 columns
- [ ] Date format is `YYYY-MM-DD`, time is `HH:MM`
- [ ] Detail file link is correct (relative path `../timeline/DD-MM-YYYY.md`)
- [ ] Detail entry follows the template (if created)
- [ ] No duplicate entries for the same action

## Reading Timeline (Start of Session)

When starting a new task, the agent SHOULD:

1. Read `docs/ai/timeline.md` to understand recent project history
2. If the current task relates to a previous entry, note the connection
3. Reference previous entries in the current detail entry's References section

## File Structure

```
docs/
├── ai/
│   └── timeline.md          ← Summary table (always updated)
└── timeline/
    ├── 10-04-2026.md         ← Daily detail (only when needed)
    ├── 12-04-2026.md
    └── 13-04-2026.md
```
