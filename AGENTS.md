# TouchlessControl — AI Agent Rules

## Project Context

Hand gesture recognition system for touchless laptop control. Uses MediaPipe for hand detection, TFLite for gesture classification, pynput for system control (mouse/keyboard).

**Tech stack:** Python 3.11 · conda env `sign` · MediaPipe 0.10.5 · TFLite 2.14.0 · OpenCV 4.10.0 · pynput · pyyaml

Phase documentation is located in `docs/ai/`.

## Documentation Structure

- `docs/ai/requirements/` — Problem understanding and requirements
- `docs/ai/design/` — System architecture and design decisions (include mermaid diagrams)
- `docs/ai/planning/` — Task breakdown and project planning
- `docs/ai/implementation/` — Implementation guides and notes
- `docs/ai/testing/` — Testing strategy and test cases
- `docs/ai/deployment/` — Deployment and infrastructure docs

## Code Style & Standards

- Follow the project's established code style and conventions
- Write clear, self-documenting code with meaningful variable names
- Add comments for complex logic or non-obvious decisions
- All terminal commands must activate conda env first: `conda activate sign`

## Development Workflow

- Review phase documentation in `docs/ai/` before implementing features
- Keep requirements, design, and implementation docs updated as the project evolves
- Reference the planning doc for task breakdown and priorities

### Timeline Tracking (REQUIRED)

After completing any significant action (code changes, bug fixes, new features, config changes), the agent **MUST** follow the **`track-progress` skill** (`.github/skills/track-progress/SKILL.md`) to update project timeline.

**Quick summary:**

1. Always append a row to `docs/ai/timeline.md` (date, time HH:MM, title, result, detail link)
2. Create/append `docs/timeline/DD-MM-YYYY.md` only when the action is complex (≥3 files, debugging, design decisions)
3. At session start, read `docs/ai/timeline.md` to understand recent project context

### Checklist Verification & Documentation Updates (REQUIRED)

When verifying or completing checklist items from planning/implementation docs:

1. **Verify** the task/resource by running actual commands or inspecting the system
2. **Update** the corresponding documentation immediately after verification:
   - Mark checklist items as `[x]` with ✅ and verification date
   - Update configuration values to reflect actual verified values

## AI Interaction Guidelines

- When implementing features, first check relevant phase documentation
- For new features, start with requirements clarification
- Update phase docs when significant changes or decisions are made

## Testing & Quality

- Write tests alongside implementation
- Follow the testing strategy defined in `docs/ai/testing/`
- Ensure code passes all tests before considering it complete

### Mandatory Testing After Implementation (REQUIRED)

After completing any implementation task, the agent **MUST** create and run tests:

1. **Unit Tests** — Test each changed module in isolation with mocks. Cover: normal flow, edge cases, error handling.
2. **Integration Tests** — Test interactions between changed modules.

**Test file naming convention:**

- Unit: `tests/unit/test_{module_name}.py`
- Integration: `tests/integration/test_{feature_name}.py`

**All tests must pass (`pytest`) before the task is marked as completed.**

### Known Technical Constraints

- **Webcam**: V4L2, MJPG 1280×720@30fps supported. YUYV 720p = only 10fps. Default: MJPG 720p. Must disable `exposure_dynamic_framerate` (v4l2-ctl) or camera drops to 10fps in low light.
- **OpenCV Qt backend**: Must set `QT_QPA_PLATFORM=xcb` before importing cv2 (Wayland fix).
- **MediaPipe Hands**: `calc_landmark_list()` returns `[x, y, z]` per point. Drawing functions need `[p[:2] for p in landmark_list]`.
- **Feature vectors**: 93-dim (21kp×xyz + 15 angles + 5 tip-wrist + 5 tip-palm + 5 states) + 42-dim legacy (XY only).
- **pynput**: Works via XWayland (DISPLAY=:0). Native Wayland input not supported.
- **CSV data format**: 43 columns (1 label + 42 features = legacy XY format). Must maintain backward compatibility.

## Documentation

- Update phase documentation when requirements or design changes
- Keep inline code comments focused and relevant
- Document architectural decisions and their rationale
- Use mermaid diagrams for any architectural or data-flow visuals

## Key Commands

- Understand project requirements and goals (`review-requirements`)
- Review architectural decisions (`review-design`)
- Plan and execute tasks (`execute-plan`)
- Verify implementation against design (`check-implementation`)
- Writing tests (`writing-test`)
- Perform structured code reviews (`code-review`)

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **TouchlessControl** (1484 symbols, 2209 relationships, 1 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/TouchlessControl/context` | Codebase overview, check index freshness |
| `gitnexus://repo/TouchlessControl/clusters` | All functional areas |
| `gitnexus://repo/TouchlessControl/processes` | All execution flows |
| `gitnexus://repo/TouchlessControl/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
