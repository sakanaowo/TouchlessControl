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

After completing any significant action, the agent **MUST**:

1. Add a **summary row** to the table in `docs/ai/timeline.md`
2. Add a **detailed entry** in the daily file `docs/timeline/DD-MM-YYYY.md`

Timeline entries **MUST include hour precision** (`HH:MM`). Daily detail files use **hour only** since the date is in the filename.

**What counts as "significant":**

- Code changes affecting core logic or architecture
- Bug fixes that required investigation
- New features or modules created
- Configuration or infrastructure changes

**Summary row format** (`docs/ai/timeline.md`):

```
| YYYY-MM-DD | HH:MM | Tiêu đề | Kết quả ngắn | [chi tiết](../../docs/timeline/DD-MM-YYYY.md) |
```

**Daily detail format** (`docs/timeline/DD-MM-YYYY.md`):

```
### HH:MM — [Tiêu đề]
**Vấn đề:** ... **Nguyên nhân:** ... **Hành động:** ... **Kết quả:** ... **References:** ...
```

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

- **Webcam**: V4L2, MJPG 1280×720@30fps supported. YUYV 720p = only 10fps. Default: MJPG 720p.
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
