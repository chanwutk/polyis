# `execution/` Review — Agreed Remediation Plan

Outcome of a four-section review (Architecture → Code Quality → Tests →
Performance) of the uncommitted `execution/` refactor that replaced the
multiprocess + shared-memory pipeline with single-process thread pools.

This document is the handoff for implementing that work. All decisions below
are settled; the remaining work is execution, commit by commit.

**Workflow note:** implementation happens on `ace`, where the repository is the
source of truth. Nothing here requires the local-edit-then-`./sync` loop.

---

## 1. Decisions

| # | Area | Decision | Files |
|---|---|---|---|
| 1 | Error propagation | **Accept as-is.** Stage failures hang the runner; no monitor | — |
| 2 | Backpressure | Bound `compress_to_detect_q` and the compress pool's `pool_out_q` | `main.py:247-252`, `pool.py:108-109` |
| 3 | GIL / thread premise | Rebuild Cython, then sweep `--compress-workers` before any further change | `benchmark.py` |
| 4 | Pool duplication | Generalize `pool.py`; delete the `main.py` copies | `main.py:148-227`, `pool.py:26-137` |
| 5 | Decode truncation | Raise on short read; allocate with `np.zeros` | `decode_stage.py:137-151` |
| 6 | Config validation | Validate in `PipelineConfig.__post_init__` | `config.py`, `prune_stage.py:33` |
| 7 | Listing / probe DRY | Add `store.list_videos`; reuse cached `get_video_resolution` | 4 sites in `execution/` |
| 8 | Native leak | Free polyomino arrays after `pack()` | `compress_stage.py:54-64` |
| 9 | Missing tests | Add `test_detect_stage.py` + classify batch-arithmetic test | new files |
| 10 | Edge cases | Full edge-case battery | `test_decode_stage.py`, `test_compress_stage.py` |
| 11 | Assertion strength | Value-level assertions. **Hold** the validation-tolerance tightening | 4 test files |
| 12 | Silent skips | `requires_artifacts` markers, then the missing validation baseline | all test files |
| 13 | Prune rescan | Bucket the ILP solution by frame once | `prune_stage.py:96-101` |
| 14 | Classify copy | Preallocate and slice-assign | `classify_stage.py:109-111` |
| 15 | Compress threshold | Hoist the threshold out of the loop only | `compress_stage.py:57-62` |
| 16 | Timing | Report `pipeline_ms` alongside `total_ms` | `main.py:317-338`, `output.py` |

Explicitly **declined**: restoring `ErrorMonitor` (#1); bounding all six queues
(#2); Cythonizing `render_collage_cpu` (#3); eliminating the double threshold
via a conditional message contract (#15B/C); an RAII wrapper for polyomino
pointers (#8B); tightening `test_validation.py` tolerances before the
run-to-run variance is known (#11C).

---

## 2. Commit sequence

`4A` restructures `pool.py` and removes ~80 lines from `main.py`, so it lands
first; every later commit would otherwise be rebased onto moved code. Tests
ship with the fixes they pin rather than as a trailing pass.

### Commit 1 — `refactor(execution): unify pool spawners and relays`

- Add optional `on_send` / `on_receive` hooks to the relay functions in
  `pool.py`, so a caller can transform a message on the way in and on the way
  out without forking the relay.
- Collapse `spawn_pool` and `spawn_thread_pool` into one spawner parameterized
  by process-vs-thread; they currently differ only in `mp.Process`/`Thread` and
  `mp.Queue`/`queue.Queue`.
- Delete `_spawn_prune_pool`, `_prune_fan_out_relay`, `_prune_fan_in_relay`
  from `main.py` (`main.py:148-227`); prune passes `_to_prune_message` as
  `on_send` and a context-restoring closure as `on_receive`. The context dict
  and its lock move alongside.
- Update `pool.py`'s module docstring, which still refers to `spawn_pool`
  (`pool.py:14`) after that name changes.
- Extend `test_pool.py` to cover the hook path and both spawner modes.

### Commit 2 — `fix(execution): fail on truncated decode, validate config`

- `_decode_needed_with_opencv`: track filled buffer positions and raise
  `RuntimeError` naming the video and the shortfall when `cap.read()` stops
  early, mirroring the FFmpeg path's behavior at `decode_stage.py:189-195`.
  Allocate with `np.zeros` rather than `np.empty` as defense in depth.
  `cv2.CAP_PROP_FRAME_COUNT` is metadata and routinely overreports, so today
  the unfilled tail is uninitialized memory that flows silently downstream.
- `PipelineConfig.__post_init__`: assert `tracking_accuracy_threshold` is
  `None` or a member of `prune_stage._ALL_ACCURACY_THRESHOLDS`, and
  range-check `relevance_threshold` and `canvas_scale`. This fires before any
  thread or process spawns, and also guards `benchmark.py` and tests that
  construct a config directly. Do **not** put this in `parse_args` — project
  convention forbids comments there, and argparse would not cover direct
  construction.
- Tests pinning both: a truncated-video decode test, and an invalid-threshold
  rejection test.

### Commit 3 — `fix(pack): free polyomino arrays after packing`

- Loop `free_polyomino_array` over `polyominoes_stacks` immediately after
  `pack()` returns, in `try`/`finally` so an exception in `pack` still
  releases. Currently `free_polyomino_array` has zero callers repo-wide, so one
  `PolyominoArray` leaks per sampled frame per video.
- Ownership verified: `convert_collage_array_to_python` (`pack.pyx:177-191`)
  copies every coordinate into a fresh array, and `CollageArray_cleanup` runs
  before `pack()` returns, so nothing references the inputs afterward.
- This also gives a purpose to the `noexcept nogil` wrapper the refactor
  already added to `free_polyomino_array` (`group_tiles.pyx:105-111`).

### Commit 4 — `perf(execution): bound compress→detect queue, hot-path cleanups`

- Bound `compress_to_detect_q` and the compress pool's `pool_out_q` at
  `detect_batch_size * 4`. No deadlock risk: detect never puts upstream.
  Compress currently runs `cpu_count // 2` threads each emitting every collage
  of its video into an unbounded queue drained by a single GPU thread.
- `prune_stage.py:96-101`: bucket the ILP `selected` list into a dict keyed by
  frame in one pass instead of rescanning the whole solution per frame. Output
  is bit-identical regardless of how `group_tiles_all` scopes polyomino ids.
- `classify_stage.py:109-111`: preallocate the classifications array and
  slice-assign each batch, removing both the Python-level iteration over a
  numpy array and the second full copy in `np.stack`.
- `compress_stage.py:57-62`: hoist the threshold out of the per-frame loop.
  `group_tiles` still has to be called per frame.
- Add the prune value-level assertions from #11 to guard the prune change.

### Commit 5 — `refactor(execution): share video listing and resolution helpers`

- Add `list_videos(dataset, videoset)` to `polyis/io/store.py`; adopt at all
  four `execution/` sites (`main.py:238-241`, `classify_stage.py:136-140`,
  `benchmark.py:44-48`, `test_integration_smoke.py:102-104`).
- Replace the ad-hoc `cv2` probe in `_load_and_optimize_model`
  (`classify_stage.py:136-147`) with the already-memoized
  `get_video_resolution` (`polyis/utilities.py:191-203`).
- `decode_stage._probe_with_opencv` stays: it also needs `frame_count`.
- Scope held at `execution/`. The ~12 copies of the same predicate across
  `scripts/` are pre-existing and belong in a separate commit.

### Commit 6 — `test(execution): detect-stage coverage, stronger assertions`

- New `test_detect_stage.py` with `polyis.models.detector.detect_batch`
  monkeypatched, covering: multi-collage accumulation, cross-video
  interleaving, the zero-collage short-circuit (`detect_stage.py:61-69`), the
  `will_complete_video` early flush (`:75-80`), completion teardown
  (`:114-121`), the final drain on sentinel (`:44-50`), and **mixed canvas
  sizes** — `pending_msgs` can hold collages from videos of different
  resolutions, which nothing currently tests.
- Narrow `_classify_one_video` test with a stub model, asserting the batch and
  previous-frame index arithmetic at `classify_stage.py:89-94`.
- Strengthen existing assertions to value level:
  - `test_compress_stage.py` — assert canvas pixels match the distinct
    per-frame colors the fixture already sets up (`:43-49`), rather than only
    `canvas_rgb.shape`; assert `is_last` appears exactly once per video and
    `total_collages` matches the observed count.
  - `test_prune_stage.py` — assert the output relevant-tile set is a subset of
    the input and the count is non-increasing. The current test would pass if
    pruning were a no-op.
  - `test_track_stage.py` — assert a trajectory invariant, **not** an exact
    track count, which would be brittle across tracker versions.
- Remaining edge cases from #10: parametrize `_compute_frame_plan` over
  `frame_count ∈ {1,2,3,10,100} × sample_rate ∈ {1,2,4,16}` with invariants
  (every sampled index present, last frame present, all positions resolve) —
  this exercises the single-frame guard at `decode_stage.py:115`, currently
  untested; plus a zero-collage compress test for `compress_stage.py:78-90`.

### Commit 7 — `test(execution): mark artifact-dependent tests, add validation baseline`

- Mark the three artifact-dependent files with
  `@pytest.mark.requires_artifacts` so `pytest -m "not requires_artifacts"` is
  the quick command and `-m requires_artifacts` is the full one:
  `test_integration_smoke.py:51-54`, `test_prune_stage.py:67-68`,
  `test_validation.py:83-86`.
- Script the `tracking.scripts.jsonl` baseline generation. `test_validation.py:8`
  claims it is "produced by the test fixture", but no such fixture exists: the
  baseline on `ace` was created by hand and cannot be reproduced elsewhere.
  The test itself is **not** dead — verified 2026-09-12 on `ace`, all three
  videos pass within tolerance, so `execution/` does currently agree with
  `scripts/p060`. The gap is reproducibility, not coverage.
- Update `DESIGN.md` §7, which lists all eight test files as "Important tests"
  without noting that three do not run by default.

### Commit 8 — `perf(execution): report pipeline_ms alongside total_ms`

- Record `pipeline_ms` (measured at last result received) next to `total_ms`
  (including output I/O) in `runtime.jsonl`.
- `save_tracking_result` currently runs inline on the collector thread
  (`main.py:334`), inside the window measured from `main.py:317` to `:338`. It
  serializes one JSON line per frame, so the reported throughput number cannot
  be decomposed into pipeline versus disk.
- If `total_ms - pipeline_ms` proves material, escalate to a dedicated writer
  thread. If it is noise, that complexity is never needed.

---

## 3. Deferred — requires `ace`

Both items need the container and the dataset artifacts.

1. **Rebuild Cython.** `polyis/pack/group_tiles.pyx` and `pack.pyx` gained
   `noexcept nogil` declarations and `with nogil:` blocks. Until
   `cd lib && ./build.sh` runs, the compress thread pool still serializes on
   the GIL through those C calls and no measurement means anything.
2. **Sweep `--compress-workers`** at 1 / 2 / 4 / 8 via `execution/benchmark.py`.
   The `nogil` work covers `group_tiles` and `pack`, but `render_collage_cpu`
   (`polyis/pack/render.py:205-266`) is pure Python/numpy — a per-polyomino
   loop doing per-tile slice assignments on ~10 KB buffers, too small for
   numpy to release the GIL usefully. So compress threads still serialize
   through the memcpy-heavy phase, and `cpu_count // 2` may be buying little
   or actively costing classify/detect latency. Decide on Cythonizing render
   only after seeing the curve.

---

## 4. Known-good — checked, no action

- `track_stage.py:60-69` calls `tracker.update()` on every frame including
  non-sampled ones. This matches `scripts/p060_exec_track.py:159-161`, which
  documents it as deliberate: SORT's age and velocity model must advance per
  frame. Changing it would change results.
- `get_video_resolution` is already memoized, so the per-video call in
  `track_stage.py:53` costs nothing.
- `pack.c` frees its own internal `candidates` buffer; the leak in #8 is
  strictly the caller-owned input arrays.


---

## 5. Environment state on `ace` (verified 2026-09-12)

- Container `polyis` is up; Python 3.13.12, `/polyis`, `/polyis-data`,
  `/polyis-cache` all mounted; 8 GPUs visible.
- **Cython is rebuilt.** `python setup.py build_ext --inplace` (there is no
  `lib/` directory and no `build.sh`; `AGENTS.md` is stale on this point).
  Verified in the generated C that `pack()` is wrapped in
  `PyEval_SaveThread()` / `PyEval_RestoreThread()`, so the `nogil` work is
  genuinely in effect and item #3 can now be measured.
- **`loginctl enable-linger` is now on.** Without it the rootless Docker
  daemon was torn down whenever the last SSH session ended, killing the
  container with exit 255. This was the cause of the intermittent
  "Cannot connect to the Docker daemon" errors.
- **Test baseline: 1 failed, 19 passed** (`pytest execution/tests`, 86s).
  The single failure is `test_prune_stage.py::test_prune_worker_round_trip`
  raising `GRBstartenv failed — check Gurobi license`. The mounted license is
  a WLS (server-validated) file dated March 2026; token checkout is failing.
  **Prune cannot run until that licence is renewed**, which also blocks
  item #13 and any end-to-end run with pruning enabled.
- `test_validation.py` and `test_integration_smoke.py` both pass here; they
  skip on a machine without the dataset artifacts.
