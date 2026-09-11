# Plan: gated telephoto activation + holistic track identity

## Problem

Today `run_tracker.py` runs two independent pipelines every frame:

- `tele_loop.step()` (`control/loop.py:77`) unconditionally calls
  `telephoto_detector.track(frame)` every iteration, whether or not the
  target is anywhere near the telephoto camera's field of view.
- `run_acquisition_step()` (`scripts/run_tracker.py:155`) runs the wide
  detector only when `tele_loop.track_manager.locked_track_id is None`.

Each pipeline owns its own `TrackManager` (`tracking/track_manager.py`),
each with a sticky lock over its own detector's ByteTrack ID space
(`tracking/detector.py:97-101`, `persist=True`). The two ID spaces are
unrelated — there is no representation of "the wide camera and the
telephoto camera are both looking at the same physical person."

This causes two concrete problems:

1. **Wasted GPU work**: telephoto inference runs every frame even when
   the wide camera's evidence says the target is nowhere near telephoto's
   current view, so most telephoto inference calls are certain to find
   nothing.
2. **No cross-camera continuity**: when telephoto loses its lock, the
   only way it re-acquires is `select_target`'s frame-center-distance
   fallback (`tracking/target.py:29-35`) applied to whatever telephoto
   sees right now — it has no notion of *which* physical target the wide
   camera has continued to track in the meantime, so re-acquisition can
   silently lock onto the wrong object if more than one is present.

`control/loop.py:50-53` already flags this as a known seam
(`track_manager` is injectable "for a future ReID-capable or
multi-camera-aware manager"). This plan fills that seam.

## Design

### 1. Evolve `TrackManager` in place — don't add a parallel class

`control/loop.py:50-53` already documents `TrackManager` as the intended
seam for this: "Injectable so a future ReID-capable or multi-camera-aware
manager can be swapped in without touching TrackingLoop again." A new
`TargetCoordinator` class living alongside it would just be two managers
stacked for no reason — the multi-camera responsibility described below
*is* what `TrackManager` was already slotted to grow into. So this plan
extends `tracking/track_manager.py`'s `TrackManager` directly rather than
introducing a new name/module.

`TrackManager` gains:

- Shared state: the canonical current target as a **position estimate**,
  not a track ID — e.g. last-known wide-frame pixel (or a shared
  world/angle coordinate; see Open Question A) plus a timestamp and
  which camera last confirmed it. Track IDs are never compared across
  cameras since they come from independent ByteTrack instances and mean
  nothing to each other.
- New methods `update_wide(detections, frame_center) -> Detection | None`
  and `update_telephoto(detections, frame_center) -> Detection | None`,
  each internally doing the same nearest-to-last-known-position
  selection `select_target` does today, but seeded from the shared
  estimate instead of a same-camera track ID.
- The existing `update()` method (and `locked_track_id` property) stay
  exactly as they are, unchanged — that's the single-camera path used by
  `sign_check.py` and `calibrate_wide_handoff.py`'s telephoto-only loop,
  which don't need cross-camera awareness. `update_wide`/`update_telephoto`
  are additive, not a breaking change to the existing interface or its
  tests (`tests/test_track_manager.py`).
- On every successful selection from *either* camera, updates the shared
  estimate (converting the telephoto's local pixel to the shared
  coordinate via the existing `WideHandoffMapper`/servo-position
  relationship, since telephoto has no independent pixel-to-world
  calibration today).
- Keeps a short staleness window (e.g. drop the estimate after N frames
  with no confirmation from either camera) so a target that's genuinely
  gone doesn't stick forever.

`run_tracker.py:138` currently creates a separate `wide_track_manager`
and lets `TrackingLoop` default-construct its own `TrackManager`
(`control/loop.py:54`). Both are replaced by one shared `TrackManager`
instance, constructed once and passed into `TrackingLoop(..., track_manager=...)`
and used directly in `run_acquisition_step()`.

### 2. Gate telephoto activation on FOV overlap

Add a method to `TrackManager` (or a small standalone helper next to
`WideHandoffMapper`): `telephoto_likely_has_target(current_pan, current_tilt) -> bool`,
using the shared target estimate and the telephoto's known angular FOV
(new config, see below) to decide whether the target is plausibly inside
telephoto's current view before paying for a telephoto inference call.

`control/loop.py`'s `step()` currently always calls `self.detector.track(frame)`.
Change `TrackingLoop` to accept an optional `should_track: Callable[[], bool]`
gate (default: always True, preserving current behavior for
`sign_check.py`/`calibrate_wide_handoff.py`, which are single-camera).
When the gate returns False, `step()` skips detection entirely for that
frame (still reads/serves the frame for RTSP if needed) and reports no
target — cheap enough to check every iteration.

`run_tracker.py` wires the gate as
`lambda: tele_loop.track_manager_has_lock() or track_manager.telephoto_likely_has_target(...)`
— i.e. once telephoto has its own lock, keep running it regardless (so
it doesn't drop mid-track); before it has a lock, only spin it up once
the wide camera's estimate says the target is within/near telephoto's
projected FOV.

### 3. Wiring changes

- `run_tracker.py:136-141`: construct one shared `TrackManager` instead
  of `wide_track_manager` + `tele_loop`'s default `TrackManager`; pass it
  into `TrackingLoop(..., track_manager=track_manager)` and use it
  directly in `run_acquisition_step()`.
- `control/loop.py`: `TrackingLoop.step()` gains the `should_track` gate
  described above; `self.track_manager.update(...)` call sites become
  `self.track_manager.update_telephoto(...)`.
- `run_acquisition_step()` (`run_tracker.py:161-166`) calls
  `track_manager.update_wide(...)` instead of `wide_track_manager.update(...)`.
- `sign_check.py` and `calibrate_wide_handoff.py`'s telephoto-only loop
  keep using plain `update()` — no changes needed there since
  `update_wide`/`update_telephoto` are additive.

### 4. New calibration data needed

`telephoto_likely_has_target` needs telephoto's angular (or wide-pixel-
equivalent) half-FOV. Add `telephoto_fov_half_width_px` /
`telephoto_fov_half_height_px` (in wide-pixel-equivalent units, using the
existing `pan_ticks_per_px` / `tilt_ticks_per_px` slopes to convert
telephoto's known tick-space FOV into wide-pixel space) to
`WideHandoffCalibration` (`control/wide_handoff.py:14-19`), populated by
`scripts/calibrate_wide_handoff.py` (it already samples both cameras
together, so it's the natural place to measure this).

## Open questions to settle before implementing

- **A: shared coordinate space.** Simplest option is to keep the shared
  estimate in *wide-pixel* coordinates and always convert telephoto
  detections back into wide-pixel space via current servo position +
  `WideHandoffMapper`'s inverse. This avoids inventing a new coordinate
  system but makes `TrackManager` depend on reading servo position.
  Alternative: store the estimate directly in pan/tilt absolute ticks
  (what the servos already use), skipping pixel space entirely for the
  shared state. Leaning toward ticks — it's what both cameras' outputs
  ultimately get converted to anyway, and it's what `should_track`'s FOV
  check needs regardless.
- **B: hysteresis at the FOV boundary.** Gating purely on "inside FOV
  vs. not" will flap telephoto on/off for a target sitting right at the
  edge. Needs a margin (activate a bit before the boundary, deactivate a
  bit after) — exact numbers to be tuned empirically, not guessed up
  front.
- **C: staleness window.** How many frames without confirmation before
  `TrackManager` drops its shared estimate and falls back to today's
  "nearest to frame center" behavior. Needs to be tuned against real
  footage.
- **D: multiple simultaneous targets.** This plan keeps the existing
  single-target sticky-lock policy (`select_target`), just makes the
  lock cross-camera. Handling multiple candidate targets holistically
  (e.g. picking the "best" one across both cameras) is out of scope here
  and would be a follow-up on top of this seam.

## Testing

- Unit tests for the new `update_wide`/`update_telephoto` behavior,
  added to the existing `tests/test_track_manager.py`: confirm wide-only
  updates, telephoto-only updates, handoff continuity (wide locks, then
  telephoto detections near the estimate get preferred over other
  detections), and staleness expiry. Existing tests for `update()` stay
  unchanged and must keep passing.
- Unit tests for `telephoto_likely_has_target` against the new
  calibration fields: inside FOV, outside FOV, boundary/hysteresis
  cases.
- `TrackingLoop`'s `should_track` gate: extend `tests/test_target.py`-
  style unit tests (mock detector) to confirm `detector.track()` is not
  called when the gate returns False, and that behavior is unchanged
  when no gate is passed (default `True`).
- Manual validation: run `scripts/run_tracker.py --verbose` (no RTSP
  needed) and confirm telephoto detector logging/timing
  (`detector.last_timing`) shows skipped frames while a target is far
  from telephoto's FOV, and confirm handoff still locks smoothly when it
  arrives.

## Rollout

Land in this order, each independently mergeable:

1. `TrackManager`'s new shared-estimate state + `update_wide`/
   `update_telephoto` + tests, not yet wired into `run_tracker.py`.
2. `TrackingLoop.should_track` gate + tests, defaulting to always-on so
   existing scripts are unaffected.
3. New calibration fields + `calibrate_wide_handoff.py` changes to
   measure them (existing calibration files will need re-running once;
   `load_wide_handoff_config` should error clearly on missing fields, as
   it already does via the `KeyError` handling at `wide_handoff.py:38-39`).
4. Wire `run_tracker.py` to use the shared `TrackManager` and the gate
   together.
