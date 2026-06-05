# Limitations

Stated honestly so results are not over-read (CLAUDE.md hard rule: never present
proxies as real measurements).

## No real depth
There is **no depth sensor and no stereo** on this Windows workbench. "Distance"
is a **proxy** from 2-D bounding-box geometry (a larger / faster-growing box is
treated as closer / approaching). It is **not** metres.

## TTC is a proxy
"Time-to-collision" is computed as `bbox_height / bbox_height_change_rate`. This
is a **pseudo-TTC**, not physical seconds. It is unstable when the box barely
changes and it ignores real ego speed and 3-D motion. Real depth-based TTC is
deferred to Project B (CARLA).

## Lane detection is classical CV
The ego-lane estimate uses ROI cropping + Canny + Hough lines, not learned
segmentation. Expect failure with faded or worn markings, glare or low light,
sharp curves, crests, and rain. A hook is left for a future lightweight
segmentation model.

## Detection / tracking bounds
A nano YOLO model is chosen for speed; small, distant, or occluded objects may be
missed. The MVP tracker is greedy IoU matching, so ids can switch under heavy
occlusion or fast motion (a SORT/Kalman upgrade is planned).

## Known failure cases (to document with examples as the MVP runs)
- Night / low-light driving.
- Heavy rain or windscreen glare.
- Crowded urban scenes (id switches).
- Sharp curves where the straight-line ego-lane assumption breaks.
