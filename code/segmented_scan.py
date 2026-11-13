"""Sequential multi-break scan (advisor's rule, 2026-09-24, revised 2026-09-29).

Grow a window from the segment start; the first month whose inclusion makes the test reject is the break, and the
next segment starts at that month. If the first testable window of a segment already rejects ("edge": no credible
date), retreat: shrink the window one month at a time until a length does not reject. If [s, e) does not reject while
every longer window up to the first test does, month e is the break: the stretch [s, e) is recorded as a segment
ending in a dated break, and the scan restarts at e. Segments after the first may start testing with a shorter
window (`restart_min_window`); they never use data from before their start.

A window too short for local PCA at the given rank returns stage "degenerate": it counts as no evidence in the
forward search (not a rejection), the edge check uses the first testable window, and retreat stops there (an
edge that cannot be dated by retreating stays an unresolved edge).
"""


def _degenerate(result):
    return result.get("stage") == "degenerate"


def _retreat(seg_start, window, test_window, batch=1, mapper=map):
    """Shrink below `window` until a length does not reject. Returns (break_month or None, trajectory).
    Lengths are tested in batches (in parallel when `mapper` is a pool map); results are used in order."""
    trajectory = []
    ends = [seg_start + length for length in range(window - 1, 0, -1)]
    for i in range(0, len(ends), batch):
        chunk = ends[i:i + batch]
        for e, result in zip(chunk, mapper(test_window, [seg_start] * len(chunk), chunk)):
            trajectory.append({"end": e, "retreat": True, **result})
            if _degenerate(result):
                return None, trajectory
            if not result["reject"]:
                return e, trajectory
    return None, trajectory


def segmented_scan(n_obs, test_window, min_window=100, restart_min_window=None, retreat=True,
                   batch=1, mapper=map, run_k=1):
    """Returns a list of segments: start, break_month (None = stable to the end), rejected_at_min_window (an
    unresolved edge), retreated (break dated by retreat), trajectory (all tested windows, in order).

    run_k (week 6): a break needs run_k consecutive rejecting windows; it is dated at the first of them, exactly as
    with a single rejection (run_k = 1, the original rule). A shorter run of rejections followed by a non-rejection is
    ignored and the search goes on; a run still shorter than run_k when the data end counts as no break."""
    restart_min_window = min_window if restart_min_window is None else restart_min_window
    segments, seg_start, is_first = [], 0, True
    while True:
        window = min_window if is_first else restart_min_window
        if n_obs - seg_start < window:
            break
        trajectory, break_end, first_test, run_start = [], None, None, None
        ends = list(range(seg_start + window, n_obs + 1))
        for i in range(0, len(ends), batch):
            chunk = ends[i:i + batch]
            for e, result in zip(chunk, mapper(test_window, [seg_start] * len(chunk), chunk)):
                trajectory.append({"end": e, **result})
                if first_test is None and not _degenerate(result):
                    first_test = e
                if result["reject"]:
                    run_start = e if run_start is None else run_start
                    if e - run_start + 1 >= run_k:
                        break_end = run_start
                        break
                else:
                    run_start = None
            if break_end is not None:
                break
        edge = break_end is not None and break_end == first_test
        if edge and retreat:
            brk, back = _retreat(seg_start, break_end - seg_start, test_window, batch, mapper)
            if brk is not None:
                # the break test is the window [s, brk + 1): the shortest one that still rejects
                brk_test = back[-2] if len(back) > 1 else next(t for t in trajectory if t["end"] == break_end)
                segments.append({"start": seg_start, "break_month": brk, "rejected_at_min_window": False,
                                 "retreated": True, "break_p": brk_test.get("p"), "trajectory": back[::-1] + trajectory})
                seg_start, is_first = brk, False
                continue
            trajectory = back[::-1] + trajectory
        brk_p = None if break_end is None else next(t for t in trajectory if t["end"] == break_end and not t.get("retreat")).get("p")
        segments.append({"start": seg_start, "break_month": None if break_end is None else break_end - 1,
                         "rejected_at_min_window": edge, "retreated": False,
                         "break_p": brk_p, "trajectory": trajectory})
        if break_end is None:
            break
        seg_start, is_first = break_end - 1, False
    return segments
