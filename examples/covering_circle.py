import math
import itertools

def _circle_from_two(p, q):
    """Circle with diameter pq."""
    cx = (p[0] + q[0]) / 2.0
    cy = (p[1] + q[1]) / 2.0
    r = math.hypot(p[0] - q[0], p[1] - q[1]) / 2.0
    return (cx, cy, r)

def _circle_from_three(a, b, c):
    """Circumcircle of triangle abc."""
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-12:
        return None  # colinear or nearly so
    a2 = ax*ax + ay*ay
    b2 = bx*bx + by*by
    c2 = cx*cx + cy*cy
    ux = (a2*(by-cy) + b2*(cy-ay) + c2*(ay-by)) / d
    uy = (a2*(cx-bx) + b2*(ax-cx) + c2*(bx-ax)) / d
    r = math.hypot(ux-ax, uy-ay)
    return (ux, uy, r)

def min_enclosing_circle(points):
    """
    points: list of (x,y), length up to 4
    returns (cx, cy, r)
    """
    # start with a huge circle
    best = (0, 0, float('inf'))
    pts = list(points)
    # check all pairs
    for p, q in itertools.combinations(pts, 2):
        cx, cy, r = _circle_from_two(p, q)
        if r < best[2] and all(math.hypot(x-cx, y-cy) <= r+1e-8 for x,y in pts):
            best = (cx, cy, r)
    # check all triples
    for a, b, c in itertools.combinations(pts, 3):
        circ = _circle_from_three(a, b, c)
        if circ is None:
            continue
        cx, cy, r = circ
        if r < best[2] and all(math.hypot(x-cx, y-cy) <= r+1e-8 for x,y in pts):
            best = (cx, cy, r)
    # if best is still infinite, all points are identical or single
    if best[2] == float('inf'):
        # pick first point, zero radius
        x0, y0 = pts[0]
        return (x0, y0, 0.0)
    return best

def min_enclosing_circle_4(traj1, traj2, traj3, traj4):
    
    r_min= 99999
    cx_min, cy_min=0,0
    for i in range(min(len(traj1), len(traj2), len(traj3), len(traj4))):
        pts = [(traj1[i][0], traj1[i][1]), (traj2[i][0], traj2[i][1]), (traj3[i][0], traj3[i][1]), (traj4[i][0], traj4[i][1])]
        cx, cy, r = min_enclosing_circle(pts)
        if r < r_min:
            r_min = r
            cx_min, cy_min = cx, cy
# Example:
if __name__ == '__main__':
    # pts = [(0,0), (1,0), (0,1), (1,1)]

    traj1 = [(0, 0), (1, 0), (0, 1), (1, 1)]
    traj2 = [(0, 0), (1, 0), (0, 1), (1, 1)]
    traj3 = [(0, 0), (1, 0), (0, 1), (1, 1)]
    traj4 = [(0, 0), (1, 0), (0, 1), (1, 1)]


    print(f"Center=({cx_min:.3f},{cy_min:.3f}), radius={r_min:.3f}")