
import alphashape
import numpy as np

from qcore.srf import *


def get_perimeter(srf_file, depth=True, plot=False):
    """
    Like get_bounds but works with roughness where the edges aren't straight.
    srf_file: assumed to be finite fault
    depth: work in progress, need to find associated points or do 3d concave hull
    plot: for testing only, plot points, perimeter and show
    """
    if plot:
        from matplotlib import pyplot as plt
        from descartes import PolygonPatch


    perimeters = []
    top_edges = []
    if depth:
        value = "depth"
    else:
        value = None

    with open(srf_file, "r") as sf:
        planes = read_header(sf, idx=True)
        points = int(sf.readline().split()[1])

        for i in range(len(planes)):
            a = []
            nstk = planes[i]["nstrike"]
            ndip = planes[i]["ndip"]
            points = np.array([get_lonlat(sf, value=None) for j in range(ndip * nstk)])

            # alpha=600 worked fine with SRF, roughness 0.1
            # 1000 was cutting into the plane and missing points entirely
            # 800 was zigzagging a bit too much along the edge
            ashape = alphashape.alphashape(points, 600.0)
            perimeters.append(np.dstack(ashape.exterior.coords.xy)[0])
            if plot:
                fig, ax = plt.subplots()
                ax.scatter(*zip(*points))
                ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
                plt.show()
                plt.close()

            # try to find edges, assume srf points are arranged "square" and corners are fixed
            c1 = np.argwhere(np.minimum.reduce(perimeters[-1] == points[0], axis=1))[0][0]
            c2 = np.argwhere(np.minimum.reduce(perimeters[-1] == points[nstk - 1], axis=1))[0][0]
            # assume shorter edge is top edge
            if abs(c2 - c1) < len(perimeters[-1]) / 2:
                # edge doesn't wrap array
                start = min(c1, c2)
                end = max(c1, c2)
                top_edges.append(perimeters[-1][start:end + 1])
            else:
                # edge wraps array ends
                start = max(c1, c2)
                end = min(c1, c2)
                top_edges.append(np.vstack((perimeters[-1][start:], perimeters[-1][:end + 1])))

    return perimeters, top_edges
