
import numpy as np
import alphashape

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

    return perimeters

# test
perimeter = get_perimeter("Hossack_REL01.srf")
