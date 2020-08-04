from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from descartes import PolygonPatch
from shapely.ops import cascaded_union, polygonize
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import numpy as np
import math
import pylab as pl

def plot_polygon(polygon, points):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    plt.plot(points[:,0], points[:,1], "o")
    plt.show()
    return fig

def alpha_shape(coords, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of coords.
    @param coords: numpy array of coords.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """

    if len(coords) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(points).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


from scipy.spatial import ConvexHull, convex_hull_plot_2d
from qcore.srf import *

def get_perimeter(srf_file, depth=True):
    """
    Like get_bounds but works with roughness where the edges aren't straight.
    srf_file: assumed to be finite fault
    """
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
            concave_hull, edge_points = alpha_shape(points, alpha=0.0001)
            lines = LineCollection(edge_points)
            plt.plot(delaunay_points[:,0], delaunay_points[:,1], 'o', hold=1, color='#f16824')
            plot_polygon(concave_hull, points)
            exit()
            hull = ConvexHull(points)
            import matplotlib.pyplot as plt
            plt.plot(points[:,0], points[:,1], 'o')
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
            plt.show()


    return perimeters

# test
perimeter = get_perimeter("Hossack_REL01.srf")
