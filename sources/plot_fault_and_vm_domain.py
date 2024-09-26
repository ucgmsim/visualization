import argparse
import re
from pathlib import Path
import sys
import webbrowser

import folium
import h5py
from shapely.geometry import Point, Polygon


def parse_model_params(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    corners = re.findall(r"c\d= *([\d\.\-]+) *([\d\.\-]+)", content)
    return [(float(lat), float(lon)) for lon, lat in corners]


def is_point_in_polygon(point, polygon):
    return polygon.contains(Point(point))


def calculate_polygon_center(polygon):
    centroid = polygon.centroid
    return [centroid.x, centroid.y]


def extract_fault_traces_from_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        corners = f["/"].attrs["corners"]
        fault_traces = []
        for plane in corners:
            plane_traces = [(point[1], point[0]) for point in plane]
            fault_traces.append(plane_traces)
    return fault_traces


# Step 1: Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Plot fault traces and model corners on a map."
)
parser.add_argument("srfinfo", type=Path, help="Path to the srfinfo file")
parser.add_argument(
    "model_params_file", type=Path, help="Path to the model_params file"
)
parser.add_argument(
    "--show", action="store_true", help="Show the map in the web browser"
)
parser.add_argument(
    "--outdir", type=Path, help="Output directory for the HTML file"
)
args = parser.parse_args()

assert args.srfinfo.exists(), f"File {args.srfinfo} does not exist."
assert args.model_params_file.exists(), f"File {args.model_params_file} does not exist."

if args.outdir is None:
    args.outdir = args.srfinfo.parent

# Step 2: Parse the model_params file to extract corners
model_corners = parse_model_params(args.model_params_file)
model_polygon = Polygon(model_corners)

# Step 3: Calculate the center of the model polygon
map_center = calculate_polygon_center(model_polygon)

# Step 4: Connect to the SQLite database and extract fault traces
fault_traces = extract_fault_traces_from_hdf5(args.srfinfo)

# Step 5: Check if all fault traces are inside the model corners
all_inside = all(
    is_point_in_polygon(point, model_polygon)
    for plane in fault_traces
    for point in plane
)


if not all_inside:
    print("ERROR: Some fault traces are outside the model corners.", file=sys.stderr)

# Step 6: Create a folium map centered around the center of the model polygon
fault_map = folium.Map(location=map_center, zoom_start=7)

# Step 7: Add fault trace points to the map and connect them with a PolyLine
for plane in fault_traces:
    folium.Polygon(plane, color="blue", weight=2.5, opacity=1).add_to(fault_map)


# Step 8: Add the rectangular plane defined by the model corners
folium.Polygon(model_corners, color="red", weight=2.5, opacity=1).add_to(fault_map)

# Step 9: Save the map to an HTML file and open it in the web browser
map_path = args.outdir / f"{args.srfinfo.stem}.html"
fault_map.save(map_path)
print(f"Map saved to {map_path}")

if args.show:
    webbrowser.open(str(map_path))
