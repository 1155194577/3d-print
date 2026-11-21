"""Fit Hong Kong wind samples and export a two-piece prototype blade in mm."""

import argparse
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import trimesh
from scipy.stats import weibull_min


HKO_WIND = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind.csv"
ARCHIVE = "https://api.data.gov.hk/v1/historical-archive"
SAMPLE = Path(__file__).with_name("hko_wind_2025.csv")
STATIONS = ("King's Park", "Cheung Chau")
FIELDS = ("archive_version", "date_time_hkt", "station", "mean_kmh")
RADIUS_M = 0.35
ROOT_M = 0.055
SPLIT_M = 0.18
GENERATOR_TARGET_RPM = 650
TSR_RANGE = (3.0, 5.0)
BLADES = 3
DESIGN_CL = 0.8
DESIGN_ALPHA_DEG = 6.0
AIR_DENSITY = 1.225
CP = 0.25
ELECTRICAL_EFFICIENCY = 0.60


def archive_request(endpoint, **params):
    query = urlencode({"url": HKO_WIND, **params})
    with urlopen(f"{ARCHIVE}/{endpoint}?{query}", timeout=30) as response:
        return response.read().decode("utf-8-sig")


def archive_versions(day):
    dates = json.loads(archive_request("list-file-versions", start=day, end=day))
    versions = [v for v in dates["timestamps"] if v.startswith(day + "-")]
    for hour in (0, 6, 12, 18):
        target = hour * 60 + 15

        def distance(version):
            return abs(int(version[-4:-2]) * 60 + int(version[-2:]) - target)

        closest = min(versions, key=distance, default=None)
        if closest is None or distance(closest) > 10:
            raise ValueError(f"no archived wind snapshot near {day} {hour:02d}:15")
        yield closest


def archive_samples(version):
    raw = archive_request("get-file", time=version)
    for row in csv.DictReader(io.StringIO(raw)):
        station = row["Automatic Weather Station"]
        if station in STATIONS:
            speed = row["10-Minute Mean Speed(km/hour)"]
            if speed not in ("", "N/A"):
                yield version, row["Date time"], station, speed


def fetch_samples(year, output):
    days = [f"{year:04d}{month:02d}{day:02d}" for month in range(1, 13) for day in (5, 20)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        versions = [version for batch in pool.map(lambda day: list(archive_versions(day)), days)
                    for version in batch]
        rows = [row for batch in pool.map(lambda version: list(archive_samples(version)), versions)
                for row in batch]
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(sorted(rows))


def load_speeds(path, station):
    with path.open(newline="") as stream:
        speeds = np.array([float(row["mean_kmh"]) / 3.6 for row in csv.DictReader(stream)
                           if row["station"] == station])
    if not np.all(np.isfinite(speeds)) or np.any(speeds < 0):
        raise ValueError("wind speeds must be finite and non-negative")
    return speeds


def wind_design(speeds):
    moving = speeds[speeds > 0]
    if len(moving) < 3 or np.ptp(moving) == 0:
        raise ValueError("need at least three non-calm samples with varying wind speeds")
    shape, _, scale = weibull_min.fit(moving, floc=0)
    energy_mode = scale * (1 + 2 / shape) ** (1 / shape)
    return shape, scale, energy_mode


def design_tsr(wind_mps):
    return float(np.clip(GENERATOR_TARGET_RPM * (2 * np.pi / 60) * RADIUS_M / wind_mps,
                         *TSR_RANGE))


def schmitz(radius_m, tsr):
    phi = (2 / 3) * np.arctan2(RADIUS_M, tsr * radius_m)
    chord_m = 16 * np.pi * radius_m * np.sin(phi / 2) ** 2 / (BLADES * DESIGN_CL)
    twist_rad = phi - np.deg2rad(DESIGN_ALPHA_DEG)
    return chord_m, twist_rad


def blade_mesh(start_m, end_m, tsr):
    radii = np.linspace(start_m, end_m, 25)
    chord_mm, twist = schmitz(radii, tsr)
    chord_mm *= 1000
    x = (1 - np.cos(np.linspace(0, np.pi, 49))) / 2
    camber = np.where(x < 0.4, 0.04 / 0.4**2 * (0.8 * x - x**2),
                      0.04 / 0.6**2 * (0.2 + 0.8 * x - x**2))
    half_thickness = 5 * 0.15 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                                + 0.2843 * x**3 - 0.1036 * x**4)
    outline_x = np.r_[x[::-1], x[1:]]
    vertices = []
    for radius, chord, angle in zip(radii, chord_mm, twist):
        # Use a 1 mm trailing edge instead of an unprintable mathematical point.
        thickness = half_thickness * chord + 0.5 * x
        section_x = (outline_x - 0.25) * chord
        section_z = np.r_[(camber * chord + thickness)[::-1], (camber * chord - thickness)[1:]]
        vertices.extend(zip(section_x * np.cos(angle) - section_z * np.sin(angle),
                            np.full(len(outline_x), radius * 1000),
                            section_x * np.sin(angle) + section_z * np.cos(angle)))

    width = len(outline_x)
    faces = []
    for ring in range(len(radii) - 1):
        for i in range(width):
            a = ring * width + i
            b = ring * width + (i + 1) % width
            faces.extend(((a, b, a + width), (b, b + width, a + width)))
    last = (len(radii) - 1) * width
    # Close each cap with strips between corresponding upper and lower points.
    leading_edge = len(x) - 1
    for i in range(leading_edge):
        upper, lower = leading_edge - i, leading_edge + i
        cap = [(upper, lower + 1, upper - 1)]
        if i:
            cap.append((upper, lower, lower + 1))
        for a, b, c in cap:
            faces.extend(((a, c, b), (last + a, last + b, last + c)))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help=f"input CSV (default: {SAMPLE.name})")
    parser.add_argument("--fetch-year", type=int, help="download archived HKO samples to --data")
    parser.add_argument("--station", default=STATIONS[0])
    parser.add_argument("--output-dir", type=Path, default=Path("turbine-output"))
    args = parser.parse_args()
    if args.fetch_year is not None and args.data is None:
        parser.error("--fetch-year needs an explicit --data destination")
    try:
        if args.fetch_year is not None:
            fetch_samples(args.fetch_year, args.data)
        speeds = load_speeds(args.data or SAMPLE, args.station)
        shape, scale, design_speed = wind_design(speeds)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    tsr = design_tsr(design_speed)
    rotor_rpm = tsr * design_speed / RADIUS_M * 60 / (2 * np.pi)
    shaft_watts = 0.5 * AIR_DENSITY * np.pi * RADIUS_M**2 * design_speed**3 * CP
    print(f"{args.station}: {len(speeds)} samples ({np.count_nonzero(speeds == 0)} calm), "
          f"Weibull k={shape:.2f}, scale={scale:.2f} m/s")
    print(f"Energy-mode design wind: {design_speed:.2f} m/s")
    print(f"Rotor: {BLADES} blades, radius={RADIUS_M:.2f} m, TSR={tsr:.2f}, {rotor_rpm:.0f} rpm")
    print(f"At design wind (estimated): {shaft_watts:.2f} W shaft, "
          f"{shaft_watts * ELECTRICAL_EFFICIENCY:.2f} W electrical")
    print("Radius (mm)  Chord (mm)  Twist (deg)")
    for radius in (ROOT_M, 0.10, SPLIT_M, 0.25, RADIUS_M):
        chord, twist = schmitz(radius, tsr)
        print(f"{radius * 1000:11.0f}  {chord * 1000:10.1f}  {np.rad2deg(twist):11.1f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, start, end in (("inner", ROOT_M, SPLIT_M), ("outer", SPLIT_M, RADIUS_M)):
        mesh = blade_mesh(start, end, tsr)
        path = args.output_dir / f"blade-{name}.stl"
        mesh.export(path)
        print(f"{path}: " + " × ".join(f"{extent:.1f}" for extent in mesh.extents) + " mm")


if __name__ == "__main__":
    main()
