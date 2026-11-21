# Hong Kong wind turbine: first prototype

**Build a 700 mm diameter, three-blade horizontal-axis rotor around King's Park wind observations.** The deliverable is a repeatable wind-to-STL design and two printable blade segments for fit and material trials. The files have no structural join or mounting holes; they are **not safe to rotate as supplied**. No physical print, electrical output, or outdoor installation has been verified.

## Hong Kong wind data and design point

The [HKO regional wind feed](https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind.csv) gives 10-minute mean and gust speeds. The live feed was checked but is only a snapshot; the fit uses [DATA.GOV.HK's historical archive](https://api.data.gov.hk/v1/historical-archive/list-file-versions?url=https%3A%2F%2Fdata.weather.gov.hk%2FweatherAPI%2Fhko_data%2Fregional-weather%2Flatest_10min_wind.csv&start=20250101&end=20250101) of that same feed. The [cached sample](hko_wind_2025.csv) preserves the archive version and HKO observation time. For the 5th and 20th of every month of 2025, it uses four archived files nearest 00:15, 06:15, 12:15 and 18:15 HKT; the actual observation times precede the archive times. Only the mean speed is fitted, converted from km/h to m/s.

| HKO station | 10-min samples | Sample mean | Weibull shape / scale | Energy-mode wind | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| King's Park | 96 | 3.00 m/s | 1.86 / 3.40 m/s | **5.04 m/s** | Urban proxy, not a rooftop measurement |
| Cheung Chau | 96 | 5.38 m/s | 1.47 / 6.01 m/s | 10.77 m/s | Exposed coastal comparison, not the operating target |

The Weibull fit fixes location at zero and fits positive observations; calms, if any, are counted separately. The design speed maximizes `v³ × WeibullPDF(v)`: `v = scale × (1 + 2/shape)^(1/shape)`. This is the mode of the wind energy contribution per speed bin, not the most common speed or a turbine power-curve optimum. Dropping July 20, the highest-wind day in the sample, moves the modes to 4.32 and 8.53 m/s respectively. This sparse, one-year sample is not an annual energy or extreme-gust assessment. HKO's [1991–2020 normals, Table 7](https://www.hko.gov.hk/en/cis/normal/1991_2020/normals.htm#table7) independently show the exposure difference: annual means of 9.9 km/h at the Observatory and 22.9 km/h at Waglan Island (different stations; not used in the fit).

## Rotor and expected output

Use a yawing horizontal-axis rotor for the guarded bench prototype: its lift-driven blades offer more power per swept area than a simple drag-driven Savonius rotor in steady inflow. A Savonius can self-start and accept changing azimuth without a yaw system, but its lower efficiency and cyclic loading make it a fallback if measurements show frequent direction changes. Neither choice makes a turbulent roof a suitable installation without site data.

- **Radius 0.35 m**, active span 55–350 mm. Match the nominal 650 rpm generator target at 5.04 m/s: `TSR = clip(ω_target × radius / wind, 3, 5)` gives **4.72**. The 650 rpm is a procurement assumption, not a measured generator characteristic. Changing the station changes the design wind, TSR and blade geometry.
- Use the Schmitz ideal blade-element/momentum design as a starting point: `φ = (2/3) × atan(1 / (TSR × r/R))`, `chord = 16πr sin²(φ/2) / (blade_count × Cl)`, `twist = φ − α`. Loft a cambered NACA 4415-style section with a 1 mm trailing edge for FDM. The assumed `Cl = 0.8` and `α = 6°` are placeholders until low-Reynolds-number polar and bench data are available; there is no iterative loss, stall or structural optimization.

  | Radius (mm) | Chord (mm) | Twist (°) |
  | ---: | ---: | ---: |
  | 55 | 107.7 | 29.6 |
  | 100 | 93.2 | 18.4 |
  | 180 (split) | 63.5 | 8.9 |
  | 250 | 48.1 | 5.0 |
  | 350 | 35.4 | 2.0 |

- At 5.04 m/s, `½ρπR²v³ × Cp` with `ρ = 1.225 kg/m³` and an assumed `Cp = 0.25` gives **7.56 W shaft / 4.53 W electrical** at assumed 60% drivetrain/electrical efficiency. These are point estimates, not measured output or an annual average. Generator cut-in, yaw error, profile roughness and turbulence can materially reduce output.

## Print plan and parts

- Print three copies of each half, with a common split plane at radius 180 mm: 125 mm inner span and 170 mm outer span. The halves fit a 256 × 256 mm plate without rescaling. Keep the span in the bed plane to align extrusion roads with spanwise loads; preview supports, brims and the 1 mm trailing edge in the slicer. Print one pair first to check the fit and weigh it.
- Prefer recycled PETG from a traceable filament lot for the prototype; use recycled PLA only for non-spinning fit/shape trials where heat and creep are not factors. Dry the PETG per supplier instructions, record recycled fraction and print settings, and test same-lot coupons for layer adhesion, creep and moisture/UV effects. Use a removable mechanical assembly so metal parts can be recovered.
- **BOM for the next iteration:** six printed segments; a captured metal spar per blade spanning the split; mechanically retained hub and fasteners; steel shaft and two sealed bearings (e.g., 608-2RS with an 8 mm shaft, subject to load checks); a low-cogging permanent-magnet generator characterized under load near 650 rpm; rectifier, dump load and RPM sensing; yaw mount/tail for steady-inflow tests; rigid bench fixture and containment guard. No adhesive-only joint.

## Reproduce and use the print pipeline

From the repo root, with Python 3.10+:

```sh
python -m pip install 'numpy>=2.0' 'scipy>=1.11' 'trimesh>=4.0' 'rtree>=1.0'
python -m turbine.design --output-dir turbine-output
python tools/printability.py turbine-output/blade-inner.stl --bed 256x256x256
python tools/printability.py turbine-output/blade-outer.stl --bed 256x256x256
```

The default is offline. To resample the archive, pass `--fetch-year 2025 --data path/to/new.csv`; to fit another file, pass `--data` and `--station` using the [cached CSV](hko_wind_2025.csv) column format. Test with `python -m pytest server/tests/test_turbine_design.py -q`.

Use the original millimetre STLs in the slicer at 100%. Reuse the pipeline's printability checker, not its photo reconstruction or `POST /print` mesh preparation: the latter flattens the bottom even with the base plate off and can alter this airfoil or split surface. No server change is required.

## Milestones and gates

1. **Design and print:** reproduce the fit and STLs, then slice and print one non-spinning pair in recycled PETG; inspect supports, mass and split fit.
2. **Guarded bench validation:** engineer the spar, hub and positive retention; test coupons and joints, balance the rotor, validate containment and overspeed shutdown, then measure cut-in and the power curve against RPM and load. Do not spin the as-supplied segments.
3. **Site decision:** acquire longer, higher-cadence hub-height data at an approved Hong Kong site (speed, direction, gusts and turbulence); re-fit, compare with a VAWT if yaw losses dominate, and validate mounting, noise, fatigue, debris containment and storm shutdown/removal before considering outdoor use. No typhoon operation or rooftop deployment is in scope.
