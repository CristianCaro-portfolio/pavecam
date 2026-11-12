"""RWIS sites in Iowa that have a camera pointed at the instrumented pavement.

Each entry pairs an Iowa DOT RWIS station (pavement sensor, the label source) with
the camera view at the same site that looks at the bridge deck or the sensor.
Camera ids and names come from the IEM webcam inventory for network IDOT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    rwis_id: str
    rwis_name: str
    camera_id: str
    camera_name: str
    lat: float
    lon: float


SITES: tuple[Site, ...] = (
    Site("RDAI4", "Adair (I-80)", "IDOT-000-02", "Adair I80 Bridge Deck", 41.4963, -94.7263),
    Site("RALI4", "Algona (US 18)", "IDOT-001-00", "Algona US18 Bridge Deck", 43.0830, -94.3985),
    Site("RTNI4", "Alton (IA10)", "IDOT-002-00", "Alton IA Hwy10 Bridge Deck", 42.9978, -96.0843),
    Site("RCII4", "Cedar Rapids (US 30)", "IDOT-010-02", "Cedar Rapids Hwy 13 US 30 Bridge Deck", 41.9300, -91.6838),
    Site("RCNI4", "Centerville (IA 2)", "IDOT-011-01", "Centerville Bridge Deck", 40.7397, -93.0027),
    Site("RDCI4", "Decorah (IA 9)", "IDOT-015-00", "Decorah IA 9 Sensor", 43.2455, -91.6997),
    Site("RMCI4", "Mason City (I-35)", "IDOT-030-01", "Mason City I-35 Deck", 43.0380, -93.3418),
    Site("RNHI4", "New Hampton (US 18)", "IDOT-033-01", "New Hampton US 18 Bridge Deck", 43.0672, -92.4592),
    Site("ROTI4", "Ottumwa (US 63)", "IDOT-036-00", "Ottumwa Bridge Deck", 41.0168, -92.4212),
    Site("RPLI4", "Pella (IA 163)", "IDOT-037-03", "Pella WB Bridge Deck", 41.3902, -92.8725),
    Site("RSPI4", "Spencer (US 18)", "IDOT-042-00", "Spencer US 18 Bridge Deck", 43.1268, -95.0888),
    Site("RWLI4", "Waterloo (US 20)", "IDOT-046-00", "Waterloo I-380 and US20 Bridge Deck", 42.4557, -92.3090),
    Site("RDYI4", "Eddyville", "IDOT-056-02", "Eddyville Bridge Deck", 41.1465, -92.6436),
    Site("RLEI4", "Leon (I-35/IA 2)", "IDOT-026-03", "Leon IA 2 Bridge Deck", 40.7387, -93.8417),
    Site("RPFI4", "Plainfield", "IDOT-058-02", "Plainfield US 218 NB Deck", 42.8332, -92.5371),
    Site("RJFI4", "Jefferson (IA 4)", "IDOT-025-02", "Jefferson IA 4 Bridge", 42.0518, -94.3783),
    Site("RMNI4", "Manchester (US 20)", "IDOT-027-03", "Manchester US 20 Bridge", 42.4710, -91.4473),
    Site("RTPI4", "Tipton (I-80)", "IDOT-044-01", "Tipton I-80 East and West Bridge Deck", 41.6448, -91.1272),
    Site("RWII4", "Williams (I-35)", "IDOT-047-03", "Williams NB Bridge Deck", 42.5875, -93.5315),
)

SITE_BY_RWIS = {s.rwis_id: s for s in SITES}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_station(lat: float, lon: float, stations: dict[str, tuple[float, float]]) -> tuple[str, float]:
    """Return (station_id, distance_km) of the closest station in a {id: (lat, lon)} map."""
    if not stations:
        raise ValueError("no stations to choose from")
    best = min(stations.items(), key=lambda kv: haversine_km(lat, lon, *kv[1]))
    return best[0], haversine_km(lat, lon, *best[1])
