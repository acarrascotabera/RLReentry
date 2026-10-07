"""Oblate-ellipsoid geometry: altitude above the WGS-84 surface and great-circle
distance helpers.

altitude port of reentry_simulator/guidance/physics/ellipsoidal_altitude.m:
    surface_radius(lat) = a*b / sqrt((b cos lat)^2 + (a sin lat)^2)
    altitude            = r_nd * R0 - surface_radius(lat)
"""
import numpy as np

from .constants import R0, A_ELLIPSOID, B_ELLIPSOID


def surface_radius_m(lat_rad):
    """Geocentric radius of the ellipsoid surface at geocentric latitude lat."""
    a, b = A_ELLIPSOID, B_ELLIPSOID
    s = np.sin(lat_rad)
    c = np.cos(lat_rad)
    return (a * b) / np.sqrt((b * c) ** 2 + (a * s) ** 2)


def altitude_m(r_nd, lat_rad):
    """Ellipsoidal altitude [m] from non-dim radius and latitude."""
    return r_nd * R0 - surface_radius_m(lat_rad)


def r_nd_from_alt(alt_m, lat_rad):
    """Inverse: non-dim radius from ellipsoidal altitude and latitude."""
    return (surface_radius_m(lat_rad) + alt_m) / R0


def great_circle_distance_m(lat1, lon1, lat2, lon2, radius=R0):
    """Haversine great-circle distance [m] between two (lat, lon) points [rad]."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(s, 0.0, 1.0)))


def bearing_rad(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing [rad] from point 1 to point 2 (0 = North, +East)."""
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(x, y)
