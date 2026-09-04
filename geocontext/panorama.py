"""Reproject equirectangular panoramas into rectilinear (perspective) crops.

Mapillary returns some images as full 360-degree panoramas. Feeding those to a
model directly is unfair in both directions: the whole scene is visible at once,
which no ordinary photograph offers, and the projection distorts buildings badly
near the edges.

This module cuts a panorama into overlapping perspective views with a plausible
field of view, so that panorama-derived and camera-derived images pose the same
task to the model.
"""
import math

import numpy as np
from PIL import Image


def is_equirectangular(img: Image.Image, tol=0.05) -> bool:
    """Equirectangular images always have a 2:1 aspect ratio."""
    w, h = img.size
    return abs(w / h - 2.0) < tol


def to_perspective(src, yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0,
                   out_size=(1024, 768)) -> Image.Image:
    """Cut one perspective view out of an equirectangular panorama.

    yaw   0 = centre of the panorama; positive turns right.
    pitch 0 = horizontal; **positive is up**. The lower half of a vehicle-mounted
              panorama is the car roof, so tilting up slightly avoids it.
    fov   horizontal field of view; 90 degrees is close to a phone main camera.
    """
    if isinstance(src, (str, bytes)) or hasattr(src, "__fspath__"):
        src = Image.open(src)
    src = src.convert("RGB")
    eq = np.asarray(src)
    eh, ew = eq.shape[:2]
    ow, oh = out_size

    # Direction vector for each pixel of the output plane (camera faces +z)
    f = 0.5 * ow / math.tan(math.radians(fov_deg) / 2)
    xs = np.arange(ow) - (ow - 1) / 2
    ys = np.arange(oh) - (oh - 1) / 2
    xx, yy = np.meshgrid(xs, ys)
    vx, vy, vz = xx, yy, np.full_like(xx, f, dtype=float)

    # Pitch about the x axis first, then yaw about the y axis
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    vy2 = vy * math.cos(p) - vz * math.sin(p)
    vz2 = vy * math.sin(p) + vz * math.cos(p)
    vx3 = vx * math.cos(y) + vz2 * math.sin(y)
    vz3 = -vx * math.sin(y) + vz2 * math.cos(y)

    # Direction vector -> lat/lon -> pixel coordinates in the equirectangular image
    norm = np.sqrt(vx3 ** 2 + vy2 ** 2 + vz3 ** 2)
    lon = np.arctan2(vx3 / norm, vz3 / norm)
    lat = np.arcsin(np.clip(vy2 / norm, -1, 1))
    u = ((lon / (2 * math.pi) + 0.5) * ew).astype(np.int32) % ew
    v = np.clip(((lat / math.pi + 0.5) * eh).astype(np.int32), 0, eh - 1)
    return Image.fromarray(eq[v, u])


def sky_ground_fraction(img: Image.Image) -> tuple[float, float]:
    """Rough estimate of sky fraction and "large flat area below" fraction,
    used to reject crops that are all sky or all car roof."""
    a = np.asarray(img.convert("RGB").resize((64, 48)), dtype=float)
    top, bot = a[:12], a[-12:]
    # Sky: bright, with the blue channel dominant
    sky = float(((top[..., 2] > top[..., 0]) & (top.mean(-1) > 120)).mean())
    # Car roof / road surface: very low colour variance in the lower region
    flat = float((bot.reshape(-1, 3).std(0).mean() < 18))
    return sky, flat


def crops(src, n=4, pitch_deg=8.0, fov_deg=90.0, out_size=(1024, 768),
          start_yaw=0.0):
    """Take n evenly spaced crops. Returns [(yaw, Image), ...].

    pitch defaults to +8 degrees: the lower half of a vehicle panorama is the
    car roof, and a level view cuts a large piece of it into frame.
    """
    step = 360.0 / n
    return [(start_yaw + i * step,
             to_perspective(src, start_yaw + i * step, pitch_deg, fov_deg, out_size))
            for i in range(n)]


def best_crops(src, n=2, candidates=8, pitch_deg=8.0, fov_deg=90.0,
               out_size=(1024, 768)):
    """Pick the n directions with the most streetscape content out of `candidates`.

    Score: subtract the sky fraction, subtract the large flat region below (car
    roof / road), add edge density -- directions with buildings and signage have
    more edges.
    """
    scored = []
    for yaw, im in crops(src, n=candidates, pitch_deg=pitch_deg,
                         fov_deg=fov_deg, out_size=out_size):
        g = np.asarray(im.convert("L").resize((160, 120)), dtype=float)
        edge = float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())
        sky, flat = sky_ground_fraction(im)
        scored.append((edge - 40 * sky - 25 * flat, yaw, im))
    scored.sort(key=lambda t: -t[0])
    return [(yaw, im) for _, yaw, im in scored[:n]]
