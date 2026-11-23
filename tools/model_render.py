"""Small CPU perspective mesh rasterizer, using OpenCV camera coordinates."""

import numpy as np
from PIL import Image


def render(vertices, faces, K, R, t, width, height):
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3) @ np.asarray(R).T + np.asarray(t)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    K = np.asarray(K, dtype=float)
    if width <= 0 or height <= 0 or not np.isfinite(vertices).all() or not np.isfinite(K).all():
        raise ValueError("invalid render geometry or dimensions")
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    depth = np.full((height, width), np.inf)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centres = triangles.mean(axis=1)
    cosine = np.abs(np.einsum("ij,ij->i", normals, centres)) / np.maximum(
        np.linalg.norm(normals, axis=1) * np.linalg.norm(centres, axis=1), 1e-20)
    shades = (205 * (0.45 + 0.55 * cosine)).astype(np.uint8)
    near = 1e-5
    raster_triangles, raster_shades = [], []
    for triangle, shade in zip(triangles, shades):
        if np.all(triangle[:, 2] < near):
            continue
        if np.all(triangle[:, 2] >= near):
            raster_triangles.append(triangle)
            raster_shades.append(shade)
            continue
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            if a[2] >= near:
                polygon.append(a)
            if (a[2] >= near) != (b[2] >= near):
                polygon.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
        for i in range(1, len(polygon) - 1):
            raster_triangles.append([polygon[0], polygon[i], polygon[i + 1]])
            raster_shades.append(shade)
    if not raster_triangles:
        return Image.fromarray(canvas)
    triangles = np.asarray(raster_triangles)
    projected = triangles @ K.T
    xy = projected[:, :, :2] / projected[:, :, 2:3]
    lows = np.maximum(np.floor(xy.min(axis=1)), [0, 0]).astype(int)
    highs = np.minimum(np.ceil(xy.max(axis=1)), [width - 1, height - 1]).astype(int)
    for tri, coordinates, lo, hi, shade in zip(triangles, xy, lows, highs, raster_shades):
        if np.any(lo > hi):
            continue
        a, b, c = coordinates
        denominator = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(denominator) < 1e-12:
            continue
        x = np.arange(lo[0], hi[0] + 1)[None, :] + 0.5
        y = np.arange(lo[1], hi[1] + 1)[:, None] + 0.5
        w1 = ((x - a[0]) * (c[1] - a[1]) - (y - a[1]) * (c[0] - a[0])) / denominator
        w2 = ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) / denominator
        w0 = 1 - w1 - w2
        inverse_z = w0 / tri[0, 2] + w1 / tri[1, 2] + w2 / tri[2, 2]
        z = np.divide(1, inverse_z, out=np.full_like(inverse_z, np.inf), where=inverse_z > 0)
        region = np.s_[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        visible = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (z < depth[region])
        depth[region][visible] = z[visible]
        canvas[region][visible] = shade
    return Image.fromarray(canvas)
