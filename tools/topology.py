"""Coarse, self-weight-like compression optimisation of a printable solid (mm, Z up)."""

import time

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg
import trimesh


def _hex_stiffness():
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    signs = 2 * corners - 1
    nu = 0.3
    elasticity = np.zeros((6, 6))
    elasticity[:3, :3] = nu
    np.fill_diagonal(elasticity[:3, :3], 1 - nu)
    elasticity[3:, 3:] = np.eye(3) * (1 - 2 * nu) / 2
    elasticity /= (1 + nu) * (1 - 2 * nu)
    stiffness = np.zeros((24, 24))
    for point in np.array(np.meshgrid(*([[-1, 1]] * 3))).reshape(3, -1).T / np.sqrt(3):
        grad = np.empty((8, 3))
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            grad[:, axis] = signs[:, axis] * np.prod(1 + signs[:, other] * point[other], axis=1) / 4
        b = np.zeros((6, 24))
        for n, (dx, dy, dz) in enumerate(grad):
            b[:, 3*n:3*n+3] = [[dx, 0, 0], [0, dy, 0], [0, 0, dz],
                                 [dy, dx, 0], [0, dz, dy], [dz, 0, dx]]
        stiffness += b.T @ elasticity @ b / 8
    return corners, stiffness.astype(np.float32)


def _densities(solid, passive, iterations, fraction, deadline):
    cells = np.argwhere(solid)
    corners, ke = _hex_stiffness()
    node_shape = np.array(solid.shape) + 1
    nodes = np.ravel_multi_index((cells[:, None, :] + corners).transpose(2, 0, 1), node_shape)
    dofs = (3 * nodes[:, :, None] + np.arange(3)).reshape(-1, 24).astype(np.int32)
    count = int(3 * np.prod(node_shape))
    fixed_nodes = np.unique(nodes[cells[:, 2] == cells[:, 2].min()].ravel())
    fixed = (3 * fixed_nodes[:, None] + np.arange(3)).ravel()
    used = np.unique(dofs)
    free = np.setdiff1d(used, fixed)
    mapping = np.full(count, -1, dtype=np.int32)
    mapping[free] = np.arange(len(free))
    local = mapping[dofs]
    rows = np.broadcast_to(local[:, :, None], (len(cells), 24, 24)).ravel().copy()
    cols = np.broadcast_to(local[:, None, :], (len(cells), 24, 24)).ravel().copy()
    valid = (rows >= 0) & (cols >= 0)
    rows, cols = rows[valid], cols[valid]
    above = np.zeros_like(solid)
    above[:, :, :-1] = solid[:, :, 1:]
    top = ~above[tuple(cells.T)]
    force = np.zeros(count, dtype=np.float32)
    np.add.at(force, (3 * nodes[top, 4:] + 2).ravel(), -1 / max(1, top.sum() * 4))
    force = force[free]
    passive_cells = passive[tuple(cells.T)]
    design = ~passive_cells
    density = np.where(passive_cells, 1., fraction)
    kernel = np.maximum(0, 1.5 - np.sqrt(np.sum((np.indices((3, 3, 3)) - 1)**2, axis=0)))
    weight = ndimage.convolve(solid.astype(float), kernel, mode="constant")
    indices = tuple(cells.T)
    def physical_density(values):
        field = np.zeros(solid.shape)
        field[indices] = values
        filtered = (ndimage.convolve(field, kernel, mode="constant") / np.maximum(weight, 1e-12))[indices]
        filtered[passive_cells] = 1
        return filtered

    volume_field = np.zeros(solid.shape)
    volume_field[indices] = design / weight[indices]
    volume_gradient = ndimage.convolve(volume_field, kernel, mode="constant")[indices]
    displacement = np.zeros(len(free), dtype=np.float32)
    for _ in range(iterations):
        if time.monotonic() > deadline:
            raise RuntimeError("time budget exceeded")
        physical = physical_density(density)
        values = ((1e-6 + physical**3)[:, None, None] * ke).astype(np.float32).ravel()[valid]
        matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(free), len(free))).tocsr()
        preconditioner = sparse.diags(1 / np.maximum(matrix.diagonal(), 1e-12))
        def check_deadline(_):
            if time.monotonic() > deadline:
                raise RuntimeError("time budget exceeded")
        displacement, status = cg(matrix, force, x0=displacement, M=preconditioner,
                                  rtol=2e-3, atol=0, maxiter=250, callback=check_deadline)
        if status < 0 or not np.isfinite(displacement).all():
            raise RuntimeError("elasticity solve failed")
        full = np.zeros(count, dtype=np.float32)
        full[free] = displacement
        element_u = full[dofs]
        energy = np.einsum("ei,ij,ej->e", element_u, ke, element_u, optimize=False)
        sensitivity = -3 * physical**2 * np.maximum(energy, 1e-20)
        sensitivity[passive_cells] = 0
        field = np.zeros(solid.shape)
        field[indices] = sensitivity / weight[indices]
        sensitivity = ndimage.convolve(field, kernel, mode="constant")[indices]
        low, high = 0., max(1., float(-sensitivity.min()) * 10000)
        for _ in range(50):
            mid = (low + high) / 2
            candidate = np.clip(density * np.sqrt(np.maximum(0, -sensitivity)
                                / (mid * np.maximum(volume_gradient, 1e-12))),
                                np.maximum(.001, density - .2), np.minimum(1, density + .2))
            candidate[passive_cells] = 1
            if physical_density(candidate)[design].sum() > fraction * design.sum():
                low = mid
            else:
                high = mid
        density = candidate
    result = np.ones(solid.shape)
    result[indices] = physical_density(density)
    return result


def optimise(mesh, shell_mm=2.0, volume_fraction=0.4, iterations=30):
    """Subtract enclosed low-stress cavities, or return the input unchanged on failure.

    The coarse grid targets 24 cells along the longest axis, capped at 6000
    solid cells to bound sparse FEM storage on the print box.
    """
    started = time.monotonic()
    try:
        if not mesh.is_volume or not len(mesh.faces):
            return mesh, {"skipped": "input is not a closed solid"}
        if shell_mm <= 0 or not 0 < volume_fraction < 1 or not 1 <= iterations <= 30:
            return mesh, {"skipped": "invalid optimisation parameters"}
        pitch = float(mesh.extents.max()) / 24
        grid = mesh.voxelized(pitch).fill()
        while np.count_nonzero(grid.matrix) > 6000:
            pitch *= 1.1
            grid = mesh.voxelized(pitch).fill()
            if time.monotonic() - started > 10:
                return mesh, {"skipped": "voxelisation time budget exceeded"}
        solid = np.pad(grid.matrix, 1)
        distance = ndimage.distance_transform_edt(solid) * pitch
        # Account for both the surface voxel and the furthest corner of a void voxel.
        interior = distance > shell_mm + np.sqrt(3) * pitch
        z = grid.transform[2, 3] + (np.arange(solid.shape[2]) - 1) * pitch
        interior[:, :, z - pitch / 2 < mesh.bounds[0, 2] + 3] = False
        if not interior.any():
            return mesh, {"skipped": "no designable interior at this resolution"}
        density = _densities(solid, solid & ~interior, iterations, volume_fraction, started + 50)
        void = interior & (density < .3)
        labels, _ = ndimage.label(void)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= max(3, int(np.ceil(8 / pitch**3)))
        keep[0] = False
        void = keep[labels]
        if not void.any():
            return mesh, {"skipped": "no significant low-density cavities"}
        from skimage.measure import marching_cubes

        vertices, faces, _, _ = marching_cubes(void.astype(np.float32), .5, spacing=(pitch,) * 3)
        vertices += grid.transform[:3, 3] - pitch
        cavity = trimesh.Trimesh(vertices=vertices, faces=faces)
        cavity.fix_normals()
        result = trimesh.boolean.difference([mesh, cavity], engine="manifold")
        if not result.is_volume or not np.allclose(result.bounds, mesh.bounds, atol=1e-4):
            return mesh, {"skipped": "subtraction did not preserve the closed outer surface"}
        before, after = float(mesh.volume), float(result.volume)
        saved = 100 * (before - after) / before
        if saved < 5:
            return mesh, {"skipped": "less than 5% volume saving"}
        if time.monotonic() - started > 55:
            return mesh, {"skipped": "time budget exceeded"}
        return result, {"volume_saved_pct": round(saved, 2),
                        "volume_cm3_before": round(before / 1000, 1),
                        "volume_cm3_after": round(after / 1000, 1)}
    except Exception as exc:
        return mesh, {"skipped": f"optimisation failed: {exc}"}
