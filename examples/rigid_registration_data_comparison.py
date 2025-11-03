"""Rigid registration comparison on sample data.

This script compares multiple rigid registration algorithms on the
`data/source.ply` and `data/target.ply` point clouds provided with the
project. The final rigid transformation matrices and execution times are
printed for each method. The evaluated methods include CPD, GMMReg,
GMMTree, and FilterReg with both point-to-point and point-to-plane
objectives.
"""
import time
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import open3d as o3
from probreg import cpd, filterreg, gmmtree
from probreg.l2dist_regs import registration_gmmreg


def read_point_cloud(path: Path, voxel_size: float) -> o3.geometry.PointCloud:
    """Load and down-sample a point cloud from disk."""
    point_cloud = o3.io.read_point_cloud(str(path))
    if voxel_size > 0.0:
        point_cloud = point_cloud.voxel_down_sample(voxel_size)
    return point_cloud


def transformation_to_matrix(transformation) -> np.ndarray:
    """Convert a probreg transformation object to a 4x4 matrix."""
    rotation = np.asarray(transformation.rot)
    translation = np.asarray(transformation.t).reshape(3)
    scale = float(np.asarray(transformation.scale)) if hasattr(transformation, "scale") else 1.0
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def run_registration(method_name: str, runner: Callable[[], object]) -> Tuple[str, np.ndarray, float]:
    """Execute a registration routine and measure its runtime."""
    start = time.perf_counter()
    transformation = runner()
    elapsed = time.perf_counter() - start
    matrix = transformation_to_matrix(transformation)
    return method_name, matrix, elapsed


def main() -> None:
    """Perform rigid registration with multiple algorithms and compare them."""
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data"
    source_path = data_dir / "source.ply"
    target_path = data_dir / "target.ply"
    # Use a moderate voxel size to limit the number of points and memory usage.
    voxel_size = 0.5

    # Ensure that both point cloud files exist before continuing.
    for path in (source_path, target_path):
        if not path.is_file():
            raise FileNotFoundError(f"Point cloud file not found: {path}")

    source_pcd = read_point_cloud(source_path, voxel_size)
    target_pcd = read_point_cloud(target_path, voxel_size)

    # Estimate normals for the target cloud to enable point-to-plane registration.
    if not target_pcd.has_normals():
        search_radius = max(voxel_size * 2.0, 0.01)
        target_pcd.estimate_normals(
            o3.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
        )
        # Enforce consistent normal orientation for stable point-to-plane alignment.
        target_pcd.orient_normals_consistent_tangent_plane(30)
        target_pcd.normalize_normals()

    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    target_normals = np.asarray(target_pcd.normals)

    print("Loaded point clouds:")
    print(f"  Source points: {len(source_points)}")
    print(f"  Target points: {len(target_points)}")

    methods = [
        (
            "CPD (rigid)",
            lambda: cpd.registration_cpd(
                source_points, target_points, tf_type_name="rigid", update_scale=False
            )[0],
        ),
        (
            "GMMReg (rigid)",
            lambda: registration_gmmreg(
                source_points, target_points, tf_type_name="rigid"
            ),
        ),
        (
            "GMMTree",
            lambda: gmmtree.registration_gmmtree(source_points, target_points).transformation,
        ),
        (
            "FilterReg (point-to-point)",
            lambda: filterreg.registration_filterreg(
                source_points, target_points, objective_type="pt2pt", update_sigma2=True
            )[0],
        ),
        (
            "FilterReg (point-to-plane)",
            lambda: filterreg.registration_filterreg(
                source_points,
                target_points,
                target_normals=target_normals,
                objective_type="pt2pl",
                update_sigma2=True,
            )[0],
        ),
    ]

    results = [run_registration(name, runner) for name, runner in methods]

    print("\nRegistration results:")
    for name, matrix, elapsed in results:
        print(f"Method: {name}")
        print(f"Elapsed time: {elapsed:.4f} [sec]")
        print("Transformation matrix:")
        print(np.array2string(matrix, precision=6, suppress_small=True))
        print()

    print("Summary of execution times:")
    for name, _, elapsed in results:
        print(f"  {name}: {elapsed:.4f} [sec]")

    fastest = min(results, key=lambda item: item[2])
    slowest = max(results, key=lambda item: item[2])
    print(
        f"Fastest method: {fastest[0]} ({fastest[2]:.4f} [sec])\n"
        f"Slowest method: {slowest[0]} ({slowest[2]:.4f} [sec])"
    )


if __name__ == "__main__":
    main()
