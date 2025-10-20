"""
Convert USD mesh to OBJ format for MuJoCo.

This script converts furniture-bench USD meshes to OBJ format,
which should preserve the mesh geometry better than the existing OBJ files.
"""

from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
import trimesh


def convert_usd_to_obj(usd_path: Path, obj_path: Path):
    """
    Convert USD file to OBJ format.

    Args:
        usd_path: Path to input USD file
        obj_path: Path to output OBJ file
    """
    print(f"Loading USD: {usd_path}")

    # Open USD stage
    stage = Usd.Stage.Open(str(usd_path))

    # Find all mesh prims
    vertices_list = []
    faces_list = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)

            # Get mesh data
            points = mesh.GetPointsAttr().Get()
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

            if points is None or face_vertex_indices is None:
                continue

            # Convert to numpy
            vertices = np.array(points, dtype=np.float64)
            indices = np.array(face_vertex_indices, dtype=np.int32)

            print(f"  Mesh '{prim.GetName()}': {len(vertices)} vertices")

            # Convert face indices (handle triangulation if needed)
            faces = []
            idx = 0
            for count in face_vertex_counts:
                if count == 3:
                    # Triangle - add directly
                    faces.append([indices[idx], indices[idx+1], indices[idx+2]])
                elif count == 4:
                    # Quad - triangulate
                    faces.append([indices[idx], indices[idx+1], indices[idx+2]])
                    faces.append([indices[idx], indices[idx+2], indices[idx+3]])
                else:
                    print(f"  Warning: {count}-sided polygon, skipping")
                idx += count

            # Offset face indices for combined mesh
            faces = np.array(faces, dtype=np.int32) + vertex_offset

            vertices_list.append(vertices)
            faces_list.append(faces)
            vertex_offset += len(vertices)

    # Combine all meshes
    if not vertices_list:
        raise ValueError("No meshes found in USD file")

    all_vertices = np.vstack(vertices_list)
    all_faces = np.vstack(faces_list)

    print(f"  Total vertices: {len(all_vertices)}")
    print(f"  Total faces: {len(all_faces)}")

    # Create trimesh and export
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    print(f"Exporting to: {obj_path}")
    mesh.export(str(obj_path))
    print("  Done!")


if __name__ == "__main__":
    # Paths
    furniture_bench_dir = Path("/Users/larsankile/code/furniture-bench")
    mesh_dir = furniture_bench_dir / "furniture_bench/assets/furniture/mesh/square_table"

    # Convert square table top
    usd_file = mesh_dir / "square_table_top.usd"
    obj_file = mesh_dir / "square_table_top_fixed.obj"

    if usd_file.exists():
        convert_usd_to_obj(usd_file, obj_file)
        print(f"\nFixed mesh saved to: {obj_file}")
        print("Update furniture_assembly.xml to use square_table_top_fixed.obj")
    else:
        print(f"Error: USD file not found: {usd_file}")
