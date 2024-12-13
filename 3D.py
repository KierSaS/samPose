import open3d as o3d

mesh = o3d.io.read_triangle_mesh("/home/wubin/code/POPE/data/Tless/tless_models/models_cad/obj_000013.ply")
if mesh.is_empty():
    print("Failed to load the .ply file.")
else:
    print("Successfully loaded the .ply file!")
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of faces: {len(mesh.triangles)}")
    if mesh.has_vertex_colors():
        print("The mesh contains vertex colors.")
    else:
        print("The mesh does not contain vertex colors.")

    o3d.visualization.draw_geometries([mesh])
