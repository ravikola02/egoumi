#!/usr/bin/env python3
"""Export a collection's dual-gripper scenes for the browser viewer (viewer/index.html, three.js).

    ~/anaconda3/envs/hawor/bin/python scripts/export_viewer.py --collection data/output/<c>
    python3 -m http.server 8765 --bind 127.0.0.1 --directory data/output/<c>/viewer

Per episode with <c>/preview/<ep>/preview_dual/scene.npz (scripts/dual_preview.py, --scene-only is enough):
  viewer/<ep>.json   fps, rows, table, head camera K + poses, per gripper: body poses (row-major 4x4, gravity-aligned
                     table frame: z up, origin on the table), visibility, TCP; gravity fit summary
  viewer/<ep>.mp4    the recording panel of dual_preview.mp4 (mesh overlay) if rendered, else the head clip (H.264)
viewer/meshes.json   one decimated mesh per gripper (body frame, jaws open; hidden internals dropped)
viewer/index.json    episode list
"""
import argparse, glob, json, os, shutil, subprocess, sys
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from umi_pipe.model import GripperModel, MARKER_CFG, load_meshes   # noqa: E402

FFMPEG = "/usr/bin/ffmpeg" if os.path.isfile("/usr/bin/ffmpeg") else "ffmpeg"
VISIBLE = ("bottom_plate", "top_cover", "cube", "grip", "finger_holder", "soft_finger")   # outside of the gripper


def gripper_mesh(name, faces):
    import open3d as o3d, trimesh
    m = GripperModel(MARKER_CFG, index=name); parts = m.part_transforms(0.0)
    keep = [mm.copy().apply_transform(parts.get(k, np.eye(4))) for k, mm in load_meshes(True, m).items()
            if k.startswith(VISIBLE)]
    cam = trimesh.creation.box(extents=[0.06, 0.03, 0.045])          # GoPro/Osmo stand-in on the top cover
    cam.apply_translation([0, 0.01, 0.0198 + 0.0225]); keep.append(cam)
    tm = trimesh.util.concatenate(keep)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces))
    om = om.simplify_quadric_decimation(faces); om.remove_degenerate_triangles(); om.remove_unreferenced_vertices()
    v = np.asarray(om.vertices); f = np.asarray(om.triangles)
    print(f"mesh {name}: {len(tm.faces)} -> {len(f)} faces")
    return dict(v=np.round(v, 5).ravel().tolist(), f=f.ravel().tolist())


def rows(T):
    return np.round(T.reshape(len(T), 16), 5).ravel().tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--faces", type=int, default=30000, help="faces per gripper mesh after decimation")
    ap.add_argument("--episodes", help="comma list (default: every episode with a scene.npz)")
    a = ap.parse_args()
    col = os.path.abspath(a.collection); out = os.path.join(col, "viewer"); os.makedirs(out, exist_ok=True)
    scenes = sorted(glob.glob(os.path.join(col, "preview", "*", "preview_dual", "scene.npz")))
    eps = [s.split(os.sep)[-3] for s in scenes]
    if a.episodes:
        want = a.episodes.split(","); scenes = [s for s, e in zip(scenes, eps) if e in want]; eps = [e for e in eps if e in want]

    mpath = os.path.join(out, "meshes.json")
    if not os.path.isfile(mpath):
        json.dump({nm: gripper_mesh(nm, a.faces) for nm in ("right", "left")}, open(mpath, "w"))
    for sc, ep in zip(scenes, eps):
        z = np.load(sc); seq = os.path.dirname(os.path.dirname(sc))
        cam = yaml.safe_load(open(os.path.join(seq, "camera.yaml")))
        G = {}
        for nm in z["grippers"]:
            vis = z[f"vis_{nm}"].astype(bool)
            G[str(nm)] = dict(T=rows(z[f"T_gb_{nm}"]), vis=vis.astype(int).tolist(),
                              tcp=np.round(z[f"tcp_{nm}"], 4).ravel().tolist(), coverage=round(100 * float(vis.mean()), 1))
        d = dict(episode=ep, fps=float(z["fps"]), n=int(z["n"]), table=dict(size=z["table_size"].tolist(),
                 offset=z["table_offset"].tolist()), cam=dict(w=cam["width"], h=cam["height"], fx=cam["fx"], fy=cam["fy"],
                 cx=cam["cx"], cy=cam["cy"], T=rows(z["cams"])), grippers=G, gravity=json.loads(str(z["gravity"])))
        json.dump(d, open(os.path.join(out, f"{ep}.json"), "w"))
        vid = os.path.join(seq, "preview_dual", "dual_preview.mp4"); dst = os.path.join(out, f"{ep}.mp4")
        rendered = os.path.isfile(vid) and int(subprocess.check_output(
            ["/usr/bin/ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_packets", "-of", "csv=p=0", vid]).decode().strip() or 0) >= d["n"] - 1
        if rendered:     # left panel of the 3-panel render: the recording with the mesh overlay
            w = int(subprocess.check_output(["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", vid]).decode().split(",")[1])
            ph = w; pw = int(round(ph * cam["width"] / cam["height"]))
            vf = f"crop={pw}:{ph}:0:0"; src = vid
        else:
            vf = "scale=960:-2"; src = os.path.join(col, "episode_clips", f"{ep}_head.mp4")
        subprocess.check_call([FFMPEG, "-v", "error", "-y", "-i", src, "-vf", vf, "-an", "-c:v", "libx264", "-crf", "23",
                               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst])
        print(f"{ep}: {d['n']} rows, video from {'the overlay render' if rendered else 'the head clip'}, "
              + ", ".join(f"{k} {v['coverage']} %" for k, v in G.items()))
    have = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(out, "ep*.json")))
    json.dump(dict(collection=os.path.basename(col), episodes=have), open(os.path.join(out, "index.json"), "w"))
    shutil.copy(os.path.join(ROOT, "viewer", "index.html"), os.path.join(out, "index.html"))
    print(f"viewer -> {out}")


if __name__ == "__main__":
    main()
