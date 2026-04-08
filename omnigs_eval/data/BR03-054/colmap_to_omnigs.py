#!/usr/bin/env python3
"""
Convert COLMAP sparse model + equirectangular video to OmniGS openMVG format.

Takes COLMAP perspective camera poses and converts them to equirectangular
camera poses in openMVG's sfm_data.json format for OmniGS training.

Usage:
    python colmap_to_omnigs.py \
        --video data/BR03-054/video.mp4 \
        --colmap data/BR03-054/colmap_sparse \
        --output data/BR03-054/omnigs \
        --every 10
"""

import argparse
import cv2
import json
import numpy as np
import os
import struct
from pathlib import Path
from scipy.spatial.transform import Rotation


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            params = struct.unpack("<4d", f.read(32))
            cameras[cam_id] = {"w": w, "h": h, "params": params}
    return cameras


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)
            images[name.decode()] = {
                "qvec": [qw, qx, qy, qz],
                "tvec": [tx, ty, tz],
                "camera_id": cam_id,
            }
    return images


def read_points3d_bin(path):
    """Read COLMAP points3D.bin and return list of (xyz, rgb) tuples."""
    points = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pid = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            points.append({"xyz": [x, y, z], "rgb": [r, g, b]})
    return points


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP to OmniGS format")
    parser.add_argument("--video", required=True, help="Path to equirectangular MP4")
    parser.add_argument("--colmap", required=True, help="Path to COLMAP sparse dir (with cameras.bin, images.bin, points3D.bin)")
    parser.add_argument("--output", required=True, help="Output directory for OmniGS data")
    parser.add_argument("--every", type=int, default=10, help="Frame extraction interval (must match COLMAP extraction)")
    parser.add_argument("--rotate180", action="store_true", help="Apply 180° rotation (for upside-down cameras)")
    args = parser.parse_args()

    colmap_dir = Path(args.colmap)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Read COLMAP model
    print("Reading COLMAP model...")
    cameras = read_cameras_bin(colmap_dir / "cameras.bin")
    colmap_images = read_images_bin(colmap_dir / "images.bin")
    print(f"  Cameras: {len(cameras)}, Images: {len(colmap_images)}")

    # Group perspective views by frame number
    # Use the "front" view (yaw=0, pitch=0) as the equirectangular frame orientation
    frame_poses = {}  # frame_num -> (R_w2c, center)
    for name, data in colmap_images.items():
        if "_front." not in name:
            continue
        frame_num = int(name.split("_")[0].replace("frame", ""))
        qw, qx, qy, qz = data["qvec"]
        tx, ty, tz = data["tvec"]

        R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz])
        center = -R_w2c.T @ t  # camera position in world

        # The "front" view's R_w2c IS the equirectangular frame's orientation
        # (since front has yaw=0, pitch=0 in the extraction)
        frame_poses[frame_num] = {
            "R_w2c": R_w2c,
            "center": center,
        }

    print(f"  Frames with poses: {len(frame_poses)}")

    # Extract equirectangular frames from video
    print(f"Extracting equirectangular frames from {args.video}...")
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {vid_w}x{vid_h}, {total_frames} frames")

    extracted = []
    for fi in sorted(frame_poses.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {fi}")
            continue

        if args.rotate180:
            frame = cv2.flip(frame, -1)

        fname = f"frame{fi:06d}.jpg"
        cv2.imwrite(str(images_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted.append((fi, fname))

    cap.release()
    print(f"  Extracted {len(extracted)} equirectangular frames")

    # Build openMVG sfm_data.json
    print("Building openMVG sfm_data.json...")

    views = []
    extrinsics = []
    for idx, (fi, fname) in enumerate(extracted):
        pose = frame_poses[fi]

        views.append({
            "key": idx,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483649 + idx,
                    "data": {
                        "local_path": "",
                        "filename": fname,
                        "width": vid_w,
                        "height": vid_h,
                        "id_view": idx,
                        "id_intrinsic": 0,
                        "id_pose": idx,
                    },
                },
            },
        })

        R = pose["R_w2c"]
        c = pose["center"]

        extrinsics.append({
            "key": idx,
            "value": {
                "rotation": R.tolist(),
                "center": c.tolist(),
            },
        })

    sfm_data = {
        "sfm_data_version": "0.3",
        "root_path": str(images_dir.resolve()),
        "views": views,
        "intrinsics": [
            {
                "key": 0,
                "value": {
                    "polymorphic_id": 2147483649,
                    "polymorphic_name": "spherical",
                    "ptr_wrapper": {
                        "id": 2147483660,
                        "data": {
                            "value0": {
                                "width": vid_w,
                                "height": vid_h,
                            }
                        },
                    },
                },
            }
        ],
        "extrinsics": extrinsics,
        "structure": [],
        "control_points": [],
    }

    sfm_path = output_dir / "data_openmvg.json"
    with open(sfm_path, "w") as f:
        json.dump(sfm_data, f, indent=2)
    print(f"  Wrote {sfm_path}")

    # Summary
    print(f"\nDone! OmniGS data prepared in {output_dir}/")
    print(f"  Images: {images_dir}/ ({len(extracted)} equirectangular frames, {vid_w}x{vid_h})")
    print(f"  SfM data: {sfm_path}")
    print(f"  Sparse PLY: copy your sparse.ply to {output_dir}/sparse.ply")
    print(f"\nNext steps:")
    print(f"  1. Build OmniGS (see CONTEXT.md)")
    print(f"  2. Run: ./bin/train_openmvg_lonlat cfg/lonlat/your_config.yaml {output_dir} /path/to/results")


if __name__ == "__main__":
    main()
