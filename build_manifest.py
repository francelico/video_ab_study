#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict
import argparse


def build_manifest(videos_dir: Path) -> dict:
    """
    Build manifest structure:
    {
      set_name: {
        method_name: [list of relative video paths]
      }
    }
    """
    manifest = defaultdict(lambda: defaultdict(list))

    videos_dir = videos_dir.resolve()
    root_name = videos_dir.name  # usually "videos"

    for video_path in sorted(videos_dir.rglob("*.mp4")):
        # Expect: videos/<set>/<method>/<video>.mp4
        try:
            rel = video_path.relative_to(videos_dir)
            set_name = rel.parts[0]
            method_name = rel.parts[1]
        except Exception:
            raise RuntimeError(
                f"Invalid directory structure for video: {video_path}\n"
                "Expected: videos/<set>/<method>/<video>.mp4"
            )

        rel_path = str(Path(root_name) / rel)
        manifest[set_name][method_name].append(rel_path)

    # sort for determinism
    manifest_sorted = {
        set_name: {
            method: sorted(videos)
            for method, videos in sorted(methods.items())
        }
        for set_name, methods in sorted(manifest.items())
    }

    return manifest_sorted


def main():
    parser = argparse.ArgumentParser(description="Generate manifest.json from videos directory")
    parser.add_argument(
        "videos_dir",
        type=Path,
        help="Path to videos directory (e.g. ./videos)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("manifest.json"),
        help="Output manifest file (default: manifest.json)"
    )

    args = parser.parse_args()

    manifest = build_manifest(args.videos_dir)

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Wrote manifest to {args.out}")


if __name__ == "__main__":
    main()
