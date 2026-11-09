"""
Script mandiri untuk mengunduh atau memperbarui dataset dari Roboflow.
"""

import argparse
import os
from pathlib import Path
from roboflow import Roboflow


def download_dataset(
    api_key: str = os.environ.get("ROBOFLOW_API_KEY"),
    workspace: str = "weapon-mpr3p",
    project_name: str = "ta-rhzr5",
    version_num: int = 1,
    export_format: str = "yolov12",
):
    print("=" * 60)
    print("             ROBOFLOW DATASET DOWNLOADER             ")
    print("=" * 60)
    print(f"Workspace : {workspace}")
    print(f"Project   : {project_name}")
    print(f"Version   : {version_num}")
    print(f"Format    : {export_format}")
    print("=" * 60)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)

    dataset = version.download(export_format)
    print(f"\n[SUKSES] Dataset berhasil diunduh ke: {Path(dataset.location).resolve()}")
    return dataset.location


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Roboflow Dataset")
    parser.add_argument("--api-key", type=str, default=os.environ.get("ROBOFLOW_API_KEY"), help="Default: variabel lingkungan ROBOFLOW_API_KEY")
    parser.add_argument("--workspace", type=str, default="weapon-mpr3p")
    parser.add_argument("--project", type=str, default="ta-rhzr5")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--format", type=str, default="yolov12")
    args = parser.parse_args()

    download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project_name=args.project,
        version_num=args.version,
        export_format=args.format,
    )
