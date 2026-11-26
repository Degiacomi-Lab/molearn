"""Create and optionally render a ChimeraX movie of a SimRNA latent path."""

from __future__ import annotations

import argparse
import colorsys
import re
import shutil
import subprocess
from pathlib import Path


CAMERA_MATRIX_RNA = (
    "0.56277,-0.20984,0.79953,376.63,"
    "-0.7439,-0.55029,0.37919,160.22,"
    "0.36041,-0.80818,-0.46579,-245.28"
)


def frame_number_RNA(path_RNA):
    match_RNA = re.fullmatch(r"s_(\d+)", path_RNA.stem)
    if match_RNA is None:
        raise ValueError(f"Unexpected transition-frame name: {path_RNA.name}")
    return int(match_RNA.group(1))


def rainbow_color_RNA(index_RNA, count_RNA):
    fraction_RNA = index_RNA / max(count_RNA - 1, 1)
    rgb_RNA = colorsys.hsv_to_rgb(0.78 * fraction_RNA, 0.90, 0.95)
    return "#" + "".join(f"{round(value_RNA * 255):02x}" for value_RNA in rgb_RNA)


def quoted_RNA(path_RNA):
    return '"' + str(path_RNA).replace('"', '\\"') + '"'


def write_chimerax_script_RNA(
    path_directory_RNA,
    movie_path_RNA,
    script_path_RNA,
    frames_per_state_RNA,
    endpoint_hold_RNA,
    width_RNA,
    height_RNA,
):
    pdb_files_RNA = sorted(
        path_directory_RNA.glob("s_*.pdb"), key=frame_number_RNA
    )
    if len(pdb_files_RNA) < 2:
        raise FileNotFoundError(
            f"Need at least two s_*.pdb files in {path_directory_RNA}"
        )

    transition_models_RNA = list(range(3, 3 + len(pdb_files_RNA)))
    commands_RNA = [
        "close all",
        "set bgColor white",
        "lighting soft",
        "set silhouettes true",
        f"open {quoted_RNA(pdb_files_RNA[0].resolve())}",
        f"open {quoted_RNA(pdb_files_RNA[-1].resolve())}",
    ]
    commands_RNA.extend(
        f"open {quoted_RNA(pdb_file_RNA.resolve())}"
        for pdb_file_RNA in pdb_files_RNA
    )
    final_model_RNA = transition_models_RNA[-1]
    commands_RNA.extend(
        [
            f"style #1-{final_model_RNA} stick",
            f"size #1-{final_model_RNA} stickRadius 0.35",
            "color #1 deepskyblue",
            "color #2 magenta",
            "transparency #1 80 target abpfcr",
            "transparency #2 80 target abpfcr",
            f"hide #3-{final_model_RNA} models",
            f"windowsize {width_RNA} {height_RNA}",
            f"view matrix camera {CAMERA_MATRIX_RNA}",
            f"movie record size {width_RNA},{height_RNA} supersample 3",
            f"wait {endpoint_hold_RNA}",
        ]
    )
    previous_RNA = None
    for index_RNA, model_RNA in enumerate(transition_models_RNA):
        if previous_RNA is not None:
            commands_RNA.append(f"hide #{previous_RNA} models")
        commands_RNA.extend(
            [
                f"color #{model_RNA} {rainbow_color_RNA(index_RNA, len(transition_models_RNA))}",
                f"show #{model_RNA} models",
                f"wait {frames_per_state_RNA}",
            ]
        )
        previous_RNA = model_RNA
    commands_RNA.extend(
        [
            f"hide #{previous_RNA} models",
            f"wait {endpoint_hold_RNA}",
            f"movie encode output {quoted_RNA(movie_path_RNA.resolve())} "
            "format h264 framerate 24 quality highest",
            "exit",
        ]
    )
    script_path_RNA.parent.mkdir(parents=True, exist_ok=True)
    script_path_RNA.write_text("\n".join(commands_RNA) + "\n")
    return len(pdb_files_RNA)


def parse_args_RNA():
    analysis_RNA = (
        Path(__file__).resolve().parent
        / "analysis_SimRNA_RNA_apo_holo_apo_to_holo"
    )
    parser_RNA = argparse.ArgumentParser(description=__doc__)
    parser_RNA.add_argument(
        "--path-directory", type=Path, default=analysis_RNA / "apo_to_holo_path"
    )
    parser_RNA.add_argument(
        "--output", type=Path, default=analysis_RNA / "SimRNA_apo_to_holo_HD.mp4"
    )
    parser_RNA.add_argument(
        "--script", type=Path, default=analysis_RNA / "SimRNA_apo_to_holo_movie.cxc"
    )
    parser_RNA.add_argument("--frames-per-state", type=int, default=4)
    parser_RNA.add_argument("--endpoint-hold", type=int, default=24)
    parser_RNA.add_argument("--width", type=int, default=1920)
    parser_RNA.add_argument("--height", type=int, default=1080)
    parser_RNA.add_argument("--render", action="store_true")
    return parser_RNA.parse_args()


def main_RNA():
    args_RNA = parse_args_RNA()
    if args_RNA.frames_per_state < 1 or args_RNA.endpoint_hold < 0:
        raise ValueError("Movie frame counts are invalid")
    if args_RNA.width < 1 or args_RNA.height < 1:
        raise ValueError("Movie width and height must be positive")
    count_RNA = write_chimerax_script_RNA(
        args_RNA.path_directory,
        args_RNA.output,
        args_RNA.script,
        args_RNA.frames_per_state,
        args_RNA.endpoint_hold,
        args_RNA.width,
        args_RNA.height,
    )
    print(f"ChimeraX script: {args_RNA.script.resolve()}")
    print(f"Transition structures: {count_RNA}")
    print(f"Movie output: {args_RNA.output.resolve()}")
    if args_RNA.render:
        executable_RNA = shutil.which("chimerax")
        if executable_RNA is None:
            raise FileNotFoundError("chimerax is not available on PATH")
        subprocess.run([executable_RNA, str(args_RNA.script.resolve())], check=True)


if __name__ == "__main__":
    main_RNA()
