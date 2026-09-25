#!/usr/bin/env python3

import os
import re
import gzip
import math
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

CASE_DIR = "."
FIELD_NAME = "UsDEM"


# ============================================================
# REGEX
# ============================================================

vector_pattern = re.compile(
    r"\(\s*"
    r"([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)"
    r"\s*\)"
)

uniform_pattern = re.compile(
    r"internalField\s+uniform\s+"
    r"\(\s*"
    r"([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)"
    r"\s*\)\s*;"
)


# ============================================================
# OPEN NORMAL OR GZIPPED FILE
# ============================================================

def open_field(path):

    if str(path).endswith(".gz"):
        return gzip.open(
            path,
            "rt",
            errors="ignore"
        )

    return open(
        path,
        "r",
        errors="ignore"
    )


# ============================================================
# READ MAXIMUM INTERNAL UsDEM
# ============================================================

def get_max_internal_velocity(path):

    with open_field(path) as f:
        text = f.read()

    # --------------------------------------------------------
    # UNIFORM internalField
    # --------------------------------------------------------

    uniform_match = uniform_pattern.search(text)

    if uniform_match:

        ux = float(uniform_match.group(1))
        uy = float(uniform_match.group(2))
        uz = float(uniform_match.group(3))

        mag = math.sqrt(
            ux * ux
            + uy * uy
            + uz * uz
        )

        return mag, (ux, uy, uz), None


    # --------------------------------------------------------
    # NONUNIFORM internalField
    # --------------------------------------------------------

    start = text.find(
        "internalField"
    )

    if start == -1:
        return None

    nonuniform = text.find(
        "nonuniform",
        start
    )

    if nonuniform == -1:
        return None

    list_start = text.find(
        "(",
        nonuniform
    )

    if list_start == -1:
        return None


    # Boundary field marks the end of internalField
    boundary_start = text.find(
        "boundaryField",
        list_start
    )

    if boundary_start == -1:
        internal_text = text[list_start:]
    else:
        internal_text = text[
            list_start:boundary_start
        ]


    max_mag = -1.0
    max_vector = None
    max_index = None

    cell_index = 0


    for match in vector_pattern.finditer(
        internal_text
    ):

        ux = float(match.group(1))
        uy = float(match.group(2))
        uz = float(match.group(3))

        mag2 = (
            ux * ux
            + uy * uy
            + uz * uz
        )

        if mag2 > max_mag:

            max_mag = mag2

            max_vector = (
                ux,
                uy,
                uz
            )

            max_index = cell_index

        cell_index += 1


    if max_vector is None:
        return None


    return (
        math.sqrt(max_mag),
        max_vector,
        max_index
    )


# ============================================================
# MAIN
# ============================================================

case_path = Path(
    CASE_DIR
)

processors = sorted(
    [
        p for p in case_path.glob("processor*")
        if p.is_dir()
    ],
    key=lambda p: int(
        re.search(
            r"\d+",
            p.name
        ).group()
    )
)


if not processors:

    raise RuntimeError(
        "No processor* directories found."
    )


global_max = -1.0
global_vector = None
global_time = None
global_processor = None
global_cell = None
global_file = None

files_checked = 0


# ============================================================
# LOOP OVER PROCESSORS AND TIME DIRECTORIES
# ============================================================

for processor in processors:

    for time_dir in processor.iterdir():

        if not time_dir.is_dir():
            continue

        # Only numeric OpenFOAM time directories
        try:
            time_value = float(
                time_dir.name
            )
        except ValueError:
            continue


        field_file = (
            time_dir
            / FIELD_NAME
        )

        field_file_gz = (
            time_dir
            / f"{FIELD_NAME}.gz"
        )


        if field_file.exists():
            path = field_file

        elif field_file_gz.exists():
            path = field_file_gz

        else:
            continue


        result = get_max_internal_velocity(
            path
        )

        if result is None:
            continue


        files_checked += 1

        magnitude, vector, cell_index = result


        if magnitude > global_max:

            global_max = magnitude
            global_vector = vector
            global_time = time_value
            global_processor = processor.name
            global_cell = cell_index
            global_file = path

# ============================================================
# RESULT
# ============================================================

if global_max < 0:
    raise RuntimeError(
        "No valid UsDEM data found."
    )


result_text = f"""
==============================================
MAXIMUM UsDEM
==============================================
Maximum |UsDEM| = {global_max:.12e} m/s
Vector          = ({global_vector[0]:.12e} {global_vector[1]:.12e} {global_vector[2]:.12e})
Time step       = {global_time:g}
Processor       = {global_processor}
"""

if global_cell is not None:
    result_text += f"Local cell index= {global_cell}\n"
else:
    result_text += "Field           = uniform\n"

result_text += f"""File            = {global_file}
Files checked   = {files_checked}
==============================================
"""


# Print to terminal
print(result_text)


# Save to file
with open("maxUsDEM.txt", "w") as f:
    f.write(result_text)


print("Saved to: maxUsDEM.txt")