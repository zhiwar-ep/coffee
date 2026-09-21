#!/usr/bin/env python3
"""
- Copy Data_template.ods -> PP_data.ods
- Read probe files in postProcessing/probes/0 (skip header, downsample)
- Put data into matching sheet (sheet name == filename without extension)
- Keep template headers; write data starting from row 2 (1-based)
"""

import os
import re
import shutil

import ezodf

# ---- config ----
PROBES_DIR = os.path.join("postProcessing", "probes", "0")
TEMPLATE_ODS = "Data_template.ods"
OUT_ODS = "PP_data.ods"

SKIP_LINES = 3
EXPECTED_COLS = 4
ROW_STEP = 100  # keep 1 row every N rows

# Sheet name rules: max 31 chars, no : \ / ? * [ ]
_illegal = re.compile(r"[:\\/?*\[\]]")

def safe_sheet_name(name: str) -> str:
    name = _illegal.sub("_", name)
    name = name.strip() or "Sheet"
    return name[:31]

def read_probe_file(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(SKIP_LINES):
            next(f, None)

        for i, line in enumerate(f):
            if ROW_STEP > 1 and (i % ROW_STEP != 0):
                continue

            s = line.strip()
            if not s:
                continue

            parts = s.split()
            if len(parts) < EXPECTED_COLS:
                continue

            rows.append(parts[:EXPECTED_COLS])
    return rows

def to_number_if_possible(x: str):
    try:
        return float(x)
    except ValueError:
        return x

def ensure_size(sheet, min_rows: int, min_cols: int):
    # ezodf sheets have fixed size; expand if needed
    add_r = max(0, min_rows - sheet.nrows())
    add_c = max(0, min_cols - sheet.ncols())
    if add_r:
        sheet.append_rows(add_r)
    if add_c:
        sheet.append_columns(add_c)

def clear_block(sheet, start_row0: int, nrows: int, ncols: int):
    # start_row0 is 0-based
    for r in range(start_row0, start_row0 + nrows):
        for c in range(0, ncols):
            sheet[r, c].set_value("")

def main():
    if not os.path.isfile(TEMPLATE_ODS):
        raise FileNotFoundError(f"Template not found: {os.path.abspath(TEMPLATE_ODS)}")
    if not os.path.isdir(PROBES_DIR):
        raise FileNotFoundError(f"Probes dir not found: {os.path.abspath(PROBES_DIR)}")

    # copy template -> output
    shutil.copy2(TEMPLATE_ODS, OUT_ODS)

    # open output doc for editing
    doc = ezodf.opendoc(OUT_ODS)

    # map sheet names for fast lookup
    sheets_by_name = {sh.name: sh for sh in doc.sheets}

    files = [
        os.path.join(PROBES_DIR, fn)
        for fn in sorted(os.listdir(PROBES_DIR))
        if os.path.isfile(os.path.join(PROBES_DIR, fn))
    ]
    if not files:
        raise RuntimeError(f"No probe files found in: {os.path.abspath(PROBES_DIR)}")

    missing_sheets = []
    filled = 0

    for filepath in files:
        fname = os.path.basename(filepath)
        target_name = safe_sheet_name(os.path.splitext(fname)[0])

        sheet = sheets_by_name.get(target_name)
        if sheet is None:
            missing_sheets.append(target_name)
            continue

        data = read_probe_file(filepath)
        n = len(data)

        # We write starting at row 2 (1-based) => index 1 (0-based)
        start_row0 = 1
        needed_rows = start_row0 + max(n, 1)
        needed_cols = EXPECTED_COLS

        ensure_size(sheet, needed_rows, needed_cols)

        # Clear existing old data area (from row 2 down, first 4 cols)
        # Clear down to current sheet size (from start_row0 to end)
        clear_block(sheet, start_row0, sheet.nrows() - start_row0, needed_cols)

        # Write new data
        for i, row in enumerate(data):
            r = start_row0 + i
            for c in range(EXPECTED_COLS):
                sheet[r, c].set_value(to_number_if_possible(row[c]))

        filled += 1

    doc.save()

    print(f"Saved: {os.path.abspath(OUT_ODS)}")
    print(f"Filled sheets: {filled}")
    if missing_sheets:
        print("WARNING: these sheets were not found in template (skipped):")
        for s in missing_sheets:
            print(" -", s)

if __name__ == "__main__":
    main()
