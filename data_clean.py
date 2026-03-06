"""
Dataset cleaning script

Reads CSVs from a directory (or a single CSV), filters to C/C++ submissions,
removes duplicate submissions by the same author for the same task, and
writes a cleaned CSV for downstream training.

Usage:
    python data_clean.py --input datasets/ --output datasets/cleaned.csv

"""
import argparse
import glob
import os
import pandas as pd


def is_cpp_file_name(fname: str) -> bool:
    return str(fname).lower().endswith(('.cpp', '.cc', '.cxx', '.c++', '.c'))


def filter_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows that look like C/C++ submissions
    if 'file' in df.columns:
        file_lower = df['file'].fillna('').astype(str).str.lower()
        df = df[file_lower.str.endswith(('.cpp', '.cc', '.cxx', '.c++', '.c'))].copy()
    else:
        # fallback heuristic
        df = df[df['flines'].str.contains(r"#include|using\\s+namespace|std::", regex=True)].copy()

    # Deduplicate by (username, task) if task exists
    if 'task' in df.columns:
        df = df.drop_duplicates(subset=['username', 'task'], keep='first').reset_index(drop=True)

    return df


def read_all_csvs(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.csv')))
        if not files:
            raise FileNotFoundError(f'No CSV files found in {path}')
        parts = []
        for f in files:
            print('Reading', f)
            parts.append(pd.read_csv(f))
        return pd.concat(parts, ignore_index=True)
    else:
        return pd.read_csv(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input CSV file or directory')
    p.add_argument('--output', '-o', required=True, help='Output cleaned CSV path')
    args = p.parse_args()

    df = read_all_csvs(args.input)
    print('Loaded rows:', len(df))

    df = df.dropna(subset=['flines', 'username'])
    df['flines'] = df['flines'].astype(str)

    cleaned = filter_and_dedup(df)
    print('After filtering/dedup:', len(cleaned))

    cleaned.to_csv(args.output, index=False)
    print('Saved cleaned dataset to', args.output)


if __name__ == '__main__':
    main()
