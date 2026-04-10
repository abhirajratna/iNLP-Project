import argparse
import os
import sys
import csv
CPP_EXTS = ('.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx', '.h', '.ipp', '.tpp')

def read_cleaned_columns(cleaned_csv_path):
    try:
        with open(cleaned_csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader)
            return header
    except Exception as e:
        raise RuntimeError(f'Failed to read header from {cleaned_csv_path}: {e}')

def read_parquet(path):
    try:
        import pandas as pd
        try:
            return pd.read_parquet(path, engine='pyarrow')
        except Exception:
            return pd.read_parquet(path)
    except Exception:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            return table.to_pandas()
        except Exception as e:
            raise RuntimeError(f'Failed to read parquet file: {e}')

def filter_cpp(df):
    try:
        import pandas as pd
    except Exception:
        raise RuntimeError('pandas is required for filtering; please install pandas')

    def detect_cpp_by_code(code):
        s = str(code).lower()
        return any((tok in s for tok in ('#include', 'std::', 'int main(', 'using namespace std', 'cout', 'cin', 'printf(')))
    mask = pd.Series(False, index=df.index)
    if 'file' in df.columns:
        files = df['file'].astype(str).str.lower()
        mask = files.str.endswith(CPP_EXTS)
    if 'language' in df.columns:
        lang_mask = df['language'].astype(str).str.contains('c\\\\+\\\\+|cpp', case=False, regex=True)
        mask = mask | lang_mask
    if 'code' in df.columns:
        code_mask = df['code'].apply(detect_cpp_by_code)
        mask = mask | code_mask
    return df[mask]

def map_to_cleaned_columns(df, columns):
    import pandas as pd
    out = pd.DataFrame(index=df.index)
    for c in columns:
        out[c] = ''
    if 'solution' in out.columns:
        out['solution'] = df['code'].astype(str) if 'code' in df.columns else ''
    if 'username' in out.columns:
        out['username'] = df['generator'].astype(str) if 'generator' in df.columns else ''
    if 'task' in out.columns:
        out['task'] = df['label'].astype(str) if 'label' in df.columns else ''
    if 'flines' in out.columns and 'code' in df.columns:
        out['flines'] = df['code'].astype(str).apply(lambda s: s.count('\n') + 1)
    return out.loc[:, columns]

def write_csv(df, out_path, append=False):
    header = not (append and os.path.exists(out_path))
    df.to_csv(out_path, mode='a' if append else 'w', index=False, header=header)

def main():
    p = argparse.ArgumentParser(description='Filter Parquet for C++ code and save CSV matching cleaned.csv columns')
    p.add_argument('parquet', help='Path to Parquet file')
    p.add_argument('--cleaned', default='datasets/cleaned.csv', help='Path to cleaned.csv (for columns)')
    p.add_argument('--out', default='datasets/ai_hum.csv', help='Output CSV path')
    p.add_argument('--append', action='store_true', help='Append to output if it exists')
    p.add_argument('--limit', type=int, default=0, help='Optional: limit number of rows written (0 = all)')
    args = p.parse_args()
    if not os.path.exists(args.parquet):
        print(f'Parquet file not found: {args.parquet}', file=sys.stderr)
        sys.exit(2)
    try:
        cols = read_cleaned_columns(args.cleaned)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    try:
        df = read_parquet(args.parquet)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    try:
        df_cpp = filter_cpp(df)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    if args.limit and args.limit > 0:
        df_cpp = df_cpp.head(args.limit)
    try:
        df_out = map_to_cleaned_columns(df_cpp, cols)
    except Exception as e:
        print(f'Failed to align columns: {e}', file=sys.stderr)
        sys.exit(2)
    try:
        import pandas as pd
        write_csv(df_out, args.out, append=args.append)
        print(f'Wrote {len(df_out)} rows to {args.out}')
    except Exception:
        try:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            write_header = not (args.append and os.path.exists(args.out))
            with open(args.out, 'a' if args.append else 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(cols)
                for _, row in df_out.iterrows():
                    writer.writerow([row.get(c, '') for c in cols])
            print(f'Wrote {len(df_out)} rows to {args.out}')
        except Exception as e:
            print(f'Failed to write CSV: {e}', file=sys.stderr)
            sys.exit(2)
if __name__ == '__main__':
    main()