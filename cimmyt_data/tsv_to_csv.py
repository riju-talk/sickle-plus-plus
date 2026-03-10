import csv
from pathlib import Path

def tsv_to_csv(src_path: str, dst_path: str = None):
    src = Path(src_path)
    if dst_path is None:
        dst = src.with_name(src.stem + '_fixed.csv')
    else:
        dst = Path(dst_path)

    with src.open('r', encoding='utf-8') as f_in:
        # read as tab-separated values
        reader = csv.reader(f_in, delimiter='\t')
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    with dst.open('w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)

    return dst

if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'cimmyt_data/variables_details.csv'
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    out = tsv_to_csv(src, dst)
    print(f'Wrote: {out}')
