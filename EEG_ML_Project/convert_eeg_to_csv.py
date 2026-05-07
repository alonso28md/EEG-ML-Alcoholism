# Convert UCI EEG Alcoholism Database from .rd.xxx.gz format to CSV.
# Source: eeg+database/eeg_full/  (one tar.gz per subject)
# Inside each tar: name/name.rd.NNN.gz  (one trial per file, 64 ch x 256 samples)
# Output: Data1.csv, Data2.csv, ... in data/raw/

import os, sys, tarfile, gzip, io, csv
sys.stdout.reconfigure(encoding='utf-8')

SRC_DIR  = r'C:\Users\alons\Downloads\eeg+database\eeg_full'
OUT_DIR  = r'C:\Users\alons\Downloads\EEG_ML_Project\EEG_ML_Project\data\raw'
MSECS_DEFAULT = 3.906

COLUMNS = ['trial number', 'sensor position', 'sample num', 'sensor value',
           'subject identifier', 'matching condition', 'channel', 'name', 'time']


def parse_rd_gz(fileobj, subject_name):
    """Parse one decompressed .rd.gz file into a list of row dicts."""
    subj_id = 'a' if 'co2a' in subject_name else 'c'
    msecs = MSECS_DEFAULT
    matching_condition = ''
    current_channel_num = None
    rows = []

    for raw_line in fileobj:
        line = raw_line.decode('utf-8', errors='replace').strip()
        if not line:
            continue
        if line.startswith('#'):
            content = line[1:].strip()
            if 'msecs' in content:
                try:
                    msecs = float(content.split()[0])
                except ValueError:
                    pass
            elif ', trial ' in content:
                # "S1 obj , trial 10"  or  "S2 mat , trial 5"
                parts = content.split(', trial ')
                matching_condition = parts[0].strip()
            elif ' chan ' in content:
                # "FP1 chan 0"
                try:
                    current_channel_num = int(content.split(' chan ')[1].strip())
                except (IndexError, ValueError):
                    current_channel_num = None
        else:
            parts = line.split()
            if len(parts) == 4:
                try:
                    trial_num  = int(parts[0])
                    sensor_pos = parts[1]
                    sample_num = int(parts[2])
                    sensor_val = float(parts[3])
                    time_s     = sample_num * msecs / 1000.0
                    rows.append((trial_num, sensor_pos, sample_num, sensor_val,
                                 subj_id, matching_condition,
                                 current_channel_num, subject_name, time_s))
                except ValueError:
                    pass
    return rows


os.makedirs(OUT_DIR, exist_ok=True)

# Remove old CSVs
old = [f for f in os.listdir(OUT_DIR) if f.endswith('.csv')]
print(f"Removing {len(old)} old CSVs from data/raw/ ...")
for f in old:
    os.remove(os.path.join(OUT_DIR, f))

subject_tarballs = sorted(
    f for f in os.listdir(SRC_DIR) if f.endswith('.tar.gz')
)
print(f"Found {len(subject_tarballs)} subject archives\n")

csv_counter = 1
total_rows  = 0

for ti, tarball in enumerate(subject_tarballs):
    subject_name = tarball.replace('.tar.gz', '')
    tar_path = os.path.join(SRC_DIR, tarball)

    with tarfile.open(tar_path, 'r:gz') as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith('.gz') and '.rd.' in m.name]

        for member in members:
            raw_gz = tf.extractfile(member)
            if raw_gz is None:
                continue
            decompressed = gzip.decompress(raw_gz.read())
            rows = parse_rd_gz(io.BytesIO(decompressed), subject_name)

            if not rows:
                continue

            out_path = os.path.join(OUT_DIR, f'Data{csv_counter}.csv')
            with open(out_path, 'w', newline='', encoding='utf-8') as fout:
                writer = csv.writer(fout)
                writer.writerow(COLUMNS)
                writer.writerows(rows)

            total_rows  += len(rows)
            csv_counter += 1

    if (ti + 1) % 10 == 0 or ti == len(subject_tarballs) - 1:
        print(f"  [{ti+1}/{len(subject_tarballs)}] {subject_name}  "
              f"→ {csv_counter-1} CSVs so far, {total_rows:,} rows")

print(f"\nDone. Generated {csv_counter-1} CSV files, {total_rows:,} total rows.")
print(f"Saved to: {OUT_DIR}")
