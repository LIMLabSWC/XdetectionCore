import numpy as np
import pandas as pd
import re
import yaml
import platform
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm

from xdetectioncore.paths import posix_from_win


def extract_suffix(filename: str) -> int:
    """Extracts the _XXX suffix as an integer. Returns None if not found."""
    match = re.search(r'_(\d{3})(?=\.\w+$|$)', filename)
    return int(match.group(1)) if match else None


def validate_sequence(suffixes: list[int], animal: str, date: str):
    """Checks if the 3-digit suffixes follow a continuous sequence starting at 0."""
    if not suffixes:
        return

    suffixes = sorted(suffixes)
    expected = list(range(min(suffixes), max(suffixes) + 1))

    if suffixes != expected:
        missing = set(expected) - set(suffixes)
        print(f"![Sequence Gap] {animal} on {date}: Missing suffixes {missing}")

    if min(suffixes) != 0:
        print(f"![Sequence Note] {animal} on {date}: Sequence starts at {min(suffixes):03d} instead of 000")


def get_animal_topology(animal, tdata_root, match_roots):
    # 1. Get Core Trial Data
    all_tdata = sorted(list((tdata_root / animal / 'TrialData').glob('*.csv')))

    # Group by date to validate sequences per day
    date_groups = {}
    assert all([len(list(set(fps))) == len(fps) for fps in date_groups.values()])
    for f in all_tdata:
        suffix = extract_suffix(f.name)
        if suffix is None: continue  # Ignore files without _00X

        date_match = re.search(r'\d{6}', f.name)
        if not date_match: continue

        date = date_match.group()
        date_groups.setdefault(date, []).append((suffix, f))

    topology_rows = []
    assert len(date_groups) == len(set(date_groups.keys()))
    for date, files in date_groups.items():
        # Validate the sequence for this day
        if len(files)>1:
            pass
        suffixes = [f[0] for f in files]
        validate_sequence(suffixes, animal, date)

        for suffix, t_file in files:
            suffix_str = f"{suffix:03d}"
            row = {
                'name': animal,
                'date': date,
                'suffix': suffix_str,
                'tdata_file': t_file,
                'status': 'Complete'
            }

            # 2. Match other types by Date AND Suffix
            for label, root in match_roots.items():
                if 'bin' in label:
                    data_type = 'Hit' if 'beh' in label else 'Sound'
                    pattern = f"{animal}*{data_type}Data*{date}*_{suffix_str}.bin"
                else:
                    pattern = f"*{animal}*{date}*_{suffix_str}*"

                matches = [c for c in root.glob(pattern) if extract_suffix(c.name) == suffix]
                if len(matches) > 1:
                    print(f"![Duplicate File] {animal} on {date}: {label} {suffix_str}: {matches}")

                if len(matches) == 1:
                    row[label] = matches[0]
                else:
                    row[label] = None
                    row['status'] = 'Partial'  # Flag if any file is missing

            topology_rows.append(row)

    return topology_rows


def add_stage_info(session_topology):

    for idx, row in session_topology.iterrows():
        td_path = row['tdata_file']
        if td_path is None:
            session_topology.loc[idx, 'Stage'] = None
            continue

        td_df = pd.read_csv(td_path)
        if 'Stage' in td_df.columns:
            session_topology.loc[idx, 'Stage'] = td_df['Stage'].iloc[0]
        else:
            session_topology.loc[idx, 'Stage'] = None
    session_topology['Stage'] = session_topology['Stage'].astype(int, errors='ignore')

def add_sess_order_info(session_topology):
    for idx, row in session_topology.iterrows():
        td_path = row['tdata_file']
        if td_path is None:
            session_topology.loc[idx, 'sess_order'] = None
        else:
            session_topology.loc[idx, 'sess_order'] = 'main'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('animals')
    parser.add_argument('--sess_top_suffix',default='')

    with open(parser.parse_args().config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    assert ceph_dir.is_dir()

    args = parser.parse_args()
    home_dir = Path(config[f'home_dir_{sys_os}'])
    tdata_dir = home_dir/'data'
    assert tdata_dir.is_dir()

    exp_name_dict = {
        'DO': 'Dammy',
        'RS': 'Ryan',
        'LP': 'Lida',
        'JK': 'JungWoo'
    }

    match_roots = {'videos_dir': ceph_dir / posix_from_win(r'X:\Dammy\mouse_pupillometry\mouse_hf'),
                   'beh_bin': ceph_dir / posix_from_win(r'X:\Dammy\harpbins'),
                   'sound_bin': ceph_dir / posix_from_win(r'X:\Dammy\harpbins')}

    all_data = []
    animals = args.animals.split(',')
    assert len(animals) == len(set(animals)), 'Animal names must be unique'
    for animal in tqdm(animals,total=len(animals),desc='Processing animals'):
        exp_name = exp_name_dict.get(animal[:2])

        all_data.extend(get_animal_topology(animal, tdata_dir/exp_name, match_roots=match_roots))

    projectdir = ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test')
    csv_path = projectdir / f'session_topology{f"_{args.sess_top_suffix}" if args.sess_top_suffix else ""}.csv'

    df = pd.DataFrame(all_data)
    add_stage_info(df)
    add_sess_order_info(df)
    df.to_csv(csv_path, index=False)