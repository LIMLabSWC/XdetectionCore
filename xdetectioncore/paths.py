import re
from pathlib import Path, PureWindowsPath, PurePosixPath
from datetime import datetime

def posix_from_win(path: str, ceph_linux_dir='/ceph/akrami') -> Path:
    if ':\\' in path:
        path_bits = PureWindowsPath(path).parts
        path_bits = [bit for bit in path_bits if '\\' not in bit]
        return Path(PurePosixPath(*path_bits))
    return Path(path).relative_to(ceph_linux_dir) if ceph_linux_dir in path else Path(path)

def get_spikedir(recdir, sorter='kilosort2_5', sorting_dir_name='sorting') -> Path:
    recdir = Path(recdir)
    while not any([sorting_dir_name in str(p) for p in recdir.iterdir()]):
        recdir = recdir.parent
    spikedir = recdir / sorting_dir_name / sorter / 'sorter_output'
    return spikedir

def extract_date(filename_str: str, out_fmt: str = '%y%m%d'):
    try:
        date = re.search(r'\d{4}-\d{2}-\d{2}', filename_str).group(0)
        date = datetime.strptime(date, '%Y-%m-%d').strftime('%y%m%d')
    except AttributeError:
        date = re.search(r'\d{6}', filename_str).group(0)
    return datetime.strptime(date, '%y%m%d').strftime(out_fmt)