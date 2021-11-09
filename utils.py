import numpy as np
import glob
import errno
import os

def subset_data(file_directory, n_samples_per_class, filters=None):
    # Get all filepaths from initial data.
    filepaths = _get_filepaths(file_directory, filters=filters)

    data = []
    for key_id in KEYS:
        class_filepaths = []
        for filepath in filepaths:
            # Obtain a list of class filepaths related to key_id.
            if _get_key_id(filepath) == key_id:
                class_filepaths.append(filepath)
        # Get randomly chosen n_samples_per_class.
        subset = np.random.choice(class_filepaths, size=n_samples_per_class,
                replace=False)
        data.append(subset)
    return flatten_list(data, inc_str=True)

def flatten_list(lst, inc_str=False):
    if not inc_str:
        wrap = lambda x: x if isinstance(x, list) else [x]
        lst = [wrap(x) for x in lst]
    return [item for sublist in lst for item in sublist]

def _get_key_id(filepath):
    _, filename = os.path.split(filepath)
    f, _ = os.path.splitext(filename)
    s = ''.join([i for i in f if not i.isdigit()])
    # Special case of numerical key ID.
    if s == '':
        s = f[0]
    return s


def _get_filepaths(file_directory, filename='*.wav', filters=None):
    if not os.path.exists(file_directory):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                file_directory)
    if not os.path.isdir(file_directory):
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR),
                file_directory)
    filepaths = glob.iglob(os.path.join(file_directory, '**', filename),
            recursive=True)
    if filters != None:
        filepaths = [_filter(filepath, filters) for filepath in filepaths]
    else:
        filepaths = [filepath for filepath in filepaths]
    return [filepath for filepath in filepaths if filepath is not None]

def _filter(f, filters):
    s = os.path.split(f)[1]
    special_chars = ['Key.backspace', 'Key.esc', 'Key.space', 'Key.enter']
    for fil in filters:
        if fil in special_chars:
            if s.startswith('Key.'):
                return None
        if s.startswith(fil):
            return None
    else:
        return f

def _get_duplicates(lst, indices=True):
    """
    Inspired from [1].

    [1] : John La Rooy stack overflow answer at
        https://stackoverflow.com/questions/11236006/identify-duplicate-values-in-a-list-in-python
    """
    if indices:
        from collections import defaultdict
        D = defaultdict(list)
        for i, item in enumerate(lst):
            D[item].append(i)
        return {k : v for k, v in D.items() if len(v) > 1}

    from collections import Counter
    return [k for k, v in Counter(lst).items() if v > 1]

def _get_ref_filepath(filepath, to_string=True):
    filepath = _list_to_str(filepath)
    path, filename = os.path.split(filepath)
    file_directory, key  = os.path.split(path)
    parent_directory, _ = os.path.split(file_directory)
    ref_filepath = [f for f in _get_filepaths(parent_directory,
        filename) if f != os.path.join(path, filename)]
    return _list_to_str(ref_filepath) if to_string else ref_filepath

def _list_to_str(lst):
    return str(lst).replace("['", "").replace("']", "")

def splitall(path):
    """
    From
    https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def set_size(width, fraction=1):
    """
    Set figure dimensions to avoid scaling in LaTeX. Taken from
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
