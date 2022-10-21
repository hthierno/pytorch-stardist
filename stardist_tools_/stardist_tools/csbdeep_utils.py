# Copied from CSBDEEP source code: https://github.com/CSBDeep/CSBDeep/blob/ac7bbf3ebc2634a04c7f891625fb749c168b6296/csbdeep/utils/utils.py

import collections

import numpy as np

from pathlib import Path
Path().expanduser()


def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


def choice(population, k=1, replace=True):
    return list(np.random.choice(population, k, replace=replace))


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x


def download_and_extract_zip_file(url, targetdir='.', verbose=True):
    import os
    import csv
    from six.moves.urllib.request import urlretrieve
    from six.moves.urllib.parse import urlparse
    from zipfile import ZipFile

    res = urlparse(url)
    if res.scheme in ('','file'):
        url = Path(res.path).resolve().as_uri()
        # local file, 'urlretrieve' will not make a copy
        # -> don't delete 'downloaded' file
        delete = False
    else:
        delete = True

    # verbosity levels:
    # - 0: no messages
    # - 1: status messages
    # - 2: status messages and list of all files
    if isinstance(verbose,bool):
        verbose *= 2

    log = (print) if verbose else (lambda *a,**k: None)

    targetdir = Path(targetdir)
    if not targetdir.is_dir():
        targetdir.mkdir(parents=True,exist_ok=True)

    provided = []

    def content_is_missing():
        try:
            filepath, http_msg = urlretrieve(url+'.contents')
            with open(filepath,'r') as contents_file:
                contents = list(csv.reader(contents_file,delimiter='\t'))
        except:
            return True
        finally:
            if delete:
                try: os.unlink(filepath)
                except: pass

        for size, relpath in contents:
            size, relpath = int(size.strip()), relpath.strip()
            entry = targetdir / relpath
            if not entry.exists():
                return True
            if entry.is_dir():
                if not relpath.endswith('/'): return True
            elif entry.is_file():
                if relpath.endswith('/') or entry.stat().st_size != size: return True
            else:
                return True
            provided.append(relpath)

        return False


    if content_is_missing():
        try:
            log('Files missing, downloading...',end='')
            filepath, http_msg = urlretrieve(url)
            with ZipFile(filepath,'r') as zip_file:
                log(' extracting...',end='')
                zip_file.extractall(str(targetdir))
                provided = zip_file.namelist()
            log(' done.')
        finally:
            if delete:
                try: os.unlink(filepath)
                except: pass
    else:
        log('Files found, nothing to download.')

    if verbose > 1:
        log('\n'+str(targetdir)+':')
        consume(map(lambda x: log('-',Path(x)), provided))

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)