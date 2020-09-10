from pathlib import Path

import numpy as np
from astropy.io import fits

def load_image(filename: str) -> np.ndarray:
    """
    TBW.

    Parameters
    ----------
    filename

    Returns
    -------

    """

    filename_path = Path(filename)

    if not filename_path.exists():
        raise FileNotFoundError(f"Input file '{filename_path}' can not be found.")

    # TODO: change to Path(path).suffix.lower().startswith('.fit')
    #       Same applies to `.npy`.
    if ".fits" in filename_path.suffix:
        data_2d = fits.getdata(filename_path)  # type: np.ndarray

    elif ".npy" in filename_path.suffix:
        data_2d = np.load(filename_path)

    elif ".txt" in filename_path.suffix:
        # TODO: this is a convoluted implementation. Change to:
        # for sep in [' ', ',', '|', ';']:
        #     try:
        #         data = np.loadtxt(path, delimiter=sep[ii])
        #     except ValueError:
        #         pass
        #     else:
        #         break
        sep = [" ", ",", "|", ";"]
        ii, jj = 0, 1
        while jj:
            try:
                jj -= 1
                data_2d = np.loadtxt(filename_path, delimiter=sep[ii])
            except ValueError:
                ii += 1
                jj += 1
                if ii >= len(sep):
                    break

    else:
        raise NotImplementedError("Only .npy and .fits implemented.")

    return data_2d