
import os
from glob import glob
import numpy as np



def parse_filename(filename: str) -> tuple[float, float, float]:
    """
    File name convention (using examples):
    lm0008_09500_0420_0020_0000_Vsini_0000.rgs → [M/H] = -0.8 dex, Teff = 9500 K, logg = 4.2 dex, vmicro = 2.0 km s-1, vmacro = 0.0 km s-1, vsini = 0.0 km s-1 
    lp0005_13500_0310_0020_0000_Vsini_0000.rgs → [M/H] = 0.5 dex, Teff = 13500 K, logg = 3.1 dex, vmicro = 2.0 km s-1, vmacro = 0.0 km s-1, vsini = 0.0 km s-1 
    """
    basename = os.path.basename(filename)
    m_h, teff, logg, *_ = basename.split("_")
    m_h = float(m_h.replace("lp", "").replace("lm", "-")) / 10
    teff = float(teff)
    logg = float(logg) / 100
    # Vmicro, macro, and vsini are always fixed.
    #vmicro = float(vmicro) / 10
    #vmacro = float(vmacro) / 10
    #vsini = float(vsini.split(".")[0]) / 10
    return (teff, logg, m_h)

def read_normalized_flux(filename: str):
    return np.loadtxt(filename, usecols=(1, ))

def read_wavelength(filename: str):
    return np.loadtxt(filename, usecols=(0, ))



if __name__ == "__main__":

    from tqdm import tqdm

    paths = glob("*/*.rgs")

    parameter_names = ("Teff", "logg", "m_H")
    parameters = list(map(parse_filename, paths))

    # Get unique values for each parameter
    parameters = np.array(parameters)

    unique_parameters = {
        k: np.unique(parameters[:, i]) for i, k in enumerate(parameter_names)
    }

    shape = tuple([len(unique_parameters[k]) for k in parameter_names])
    for k, v in unique_parameters.items():
        print(f"{k}: ({v[0]:.0f}, {v[-1]:.0f}), n={len(v)}, diffs: {np.unique(np.round(np.diff(v), 1))}")

    wavelength = read_wavelength(paths[0])
    n_pixels = wavelength.size

    flux = np.nan * np.ones((*shape, n_pixels))
    converged_flag = np.zeros(shape, dtype=bool)

    print(f"flux.shape: {flux.shape}")
    for p, path in zip(parameters, tqdm(paths)):
        idx = tuple(
            np.where(unique_parameters[k] == v)[0][0]
            for k, v in zip(parameter_names, p)
        )
        f = read_normalized_flux(path)
        flux[idx] = f
        converged_flag[idx] = np.all(np.isfinite(f) * (f > 0))
    
    import h5py as h5

    with h5.File("hot_stars_apogee.h5", "w") as fp:
        fp.attrs["parameter_names"] = parameter_names
        for k, v in unique_parameters.items():
            fp.create_dataset(k, data=v)
        fp.create_dataset("wavelength", data=wavelength)
        fp.create_dataset("flux", data=flux)
        fp.create_dataset("converged_flag", data=converged_flag)


