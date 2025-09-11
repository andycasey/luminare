import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(data, mo, np):
    selection = "hot_stars"
    mask = np.isin(data["sources/sdss_id"][:], data[f"selections/{selection}"][:])

    keys = ("sdss_id", "ra", "dec", "zgr_teff", "zgr_logg", "zgr_fe_h")
    DL = { key: data[f"sources/{key}"][:][mask] for key in keys }
    rows = [dict(zip(DL,t)) for t in zip(*DL.values())]
    table = mo.ui.table(
        data=rows, 
        label=f"Stars in `{selection}` selection",
        selection="single"
    )
    table
    return (table,)


@app.cell(hide_code=True)
def _(
    A,
    estimate_vsini,
    f,
    fit_spectrum,
    flux,
    ivar,
    jnp,
    model,
    np,
    plt,
    rf,
    sdss_ids,
    table,
    λ,
):
    sdss_id = table.value[0]["sdss_id"]
    index = sdss_ids.searchsorted(sdss_id)
    θ_w, θ_c, rectified_flux, continuum = f(flux[index], ivar[index])


    θ_initial = jnp.hstack([
        model.transform(np.hstack([rf.predict(np.atleast_2d(θ_w))[0], 0])),
        θ_c
    ])
    vsini_index = model.label_names.index("vsini")
    vsini, vsinis, chi2 = estimate_vsini(
        model, flux[index], ivar[index], θ_initial, vsini_index
    )
    θ_initial = θ_initial.at[vsini_index].set(vsini)

    r = fit_spectrum(flux[index], ivar[index], A, θ_initial=θ_initial)

    continuum = A @ r[-A.shape[1]:]

    p = dict(zip(model.label_names, model.inverse_transform(r)))
    model_label = f"Teff={p['Teff']:.0f}, logg={p['logg']:.2f}, vsini={p['vsini']:.0f}, m_H={p['m_H']:.2f}"

    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    ax.plot(λ, flux[index] / continuum, c='k')
    ax.plot(λ, model(r) / continuum, c="tab:red", label=model_label)
    ax.set_ylim(0.5, 1.2)
    ax.set_xlim(λ[0], λ[-1])
    ax.legend(loc="upper left")
    fig

    return


@app.cell(hide_code=True)
def _(default_initial_parameters_dict, transform):
    import sys
    import h5py as h5
    import numpy as np
    import optimistix
    import pickle
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import marimo as mo

    from scipy import optimize as op
    from time import time
    from functools import partial

    from luminare.emulator import create_stellar_spectrum_model
    from luminare.continuum import create_design_matrix, initial_theta
    from luminare.utils import air_to_vacuum
    from luminare.initial import create_initial_estimator, estimate_vsini



    with open("hot_stars_apogee.model", "rb") as fp:
        serialised_model = pickle.load(fp)

    with open("hot_stars_apogee.model.rf", "rb") as fp:
        rf = pickle.load(fp)

    λ = 10**(4.179 + 6e-6 * jnp.arange(8575))

    continuum_n_modes = 9
    continuum_regions = (
        (15120.0, 15820.0),
        (15840.0, 16440.0),
        (16450.0, 16960.0),
    )

    label_names = serialised_model["parameter_names"]

    model = create_stellar_spectrum_model(
        λ=λ,
        H=serialised_model["H"],
        X=serialised_model["X"],
        stellar_label_names=serialised_model["parameter_names"],
        min_stellar_labels=jnp.array(serialised_model["min_parameters"]),
        max_stellar_labels=jnp.array(serialised_model["max_parameters"]),
        n_stellar_label_points=serialised_model["n_points_per_parameter"],
        λ_model=serialised_model["λ"],
        n_modes=serialised_model["n_modes"],
        spectral_resolution=serialised_model["spectral_resolution"],
        max_vsini=400.0,
        continuum_n_modes=continuum_n_modes,
        continuum_regions=continuum_regions,
        model_in_air_wavelengths=(serialised_model["medium"] == "air"),
    )

    data = h5.File("../data/2025-09-09-spectrum-pack.h5", "r")

    @jax.jit
    def residual(θ, args):
        flux, inv_sigma = args
        return (model(θ) - flux) * inv_sigma

    A = create_design_matrix(λ, continuum_regions, continuum_n_modes)

    is_hot_star = np.isin(data["spectra/apogee/sdss_id"][:], data["selections/hot_stars"][:])

    flux = data["spectra/apogee/flux"][is_hot_star]
    ivar = data["spectra/apogee/ivar"][is_hot_star]
    sdss_ids = data["spectra/apogee/sdss_id"][is_hot_star]

    bad = (~np.isfinite(flux)) | (~np.isfinite(ivar))
    flux[bad] = 0.0
    ivar[bad] = 0.0

    flux, ivar = (jnp.array(flux), jnp.array(ivar))


    H_i = np.array([np.interp(λ, air_to_vacuum(serialised_model["λ"]), h) for h in serialised_model["H"]])
    A_i = np.hstack([H_i.T, A])
    f = create_initial_estimator(H_i.T, A)

    @partial(jax.jit, static_argnames=("atol", "rtol"))
    def fit_spectrum(flux, ivar, A, θ_initial, atol=1e-3, rtol=1e-3):
        use = (ivar > 0) * jnp.any(A != 0, axis=1)

        if θ_initial is None:
            θ_initial = transform(jnp.hstack([
                jnp.array([
                    default_initial_parameters_dict[k]
                    for k in label_names
                ]),
                initial_theta(len(continuum_regions), continuum_n_modes)
            ]))

            initial_rectified_flux = model(θ_initial)
            pseudo_rectified_flux = flux / initial_rectified_flux
            pseudo_rectified_ivar = ivar * initial_rectified_flux**2

            Y = jnp.where(use, pseudo_rectified_flux, 0.0)
            C_inv = jnp.diag(jnp.where(use, pseudo_rectified_ivar, 0.0))
            X, *extras = jnp.linalg.lstsq(A.T @ C_inv @ A, A.T @ C_inv @ Y, rcond=None)

            θ_initial = jnp.hstack([θ_initial[:θ_initial.size - X.size], X])

        inv_sigma = jnp.sqrt(ivar)

        t = -time()
        r = optimistix.least_squares(
            residual,
            args=(flux, inv_sigma),
            solver=optimistix.LevenbergMarquardt(rtol=rtol, atol=atol),
            y0=θ_initial,
            throw=True,
            max_steps=4096
        )
        t += time()
        #print(f"Optimisation took {t:.2f} seconds")

        return r.value


    sdss_id_dropdown = mo.ui.dropdown(
        options=list(map(str, sdss_ids)),
        value=f"{sdss_ids[0]}",
        label="SDSS_ID",
        searchable=True,
    )
    return (
        A,
        data,
        estimate_vsini,
        f,
        fit_spectrum,
        flux,
        ivar,
        jnp,
        mo,
        model,
        np,
        plt,
        rf,
        sdss_ids,
        λ,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
