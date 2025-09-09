import numpy as np
import h5py as h5
from astropy.table import Table
from peewee import fn
from tqdm import tqdm

from astra.models import Source
from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
from astra.utils import expand_path

seed = 0.42
limit = 100_000
limit_wide_binaries = 20_000
minimum_oc_members = 12
output_path = "2025-09-09-spectrum-pack.h5"

# HDF structure:
# spectra/
#   apogee/
#       sdss_id
#       wavelength
#       flux
#       ivar
#       model_flux
#       continuum
#   boss/
#       sdss_id
#       wavelength
#       flux
#       ivar
#       model_flux
#       continuum
# sources/
#   sdss_id
#   ra
#   ...
# selections/
#    open_clusters
#    globular_clusters/
# ext/
#   apokasc/
#       sdss_id
#.  etc

moca_db = Table.read(expand_path("$MWM_ASTRA/aux/external-catalogs/mocadb_OC_members_w_sdss_id.csv"))
occam_db = Table.read(expand_path("$SAS_BASE_DIR/dr19/vac/mwm/apogee-occam/occam_member-DR19.fits"))
gc_members = Table.read(expand_path("$MWM_ASTRA/aux/external-catalogs/gc_members.csv"))
apokasc = Table.read(expand_path("$MWM_ASTRA/aux/external-catalogs/APOKASC_cat_v7.0.5.fits"))
wb = Table.read(expand_path("$MWM_ASTRA/aux/external-catalogs/2025-09-09-wide-binaries.csv"), format="ascii.tab")

# cross-match apokasc
q = (
    Source
    .select(Source.sdss_id, Source.gaia_dr3_source_id)
    .where(Source.gaia_dr3_source_id.in_(tuple(apokasc["GAIAEDR3_SOURCE_ID"])))
    .tuples()
)
apokasc_sdss_ids = [-1] * len(apokasc)
gaia_to_sdss = {g: s for s, g in q}
for i, g in enumerate(apokasc["GAIAEDR3_SOURCE_ID"]):
    apokasc_sdss_ids[i] = gaia_to_sdss.get(np.int64(g), -1)
apokasc["sdss_id"] = apokasc_sdss_ids
keep = (apokasc["sdss_id"] > -1)
apokasc = apokasc[keep]

# restrict by min number of members
def restrict_to_minimum_by_group(table, group_by, min_members, table_name=""):
    before = len(table)
    table = table.group_by(group_by)
    group_size = list(map(len, table.groups))
    keep_names = [group[group_by][0] for group, size in zip(table.groups, group_size) if size >= min_members]
    table = table[np.isin(table[group_by], keep_names)]
    print(f"Restricting {table_name} to groups with at least {min_members} members: {before} -> {len(table)}")
    return table

def randomly_subsample(table, n, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(table)), n, replace=False)
    return table[indices]


if minimum_oc_members is not None:
    moca_db = restrict_to_minimum_by_group(moca_db, "name", minimum_oc_members, "moca")
if limit_wide_binaries is not None:
    wb = randomly_subsample(wb, limit_wide_binaries)


selections = {
    "open_clusters": (
        # Open clusters
            Source.sdss_id.in_(tuple(moca_db["sdss_id"]))
        |   Source.sdss_id.in_(tuple(occam_db["SDSS_ID"]))
        #|    Source.assigned_to_carton_with_alt_name("mwm_cluster_openfiber") # 185,000 sources
    ),
    "globular_clusters": (
        # Globular clusters
           Source.sdss_id.in_(tuple(gc_members["sdss_id"]))
    ),
    "wide_binaries": (
        # Wide binaries
            Source.sdss_id.in_(tuple(wb["sdss_id_1"]))
        |   Source.sdss_id.in_(tuple(wb["sdss_id_2"]))
    ),
    "asteroseismology": (
        # Asteroseismology
        Source.sdss_id.in_(tuple(apokasc["sdss_id"]))
    ),
    "accessible_with_interferometry": (
        # Observable with CHARA/VLTI
            (Source.h_mag < 7)
        &   (Source.plx.is_null(False) & (Source.plx > 0))
        &   ((Source.g_mag - Source.rp_mag) < 4.5)
        &   ((Source.g_mag + 5 * fn.log10(Source.plx) - 5) < 4.5)
    ),
    "hot_stars": (
            Source.assigned_to_carton_with_name("manual_mwm_validation_hot_boss")
        |   Source.assigned_to_carton_with_name("manual_mwm_validation_hot_apogee")
    ),
    "observed_with_both_instruments": (
        # Observed with both instruments
        (Source.n_apogee_visits > 0) & (Source.n_boss_visits > 0)
    ),
    "validation": (
        # Validation cartons
        Source.assigned_to_carton_attribute("program", "mwm_validation")
    )
}


max_len = max(len(name) for name in selections.keys())
where = None
for name, w in selections.items():
    where = w if where is None else (where | w)

Source.select(fn.setseed(seed)).execute()
q_sources = (
    Source
    .select()
    .where(where)
    .order_by(fn.Random())
    .limit(limit)
)
all_sdss_ids = [s.sdss_id for s in q_sources]

for name, w in selections.items():
    count = Source.select().where(w).count()
    subset_count = Source.select().where(w).where(Source.sdss_id.in_(all_sdss_ids)).count()
    print(f"{name: <{max_len}}: {count:,} sources ({subset_count:,} in subset)")


print(f"Total sources: {q_sources.count():,}")

q = Source.select().where(Source.sdss_id.in_(all_sdss_ids))

# estimate file size
n_boss_spectra = q.where(Source.n_boss_visits > 0).count()
n_apogee_spectra = q.where(Source.n_apogee_visits > 0).count()

n_bytes_per_float = 4
n_arrays_per_spectrum = 4 # flux, ivar, model, continuum
n_apogee_pixels = n_apogee_spectra * n_arrays_per_spectrum * 8575 
n_boss_pixels = n_apogee_spectra * n_arrays_per_spectrum * 4648

n_pixels = n_apogee_pixels + n_boss_pixels
n_bytes = n_pixels * n_bytes_per_float
print(f"Estimated size of spectra: {n_bytes / 1e9:.1f} GB")

# Get apogee spectra together.
q_apogee = (
    ApogeeCoaddedSpectrumInApStar
    .select(
        ApogeeCoaddedSpectrumInApStar,
        Source.sdss_id.alias("sdss_id")
    )
    .distinct(Source.sdss_id)
    .join(Source, on=(Source.pk == ApogeeCoaddedSpectrumInApStar.source_pk))
    .where(Source.sdss_id.in_(all_sdss_ids))
)
count = q_apogee.count()
flux = np.ones((count, 8575), dtype=np.float32)
ivar = np.zeros((count, 8575), dtype=np.float32)
sdss_id = np.ones((count,), dtype=np.int64) * -1

for i, spec in enumerate(tqdm(q_apogee)):
    try:
        flux[i, :] = spec.flux
        ivar[i, :] = spec.ivar
        sdss_id[i] = spec.source.sdss_id
    except:
        continue

keep = (sdss_id > -1)
flux, ivar, sdss_id = (flux[keep], ivar[keep], sdss_id[keep])


with h5.File(output_path, "w") as fp:
    sources_group = fp.create_group("sources")
    converters = {
        "AUTO": (np.int64, -1),
        "BIGINT": (np.int64, -1),
        "TEXT": (h5.special_dtype(vlen=str), ""),
        "INT": (int, -1),
        "BLOB": (bytes, b''),
        "FLOAT": (float, np.nan)
    }
    sources = list(q_sources)
    for key in sources[0].__data__.keys():
        try:
            dtype, null = converters[getattr(Source, key).field_type]
        except KeyError:
            print(f"Dropping {key} source field")
            continue

        if key == "sdss5_target_flags":            
            ds = [np.frombuffer(bytes(getattr(s, key) or null), dtype=np.uint8) for s in sources]
            n = max(len(d) for d in ds)
            data = np.zeros((len(ds), n), dtype=np.uint8)
            for i, d in enumerate(ds):
                data[i, :len(d)] = d                        
        else:
            data = [(getattr(s, key) or null) for s in sources]

        sources_group.create_dataset(key, data=data)
        print(f"Created sources/{key} with {len(sources):,} entries")
    

    spectra_group = fp.create_group("spectra")

    apogee_group = spectra_group.create_group("apogee")
    apogee_group.create_dataset("sdss_id", data=sdss_id)
    apogee_group.create_dataset("wavelength", data=10**(4.179 + 6e-6 * np.arange(8575)), dtype=np.float32)
    apogee_group.create_dataset("flux", data=flux)
    apogee_group.create_dataset("ivar", data=ivar)

    selections_group = fp.create_group("selections")
    for n, w in selections.items():
        selection_sdss_ids = Source.select(Source.sdss_id).where(w).tuples()
        if limit is not None and selection_sdss_ids.count() > 0:
            selection_sdss_ids = set(next(zip(*list(selection_sdss_ids))))
            selection_sdss_ids = list(selection_sdss_ids.intersection(set(all_sdss_ids)))
        selections_group.create_dataset(n, data=np.array(selection_sdss_ids, dtype=np.int64))
        print(f"Created selections/{n} with {len(selection_sdss_ids):,} entries")
    

    ext_group = fp.create_group("ext")

    apokasc_group = ext_group.create_group("asteroseismology/apokasc")

    for col in apokasc.colnames:
        apokasc_group.create_dataset(col, data=np.array(apokasc[col]))

    wb_group = ext_group.create_group("wide_binaries")
    for col in wb.colnames:
        wb_group.create_dataset(col, data=np.array(wb[col]))

    gc_group = ext_group.create_group("globular_clusters")
    for key in ("apogee_id", "gc_name"):
        gc_members[key] = gc_members[key].astype("S")
    for col in gc_members.colnames:        
        gc_group.create_dataset(col, data=gc_members[col])

    oc_group = ext_group.create_group("open_clusters")
    occam_group = oc_group.create_group("occam")
    for col in occam_db.colnames:
        occam_group.create_dataset(col, data=np.array(occam_db[col]))
    moca_group = oc_group.create_group("moca")
    moca_db["name"] = [n.encode("utf-8") for n in moca_db["name"]]
    for key in ("moca_aid", "name"):
        moca_db[key] = moca_db[key].astype("S")
    for col in moca_db.colnames:
        moca_group.create_dataset(col, data=np.array(moca_db[col]))


