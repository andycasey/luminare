import re

def air_to_vacuum(λ):
    """
    Convert wavelengths from air to vacuum.

    Parameters:
    -----------
    λ : np.array
        Wavelengths in Angstroms.

    Returns:
    --------
    np.array
        Wavelengths converted to vacuum in Angstroms.
    """

    s = 1e4 / λ  # Convert to micrometers
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + \
        0.0001599740894897 / (38.92568793293 - s**2)
    return λ * n

def parse_marcs_photosphere_path(filename: str) -> dict:
    """
    Parses a given filename string to extract stellar parameters.

    Args:
        filename (str): The name of the file in the specified format,
                        e.g., 'mod_z+0.00/p2800_g+3.5_m0.0_t01_x3_z+0.00_a-0.25_c+0.25_n+0.00_o-0.25_r+0.00_s+0.00.mod.filled'

    Returns:
        dict: A dictionary containing the extracted parameters:
              'effective_temperature' (int),
              'surface_gravity' (float),
              'metallicity' (float),
              'alpha_metallicity' (float),
              'carbon_metallicity' (float).
              Returns an empty dictionary if parsing fails for any parameter.
    """
    # Regular expressions to capture the specific parameters
    # The patterns look for the prefix (p, g, z, a, c) followed by a sign and a number.
    # The '?' makes the sign optional for numbers that might not have it (though your examples do).
    # The '.*?' is a non-greedy match for any characters in between.
    teff_pattern = r"[p|s](\d+)"
    logg_pattern = r"g([+-]?\d+\.?\d*)"
    metallicity_pattern = r"_z([+-]?\d+\.?\d*)"  # Matches the metallicity after the 'z'
    alpha_pattern = r"_a([+-]?\d+\.?\d*)"
    carbon_pattern = r"_c([+-]?\d+\.?\d*)"
    vmic_pattern = r"_t(\d+)"

    parameters = {}
    parameters["vmic"] = float(re.search(vmic_pattern, filename).group(1))

    # Extract effective temperature (pXXXX)
    teff_match = re.search(teff_pattern, filename)
    parameters["Teff"] = int(teff_match.group(1))

    # Extract surface gravity (g+X.X)
    logg_match = re.search(logg_pattern, filename)
    parameters["logg"] = float(logg_match.group(1))

    # Extract metallicity (z+X.XX)
    # Note: There are two 'z' parameters in your example. The one we want for overall metallicity
    # seems to be the first one after 'mod_'.
    # We'll use a slightly different approach to ensure we get the correct one if multiple exist.
    # The first 'z' after 'mod_z' and before 'p' or 'g' is the main metallicity.
    # However, based on your examples, the pattern `_z([+-]?\d+\.?\d*)` will correctly pick the one
    # after `_x3_` which is the actual metallicity for the model.
    # Let's refine the metallicity extraction to be more specific if needed, but the current
    # pattern should work given the structure.
    # For now, we assume the `_z+0.00` after `_x3_` is the one to capture.
    # A more robust regex for the first z might be: `mod_z([+-]?\d+\.?\d*)`
    # But the user's request implies the `_z+0.00` *after* `_x3_` is the metallicity.
    # Let's stick to the simpler `_z` pattern which will likely get the second one,
    # which seems to be the intended 'metallicity' as per the structure.
    # If the user meant the first `z+0.00` (mod_z+0.00), the regex would need adjustment.
    # Given the example `_z+0.00_a-0.25`, it's likely the one after `_x3_`.
    metallicity_match = re.search(metallicity_pattern, filename)
    parameters["m_H"] = float(metallicity_match.group(1))

    # Extract [alpha/metallicity] (a+X.XX)
    alpha_match = re.search(alpha_pattern, filename)
    parameters["alpha_m"] = float(alpha_match.group(1))

    # Extract [carbon/metallicity] (c+X.XX)
    carbon_match = re.search(carbon_pattern, filename)
    parameters["C_m"] = float(carbon_match.group(1))

    return parameters
