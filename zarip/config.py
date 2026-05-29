import numpy as np

PROVINCE_COORDINATES = {
    'Region_I': (32.7, -18.2),     # Manicaland (Eastern Highlands)
    'Region_II': (31.0, -17.5),    # Mashonaland Central/West (High Rainfall)
    'Region_III': (29.8, -19.4),   # Midlands (Medium Rainfall)
    'Region_IV': (30.8, -20.1),    # Masvingo (Low Rainfall/Vulnerable)
    'Region_V': (29.0, -21.0)      # Matabeleland South (Arid)
}

REGIONS = {
    'Region_I':   {'name': 'Manicaland',         'ha': 25000.0,  'lon': 32.7, 'lat': -18.2},
    'Region_II':  {'name': 'Mashonaland',        'ha': 225000.0, 'lon': 31.0, 'lat': -17.5},
    'Region_III': {'name': 'Midlands',           'ha': 200000.0, 'lon': 29.8, 'lat': -19.4},
    'Region_IV':  {'name': 'Masvingo',           'ha': 175000.0, 'lon': 30.8, 'lat': -20.1},
    'Region_V':   {'name': 'Matabeleland South', 'ha': 125000.0, 'lon': 29.0, 'lat': -21.0},
}

DEFAULT_REGIONAL_EXPOSURES = {region: info['ha'] for region, info in REGIONS.items()}

PALETTE = {
    "bg_dark":   "#0D1B2A",
    "bg_mid":    "#1B2B3A",
    "bg_panel":  "#162233",
    "accent":    "#00C896",
    "accent2":   "#F59E0B",
    "danger":    "#EF4444",
    "text":      "#E8F4F8",
    "text_dim":  "#7A9BAF",
    "border":    "#2A4054",
    "highlight": "#1E3A52",
}


def calculate_distance_km(coord1, coord2) -> float:
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    dx = (lon1 - lon2) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2.0))
    dy = (lat1 - lat2) * 111.0
    return np.sqrt(dx**2 + dy**2)


def generate_spatial_correlation(decay_parameter: float = 200.0) -> np.ndarray:
    regions = list(PROVINCE_COORDINATES.keys())
    n = len(regions)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                d = calculate_distance_km(PROVINCE_COORDINATES[regions[i]], PROVINCE_COORDINATES[regions[j]])
                corr_matrix[i, j] = np.exp(-d / decay_parameter)
    return corr_matrix
