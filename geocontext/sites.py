"""Photo-site definitions (batch 2, hand-picked before the map-API selector).

This table used to live in `bin/survey_candidates.py` and was imported across
scripts by `fetch_candidates.py` and `make_picker.py`. On release that script
is renamed to `01_survey_sites.py`, and a module name starting with a digit
cannot be imported -- which broke both scripts in the released package. The
table is library data rather than script logic, so it moved here.

Coordinates are always taken *inside a block* rather than on a monument. This
benchmark measures an ordinary corner of a streetscape, not a postcard. Site
selection also deliberately avoids each city's single most iconic landmark
(Sacre-Coeur, Sagrada Familia, Notre-Dame, ...): otherwise a model can answer
from fame alone and the legibility stratification collapses to a constant.
"""

#: site_id -> (lat, lon, note)
CANDIDATES = {
    # --- Paris: three sites, all placed within a block rather than on a monument ---
    "paris_montmartre":   (48.8840, 2.3380, "Abbesses / rue des Martyrs, avoids Sacre-Coeur"),
    "paris_bastille":     (48.8530, 2.3690, "blocks east of Place de la Bastille"),
    "paris_cite":         (48.8560, 2.3410, "Place Dauphine, west end of Ile de la Cite, avoids Notre-Dame"),
    "paris_canal":        (48.8710, 2.3660, "Canal Saint-Martin (spare site, very non-iconic)"),

    # --- London ---
    "london_shoreditch":  (51.5250, -0.0780, "Shoreditch, brick and street art"),
    "london_covent":      (51.5120, -0.1230, "streets around Covent Garden"),
    "london_nottinghill": (51.5150, -0.2050, "Notting Hill pastel terraces"),

    # --- Barcelona: avoids Sagrada Familia ---
    "barcelona_gracia":   (41.4030, 2.1560, "old town of Gracia"),
    "barcelona_born":     (41.3850, 2.1810, "El Born medieval lanes"),

    # --- San Francisco ---
    "sf_mission":         (37.7600, -122.4190, "Mission District, murals and Victorians"),
    "sf_northbeach":      (37.8000, -122.4090, "North Beach"),
    "sf_hayes":           (37.7760, -122.4240, "Hayes Valley"),

    # --- Mexico City: Spanish-speaking and Global South, badly under-represented
    #     in existing geolocation benchmarks ---
    "cdmx_roma":          (19.4190, -99.1620, "Roma Norte"),
    "cdmx_condesa":       (19.4110, -99.1710, "Condesa"),

    # --- Second and third Tokyo sites (Shibuya already covered) ---
    "tokyo_shimokita":    (35.6610, 139.6680, "Shimokitazawa"),
    "tokyo_yanaka":       (35.7270, 139.7660, "Yanaka, old-town streets"),
}

#: Batch 3, the second expansion (2026-09-02): Wikidata-selected landmark
#: sites across 29 additional cities, picked by scripts/00_select_sites.py.
#: label = "<city>: <landmark>" so a bare coordinate dump is still legible.
#: Continent code "AN", not "NA", for the North America section below --
#: pandas.read_csv treats the literal string "NA" as a missing value by
#: default, which is exactly what silently blanked that column the first
#: time this table round-tripped through pandas.
BATCH3 = {
    # --- AF ---
    "marrakech_cervantes": (31.637215, -8.016661, "Marrakech: Cervantes institute library in Marrakesh"),
    "marrakech_place": (31.619900, -7.984800, "Marrakech: Place des Ferblantiers"),
    "marrakech_royal": (31.629444, -8.014722, "Marrakech: Royal Theatre"),
    # --- AN ---
    "boston_chinatown": (42.352400, -71.062600, "Boston: Chinatown station"),
    "boston_longfellow": (42.361667, -71.075278, "Boston: Longfellow Bridge"),
    "boston_park": (42.356389, -71.062500, "Boston: Park Street station"),
    "mexicocity_balderas": (19.427440, -99.149036, "Mexico City: Balderas"),
    "mexicocity_centro": (19.406637, -99.155753, "Mexico City: Centro Médico"),
    "mexicocity_insurgentes": (19.423292, -99.163177, "Mexico City: Insurgentes"),
    "newyork_amsterdam": (40.756111, -73.987778, "New York: New Amsterdam Theatre"),
    "newyork_morgan": (40.748803, -73.981556, "New York: The Morgan Library & Museum"),
    "newyork_studio": (40.764361, -73.983778, "New York: Studio 54"),
    "sanfrancisco_cartoon": (37.787100, -122.401000, "San Francisco: Cartoon Art Museum"),
    "sanfrancisco_contemporary": (37.785800, -122.404000, "San Francisco: Contemporary Jewish Museum"),
    "sanfrancisco_joseph": (37.806600, -122.419000, "San Francisco: Joseph Conrad Square"),
    "toronto_rosedale": (43.676464, -79.388537, "Toronto: Rosedale"),
    "toronto_sherbourne": (43.672222, -79.376389, "Toronto: Sherbourne"),
    "toronto_tmu": (43.656537, -79.381022, "Toronto: TMU station"),
    # --- AS ---
    "hanoi_botanic": (21.040556, 105.831667, "Hanoi: Hanoi Botanic Garden"),
    "hanoi_college": (21.022136, 105.842789, "Hanoi: Hanoi College of Fine Arts"),
    "hanoi_well": (21.029820, 105.835925, "Hanoi: Well of Heavenly Clarity"),
    "hongkong_fortress": (22.288100, 114.193600, "Hong Kong: Fortress Hill station"),
    "hongkong_sai": (22.285600, 114.143000, "Hong Kong: Sai Ying Pun station"),
    "hongkong_statue": (22.281650, 114.160040, "Hong Kong: Statue Square"),
    "singapore_asian": (1.287500, 103.851389, "Singapore: Asian Civilisations Museum"),
    "singapore_national": (1.297780, 103.854000, "Singapore: National Library Singapore"),
    "singapore_somerset": (1.300514, 103.839028, "Singapore: Somerset MRT station"),
    "telaviv_dizengoff": (32.078056, 34.774167, "Tel Aviv: Dizengoff Square"),
    "telaviv_habima": (32.073053, 34.779511, "Tel Aviv: Habima Square"),
    "telaviv_nahum": (32.060800, 34.766700, "Tel Aviv: Nahum Gutman Museum of Art"),
    "tokyo_artizon": (35.678889, 139.771944, "Tokyo: Artizon Museum"),
    "tokyo_ginza": (35.671231, 139.765000, "Tokyo: Ginza Station"),
    "tokyo_national": (35.690553, 139.754642, "Tokyo: The National Museum of Modern Art, Tokyo"),
    # --- EU ---
    "amsterdam_church": (52.376528, 4.901111, "Amsterdam: Church of St Nicholas"),
    "amsterdam_natura": (52.366111, 4.916667, "Amsterdam: Natura Artis Magistra"),
    "amsterdam_vondelpark": (52.356944, 4.866389, "Amsterdam: Vondelpark"),
    "athens_kotzia": (37.981667, 23.727778, "Athens: Kotzia Square"),
    "athens_monastiraki": (37.975985, 23.725390, "Athens: Monastiraki metro station"),
    "athens_national": (37.974167, 23.738333, "Athens: National Garden of Athens"),
    "barcelona_fontana": (41.402630, 2.152750, "Barcelona: Fontana"),
    "barcelona_lesseps": (41.406111, 2.149444, "Barcelona: Lesseps"),
    "barcelona_sants": (41.378889, 2.140000, "Barcelona: Barcelona Sants railway station"),
    "berlin_alte": (52.520810, 13.398353, "Berlin: Alte Nationalgalerie"),
    "berlin_jewish": (52.502312, 13.395447, "Berlin: Jewish Museum Berlin"),
    "berlin_st": (52.515833, 13.394722, "Berlin: St. Hedwig's Cathedral"),
    "budapest_batthyny": (47.506222, 19.038611, "Budapest: Batthyány Square"),
    "budapest_museum": (47.486111, 19.068333, "Budapest: Museum of Applied Arts"),
    "budapest_university": (47.479850, 19.056080, "Budapest: Budapest University of Technology and Economics"),
    "copenhagen_castle": (55.677222, 12.580000, "Copenhagen: Copenhagen Castle"),
    "copenhagen_det": (55.673333, 12.555278, "Copenhagen: Det Ny Theater"),
    "copenhagen_museum": (55.674333, 12.572611, "Copenhagen: Museum of Copenhagen"),
    "edinburgh_dugald": (55.954500, -3.184480, "Edinburgh: Dugald Stewart Monument"),
    "edinburgh_greyfriars": (55.946900, -3.191330, "Edinburgh: Greyfriars Bobby Fountain"),
    "edinburgh_museum": (55.951400, -3.179600, "Edinburgh: Museum of Edinburgh"),
    "lisbon_cais": (38.706130, -9.145080, "Lisbon: Cais do Sodré station"),
    "lisbon_edward": (38.728250, -9.152833, "Lisbon: Edward VII Park"),
    "lisbon_martim": (38.716812, -9.135749, "Lisbon: Martim Moniz"),
    "london_barbican": (51.520200, -0.095000, "London: Barbican Centre"),
    "london_lambeth": (51.495560, -0.119720, "London: Lambeth Palace"),
    "london_royal": (51.509167, -0.139444, "London: Royal Academy of Arts"),
    "madrid_pacfico": (40.401300, -3.675080, "Madrid: Pacífico"),
    "madrid_pera": (40.418100, -3.709340, "Madrid: Ópera"),
    "madrid_royal": (40.415020, -3.690910, "Madrid: Royal Spanish Academy"),
    "milan_milano": (45.468618, 9.175196, "Milan: Milano Cadorna railway station"),
    "milan_museo": (45.467296, 9.189339, "Milan: Museo Teatrale alla Scala"),
    "milan_santa": (45.454396, 9.187669, "Milan: Santa Maria presso San Celso Church"),
    "paris_bataclan": (48.863056, 2.370833, "Paris: Bataclan"),
    "paris_gare": (48.876944, 2.359167, "Paris: Gare de Paris-Est"),
    "paris_pont": (48.857500, 2.341667, "Paris: Pont Neuf"),
    "stockholm_grdet": (59.345833, 18.098889, "Stockholm: Gärdet metro station"),
    "stockholm_karlaplan": (59.338611, 18.090556, "Stockholm: Karlaplan metro station"),
    "stockholm_st": (59.313806, 18.072500, "Stockholm: St. Eric's Cathedral, Stockholm"),
    "vienna_heldenplatz": (48.206281, 16.363761, "Vienna: Heldenplatz"),
    "vienna_naschmarkt": (48.198900, 16.363600, "Vienna: Naschmarkt"),
    "vienna_university": (48.208419, 16.382219, "Vienna: University of Applied Arts Vienna"),
    # --- SA ---
    "bogota_general": (4.593936, -74.076900, "Bogota: General Archive of the Nation of Colombia"),
    "bogota_national": (4.609553, -74.068650, "Bogota: National Library of Colombia"),
    "bogota_planetarium": (4.612222, -74.068889, "Bogota: Planetarium of Bogotá"),
    "delhi_holy": (28.642700, 77.233500, "Delhi: Holy Trinity Church, Delhi"),
    "delhi_national": (28.610182, 77.234401, "Delhi: National Gallery of Modern Art"),
    "delhi_vijay": (28.614130, 77.217160, "Delhi: Vijay Chowk"),
    "lima_alejandro": (-12.068542, -77.022886, "Lima: Alejandro Villanueva Stadium"),
    "lima_equestrian": (-12.047611, -77.026000, "Lima: equestrian statue of Simón Bolívar"),
    "lima_estadio": (-12.067139, -77.033722, "Lima: Estadio Nacional del Perú (1897)"),
    "mumbai_bowen": (18.923333, 72.832656, "Mumbai: Bowen Memorial Methodist Church"),
    "mumbai_girgaon": (18.952103, 72.822195, "Mumbai: Girgaon"),
    "mumbai_mughal": (18.926900, 72.832700, "Mumbai: Mughal Museum"),
    "santiago_baslica": (-33.441300, -70.661800, "Santiago: Basílica del Salvador"),
    "santiago_moneda": (-33.445028, -70.655667, "Santiago: La Moneda"),
    "santiago_plaza": (-33.437967, -70.650400, "Santiago: Plaza de Armas"),
}

#: All sites for which a context ladder has been built, batches 1-3
#: combined. This is the superset `analysis/build_ladder.py` and
#: `scripts/05_build_ladders.py` both need; `CANDIDATES` above only covers
#: batch 2, which went through the `01_survey_sites.py` -> `02_fetch_images.py`
#: -> `04_select_images.py` screening flow. Batch 1 predates that flow (its
#: images came from the older MMS-VPR / early-Mapillary sourcing) and is kept
#: separate here rather than folded into CANDIDATES, since it was never
#: "surveyed" in that sense. Batch 3 went through the Wikidata-driven
#: `00_select_sites.py` flow instead of manual picking.
#:
#: NOTE: `sanjose_rose` is not an experimental site at all -- it is the San
#: Jose Rosicrucian Park / Municipal Rose Garden control used only to
#: validate the referenceability-vs-pageviews distinction in `audit.py`'s
#: module docstring.
SITES = {
    "nyc_soho":         (40.7233, -74.0030, "New York SoHo"),
    "tokyo_shibuya":    (35.6595, 139.7005, "Tokyo Shibuya"),
    "paris_marais":     (48.8590, 2.3620, "Paris Marais"),
    "sanjose_rose":     (37.3330, -121.9230, "San Jose Rose Garden (control, not an experimental site)"),
    **{k: (lat, lon, note) for k, (lat, lon, note) in CANDIDATES.items()},
    **{k: (lat, lon, note) for k, (lat, lon, note) in BATCH3.items()},
}
