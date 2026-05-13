# ============================================================
# Taxonomic / Visual Similarity Hierarchy
# For supervised contrastive learning
# ============================================================

SPECIES_TO_GROUP = {

    # ========================================================
    # GRENADIERS / RATTAILS
    # ========================================================
    "Coryphaenoides": "grenadier",
    "Coryphaenoides rudis": "grenadier",
    "Coelorinchus": "grenadier",
    "Malacocephalus": "grenadier",
    "Macrouridae": "grenadier",

    # ========================================================
    # EELS
    # ========================================================
    "Synaphobranchus": "eel",
    "Synaphobranchidae": "eel",
    "Anguilliformes": "eel",
    "Ophichthidae": "eel",

    # ========================================================
    # EEL-LIKE / ELONGATE FISH
    # ========================================================
    "Halosauridae": "eel_like_fish",

    # ========================================================
    # COD-LIKE / CUSK EELS
    # ========================================================
    "Antimora rostrata": "cod_like_fish",
    "Physiculus rhodopinnis": "cod_like_fish",
    "Moridae": "cod_like_fish",

    "Ophidiidae": "cusk_eel",
    "Ophidiformes": "cusk_eel",

    # ========================================================
    # SHARKS
    # ========================================================
    "Echinorhinus cookei": "shark",
    "Pseudotriakis microdon": "shark",
    "Hexanchus griseus": "shark",
    "Apristurus": "shark",
    "Somniosidae": "shark",
    "Squaliformes": "shark",

    # ========================================================
    # RAYS
    # ========================================================
    "Plesiobatis daviesi": "ray",
    "Myliobatiformes": "ray",

    # ========================================================
    # CHIMAERAS
    # ========================================================
    "Hydrolagus": "chimaera",
    "Chimaeridae": "chimaera",

    # ========================================================
    # SNAPPERS / TELEOSTS
    # ========================================================
    "Etelis boweni": "snapper",
    "Etelis": "snapper",
    "Lutjanidae": "snapper",

    "Ruvettus pretiosus": "pelagic_fish",
    "Scombriformes": "pelagic_fish",

    "Trachichthyidae": "teleost_fish",
    "Actinopteri": "bony_fish",

    # ========================================================
    # KING CRABS
    # ========================================================
    "Neolithodes": "king_crab",
    "Lithodes longispina": "king_crab",
    "Lithodidae": "king_crab",

    # ========================================================
    # TRUE CRABS
    # ========================================================
    "Chaceon micronesicus": "crab",
    "Geryonidae": "crab",

    # ========================================================
    # SHRIMP
    # ========================================================
    "Heterocarpus": "shrimp",
    "Nematocarcinus": "shrimp",
    "Caridea": "shrimp",
    "Pandalidae": "shrimp",
    "Acanthephyridae": "shrimp",
    "Aristeidae": "shrimp",
    "Penaeoidea": "shrimp",

    # ========================================================
    # GENERAL DECAPODS
    # ========================================================
    "Decapoda": "decapod",

    # ========================================================
    # SQUAT LOBSTERS
    # ========================================================
    "Munida": "squat_lobster",

    # ========================================================
    # AMPHIPODS / ISOPODS / MYSIDS
    # ========================================================
    "Eurythenes": "amphipod",
    "Amphipoda": "amphipod",

    "Mysida": "mysid",

    "Munnopsidae": "isopod",
    "Isopoda": "isopod",

    # ========================================================
    # CORALS / CNIDARIANS
    # ========================================================
    "Primnoidae": "coral",
    "Anthozoa": "coral",

    "Aeginidae": "jelly_cnidarian",

    # ========================================================
    # ECHINODERMS
    # ========================================================
    "Goniasteridae": "sea_star",
    "Echinoidea": "sea_urchin",
    "Echinothuriidae": "sea_urchin",

    # ========================================================
    # UNKNOWN
    # ========================================================
    "Animalia": "unknown_animal",
}


# ============================================================
# HIGHER-LEVEL VISUAL / MORPHOLOGICAL GROUPS
# ============================================================

GROUP_TO_METAGROUP = {

    # --------------------------------------------------------
    # ELONGATE DEEP-SEA FISH
    # --------------------------------------------------------
    "grenadier": "elongate_fish",
    "eel": "elongate_fish",
    "eel_like_fish": "elongate_fish",
    "cod_like_fish": "elongate_fish",
    "cusk_eel": "elongate_fish",

    # --------------------------------------------------------
    # CARTILAGINOUS FISH
    # --------------------------------------------------------
    "shark": "cartilaginous_fish",
    "ray": "cartilaginous_fish",
    "chimaera": "cartilaginous_fish",

    # --------------------------------------------------------
    # TELEOSTS
    # --------------------------------------------------------
    "snapper": "teleost_fish",
    "pelagic_fish": "teleost_fish",
    "teleost_fish": "teleost_fish",
    "bony_fish": "teleost_fish",

    # --------------------------------------------------------
    # LARGE BENTHIC CRUSTACEANS
    # --------------------------------------------------------
    "king_crab": "benthic_crustacean",
    "crab": "benthic_crustacean",
    "shrimp": "benthic_crustacean",
    "decapod": "benthic_crustacean",
    "squat_lobster": "benthic_crustacean",

    # --------------------------------------------------------
    # SMALL CRUSTACEANS
    # --------------------------------------------------------
    "amphipod": "small_crustacean",
    "mysid": "small_crustacean",
    "isopod": "small_crustacean",

    # --------------------------------------------------------
    # BENTHIC INVERTEBRATES
    # --------------------------------------------------------
    "coral": "benthic_invertebrate",
    "sea_star": "benthic_invertebrate",
    "sea_urchin": "benthic_invertebrate",

    # --------------------------------------------------------
    # GELATINOUS
    # --------------------------------------------------------
    "jelly_cnidarian": "gelatinous",

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------
    "unknown_animal": "unknown",
}


# ============================================================
# SIMILARITY FUNCTION
# ============================================================

def get_similarity(label_a, label_b):
    """
    Returns:
        1.0  = strong positive
        0.5  = medium positive
        0.2  = weak positive
        0.0  = negative
    """

    # same exact label
    if label_a == label_b:
        return 1.0

    group_a = SPECIES_TO_GROUP.get(label_a)
    group_b = SPECIES_TO_GROUP.get(label_b)

    # unknown labels
    if group_a is None or group_b is None:
        return 0.0

    # same biological/visual group
    if group_a == group_b:
        return 0.5

    meta_a = GROUP_TO_METAGROUP.get(group_a)
    meta_b = GROUP_TO_METAGROUP.get(group_b)

    # same broader morphology
    if meta_a == meta_b:
        return 0.2

    # otherwise negative
    return 0.0