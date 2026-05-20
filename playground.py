# from pathlib import Path

# crops_dir = Path("crops/Chaceon micronesicus")
# to_delete = list(crops_dir.glob("PLW_dscm_05_*.jpg"))
# print(f"Found {len(to_delete)} files to delete")

# # Preview first, then uncomment the deletion
# for f in to_delete:
#     f.unlink()

import pandas as pd

crabs = pd.read_csv("mission05_crabs.csv")

high_conf_crabs = crabs[crabs["confidence"] >= 0.7]

print(high_conf_crabs.head())

print("Length of original dataframe:", len(crabs))
print("Length of filtered confidence dataframe:", len(high_conf_crabs))

high_conf_crabs.to_csv("high_conf_mission05_crabs.csv")