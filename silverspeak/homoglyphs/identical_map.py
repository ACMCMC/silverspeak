# %%
import os
import pathlib
import pandas as pd

# Chars map is a CSV file in this directory. Ignore everything after the # symbol.
chars_map = pd.read_csv(
    os.path.join(pathlib.Path(__file__).parent, "identical_map.csv"),
    sep=";",
    header=None,
    names=["original", "replacement"],
    comment="#",
)

# Strip the whitespace from the original and replacement columns
chars_map["original"] = chars_map["original"].str.strip()
chars_map["replacement"] = chars_map["replacement"].str.strip()

# Now, we have two columns with row like '0021' and '01C3'. We want to map this to their Unicode characters.
# We will use the chr() function to convert the hex to a Unicode character.
chars_map = {
    chr(int(row["original"], 16)): chr(int(row["replacement"], 16)) for _, row in chars_map.iterrows()
}
# %%
