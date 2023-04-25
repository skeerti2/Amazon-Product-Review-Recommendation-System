""""
pandas profiling  for amazon book dataset and output is given in output.html
Submitted by : Gmon Kuzhiyanikkal
Date : 23 april 2023

"""

import pandas as pd

from ydata_profiling import ProfileReport

# Load the TSV file into a pandas dataframe
df = pd.read_csv("book.tsv", delimiter="\t", error_bad_lines=False, warn_bad_lines=True,nrows=20000)

# Generate a profile report using ydata profiling
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

# Save the report as an HTML file
profile.to_file("output.html")


