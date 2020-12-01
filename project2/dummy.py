import pandas as pd
pd.options.display.max_colwidth=999
pd.options.display.max_rows=999
df = pd.read_csv("./data/project1/cv_results_.csv")
print(df)