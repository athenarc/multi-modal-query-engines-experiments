import pandas as pd
import numpy as np

df_banking_data = pd.read_csv("datasets/banking_data/banking_data_test.csv").dropna(subset=['text']).drop_duplicates(subset=['text'])

indices = df_banking_data.index.to_list()
np.random.shuffle(indices)
df_shuffled = df_banking_data.loc[indices].reset_index(drop=True)
sizes = [500, 1000, 2000, 4000]

for size in sizes:
    out_df = df_shuffled.head(size)
    out_df.to_csv(f"datasets/enron_emails/enron_emails_shuffled_{size}.csv", index=False)