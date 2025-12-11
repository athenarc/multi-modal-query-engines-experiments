import pandas as pd
import numpy as np

df_enron_emails = pd.read_csv("datasets/enron_emails/enron_spam_data.csv").dropna(subset=['Message'])
df_enron_emails = df_enron_emails[df_enron_emails['Message'].str.len() < 10000]

indices = df_enron_emails.index.to_list()
np.random.shuffle(indices)
df_shuffled = df_enron_emails.loc[indices].reset_index(drop=True)

sizes = [500, 1000, 2000, 4000]

for size in sizes:
    out_df = df_shuffled.head(size)
    out_df.to_csv(f"datasets/enron_emails/enron_emails_shuffled_{size}.csv", index=False)