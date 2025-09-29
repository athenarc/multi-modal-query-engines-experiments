import pandas as pd
import numpy as np

df_enron_emails = pd.read_csv("datasets/enron_emails/enron_spam_data.csv")

indicies = df_enron_emails.index.to_list()
np.random.shuffle(indicies)

shuffled_1000 = df_enron_emails.loc[indicies].reset_index(drop=True).head(30000)

shuffled_1000.to_csv("datasets/enron_emails/enron_emails_shuffled_30000.csv")
