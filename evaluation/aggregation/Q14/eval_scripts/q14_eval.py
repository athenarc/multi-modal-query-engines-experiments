import pandas as pd

enron_emails = pd.read_csv("datasets/enron_emails/enron_emails_shuffled_10000.csv")

print(enron_emails['Spam/Ham'].value_counts())