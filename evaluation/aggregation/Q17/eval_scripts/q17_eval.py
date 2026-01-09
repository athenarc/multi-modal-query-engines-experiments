import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=1000, const=1000, type=int, help="The input size")
args = parser.parse_args()

enron_emails = pd.read_csv(f"datasets/enron_emails/enron_emails_shuffled_{args.size}.csv")

enron_emails = enron_emails['Spam/Ham'].value_counts()
spam_count = enron_emails.get('spam', 0)
print(spam_count)