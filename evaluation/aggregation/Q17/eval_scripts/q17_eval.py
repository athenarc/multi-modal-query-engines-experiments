import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=1000, const=1000, type=int, help="The input size")
args = parser.parse_args()

enron_emails = pd.read_csv(f"datasets/enron_emails/enron_emails_sample_{args.size}.csv")

enron_emails = enron_emails['Spam/Ham'].value_counts()
spam_count = enron_emails.get('spam', 0)
print("Spam count: ", spam_count)

with open('statistics/aggregation/Q17.log', 'a') as file:
    file.write(f"Ground Truth Answer: {spam_count}" + "\n")
    file.write("------------------------------------------------------\n\n\n")