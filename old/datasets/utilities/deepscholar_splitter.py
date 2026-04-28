import pandas as pd

papers = pd.read_csv("datasets/deepscholar-bench/paper_content.csv", index_col=0)
citations = pd.read_csv("datasets/deepscholar-bench/citations.csv", index_col=0)

cited_papers = pd.DataFrame(citations['cited_paper_title']).drop_duplicates(subset='cited_paper_title')

cited_papers_63 = cited_papers.sample(63)
cited_papers_63.to_csv("datasets/deepscholar-bench/cited_papers_63.csv", index=False)
