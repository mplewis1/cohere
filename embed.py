import cohere
# import pandas as pd
# import numpy as np
# import altair as alt
# import textwrap as tr

api_key = 'cUkUMhISEr8QsUhZ8uaVMxZtdL3UJrlaESCyNtHR'
co = cohere.Client(api_key)

df = pd.read_csv("https://github.com/cohere-ai/notebooks/raw/main/notebooks/data/hello-world-kw.csv", names=["search_term"])
df.head()

def embed_text(texts):
  output = co.embed(
                model="embed-english-v2.0",
                texts=texts)
  embedding = output.embeddings

  return embedding

df["search_term_embeds"] = embed_text(df["search_term"].tolist())