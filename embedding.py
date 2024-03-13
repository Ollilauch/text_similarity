import pandas as pd
import sys
import numpy as np
from dotenv import load_dotenv
import os
import time
import itertools
import threading
from openai import OpenAI

load_dotenv()

cls = lambda: os.system('cls' if os.name=='nt' else 'clear')
done = False

client = OpenAI(api_key=os.getenv('OPENAI-API-KEY'))

embedding_model = "text-embedding-3-small"

def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], 
model=model).data[0].embedding

def cosine_similarity_matrix(embeddings):
    """Compute the cosine similarity matrix from embeddings."""
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    return similarity_matrix

# ----------------Loading Bar-----------------------------------------------
def loading_spinner():
    spinner = itertools.cycle(['-', '/', '|', '\\'])
    sys.stdout.write("Loading ")
    while done == False:
        sys.stdout.write(next(spinner))   # write the next character
        sys.stdout.flush()                # flush stdout buffer (actual character display)
        time.sleep(0.1)
        sys.stdout.write('\b')            # erase the last written char
# --------------------------------------------------------------------------

def main():
    # load & inspect dataset
    if len(sys.argv) < 2:
        print("usage: python3 embedding.py <csv_file.csv>")
        exit()

    input_datapath = sys.argv[1]  # to save space, we provide a pre-filtered dataset
    df = pd.read_csv(input_datapath, index_col=0)
    # DEBUG
    print(df.head())
    df = df[["Nr.", "Typ", "ID", "Beschreibung"]]
    df = df.dropna()
    df["combined"] = (
        "Beschreibung: " + df.Beschreibung.str.strip()
    )
    # DEBUG
    print(df.head(2))

    t = threading.Thread(target=loading_spinner)
    t.start()

    ids = df["ID"].tolist()

    df["embedding"] = df.Beschreibung.apply(lambda x: get_embedding(x, model=embedding_model))

    similarity_matrix = cosine_similarity_matrix(df['embedding'].tolist())

    similarity_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)

    # Saving the similarity matrix to a CSV file, with IDs as rows and columns
    output_csv_file_path = f"{os.path.splitext(input_datapath)[0]}_similarity_matrix.csv"
    similarity_df.to_csv(output_csv_file_path)

    # Extract the upper triangle of the matrix, excluding the diagonal
    triu_indices = np.triu_indices_from(similarity_df, k=1)
    similarities = similarity_df.values[triu_indices]
    pair_ids = [(df["ID"].iloc[i], df["ID"].iloc[j]) for i, j in zip(*triu_indices)]

    # Create a DataFrame for sorted similarities
    sorted_similarities_df = pd.DataFrame({
        'Paar_IDs': pair_ids,
        'Aehnlichkeit': similarities
    })

    sorted_similarities_df = sorted_similarities_df.sort_values(by='Aehnlichkeit', ascending=False).reset_index(drop=True)

    # Save the sorted similarities to a CSV file
    output_csv_file_path = f"{os.path.splitext(input_datapath)[0]}_sorted_similarities.csv"
    sorted_similarities_df.to_csv(output_csv_file_path, index=False)

    # set global varaible done to True to stop loading bar
    global done
    done = True
    cls()

    print(f"Sorted similarities saved to {output_csv_file_path}")
    df.to_csv(f"{os.path.splitext(sys.argv[1])[0]}_embedded.csv")

if __name__ == "__main__":
    main()
