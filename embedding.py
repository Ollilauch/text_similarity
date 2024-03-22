import pandas as pd
import sys
import numpy as np
from dotenv import load_dotenv
import os
import time
import itertools
import threading
from openai import OpenAI
import ast

load_dotenv()

cls = lambda: os.system('cls' if os.name=='nt' else 'clear')
done = False

client = OpenAI(api_key=os.getenv('OPENAI-API-KEY'))

# Alternative: text-embedding-3-small
embedding_model = "text-embedding-3-large"

def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], 
model=model).data[0].embedding

def cosine_similarity_matrix(embeddings):
    """Compute the cosine similarity matrix from embeddings."""
    norm_embeddings = np.round(embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis], 2)
    similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    return similarity_matrix

# ----------------Loading Bar-----------------------------------------------
def loading_spinner():
    spinner = itertools.cycle(['-', '/', '|', '\\'])
    sys.stdout.write("Embedding Text ")
    while done == False:
        sys.stdout.write(next(spinner))   # write the next character
        sys.stdout.flush()                # flush stdout buffer (actual character display)
        time.sleep(0.1)
        sys.stdout.write('\b')            # erase the last written char
# --------------------------------------------------------------------------

# Function to convert string representations of lists back to actual lists
def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        # Handle the error or return an empty list if the string cannot be converted
        return []

def main():
    # load & inspect dataset
    if len(sys.argv) < 2:
        print("\nUsage: python3 embedding.py <path_to_csv_file.csv>")
        exit()

    input_datapath = sys.argv[1]  
    if not os.path.splitext(input_datapath)[1] == ".csv":
        print("\nNo .csv file given")
        exit()

    df = pd.read_csv(input_datapath)

    # DEBUG
    # print(df.head())

    try:
        df = df[["Bereich", "Nr.", "Typ", "ID", "Beschreibung"]]
    except KeyError:
        print("\nInvalid file, must contain:\n'Bereich, Nr., Typ, ID, Beschreibung' columns")
        exit()

    df = df.dropna()
    df["combined"] = (
        "Beschreibung: " + df.Beschreibung.str.strip()
    )

    # DEBUG
    # print(df.head(2))

    t = threading.Thread(target=loading_spinner)
    t.start()

    ids = df["ID"].tolist()

    # Check if embedded file modifcation time is less than input file,
    # if yes create new embedded file
    embedded_csv_file_path = f"{os.path.splitext(input_datapath)[0]}_embedded.csv"
    if not os.path.exists(embedded_csv_file_path) or os.path.getmtime(input_datapath) > os.path.getmtime(embedded_csv_file_path):
        df["embedding"] = df.Beschreibung.apply(lambda x: get_embedding(x, model=embedding_model))
        df.to_csv(embedded_csv_file_path)

    else:
        df = None
        df = pd.read_csv(embedded_csv_file_path, sep=',', decimal='.', index_col=0)
        # Convert the string representations of embeddings back to lists
        df['embedding'] = df['embedding'].apply(string_to_list)

    # calculate text similarity from embedding
    similarity_matrix = cosine_similarity_matrix(df['embedding'].tolist())

    similarity_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)
    similarity_df.columns.name = "ID"

    # Saving the similarity matrix to a CSV file, with IDs as rows and columns
    similarity_csv_file_path = f"{os.path.splitext(input_datapath)[0]}_similarity_matrix.csv"
    similarity_df.to_csv(similarity_csv_file_path)

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
    sorted_similarities_csv_file_path = f"{os.path.splitext(input_datapath)[0]}_sorted_similarities.csv"
    sorted_similarities_df.to_csv(sorted_similarities_csv_file_path, index=False)

    # set global varaible done to True to stop loading bar
    global done
    done = True
    cls()
    
    end_input = 0
    print_counter = 0
    sorted_similarities_csv_df = pd.read_csv(sorted_similarities_csv_file_path)
    sorted_similarities_csv_df_rows = sorted_similarities_csv_df.shape[0]
    while(end_input != '9' and print_counter+10 < sorted_similarities_csv_df_rows):
        cls()
        print_counter+=10
        print(sorted_similarities_csv_df.iloc[print_counter-10:print_counter,:])
        print(f"Sorted similarities saved to {sorted_similarities_csv_file_path}")
        end_input = input("\nPress 1 or 9\n1: print 10 more rows\n9: exit the program\n")
    if(end_input != '9'):
        cls()
        print_counter+=10
        print(sorted_similarities_csv_df.iloc[print_counter-10:print_counter,:])
        print(f"Sorted similarities saved to {sorted_similarities_csv_file_path}")
        print("\nEnd of CSV file reached.")
    exit()


if __name__ == "__main__":
    main()
