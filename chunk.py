import pandas as pd

def chunk_rows_by_char_limit(transactions, chunk_size):
    chunks = []
    current_chunk = ""

    for row in transactions:
        text = str(row).strip()
        # print(row)

        if not text:
            continue

        candidate = text if current_chunk == "" else current_chunk + "\n\n" + text

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = text

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def save_chunks(chunks, filename):
    df_chunks = pd.DataFrame({
        "chunk_id": range(1, len(chunks) + 1),
        "text": chunks,
        "char_count": [len(chunk) for chunk in chunks]
    })
    df_chunks.to_csv(filename, index=False)

# -------- LOAD DATA --------
df = pd.read_csv("data/transactions_with_text.csv")
transactions = df["text"].tolist()
# print(transactions)

# -------- CREATE CHUNKS --------
chunks_500 = chunk_rows_by_char_limit(transactions, chunk_size=500)
chunks_1000 = chunk_rows_by_char_limit(transactions, chunk_size=1000)
chunks_2000 = chunk_rows_by_char_limit(transactions, chunk_size=2000)

# -------- SAVE --------
save_chunks(chunks_500, "data/chunks_500.csv")
save_chunks(chunks_1000, "data/chunks_1000.csv")
save_chunks(chunks_2000, "data/chunks_2000.csv")

# -------- PRINT INFO --------
print(f"500-char chunks: {len(chunks_500)}")
print(f"1000-char chunks: {len(chunks_1000)}")
print(f"2000-char chunks: {len(chunks_2000)}")