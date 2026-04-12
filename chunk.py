import pandas as pd

def chunk_rows_by_char_limit(records, chunk_size):
    chunks = []
    current_chunk = ""
    current_row_ids = []

    for record in records:
        row_id = record["Row ID"]
        text = str(record["text"]).strip()

        if not text:
            continue

        candidate = text if current_chunk == "" else current_chunk + "\n\n" + text

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            current_row_ids.append(row_id)
        else:
            if current_chunk:
                chunks.append({
                    "row_id": current_row_ids[0],   # safe here for 500-char chunks
                    "text": current_chunk
                })
            current_chunk = text
            current_row_ids = [row_id]

    if current_chunk:
        chunks.append({
            "row_id": current_row_ids[0],
            "text": current_chunk
        })

    return chunks

def save_chunks(chunks, filename):
    df_chunks = pd.DataFrame({
        "chunk_id": range(1, len(chunks) + 1),
        "row_id": [chunk["row_id"] for chunk in chunks],
        "text": [chunk["text"] for chunk in chunks],
        "char_count": [len(chunk["text"]) for chunk in chunks]
    })
    df_chunks.to_csv(filename, index=False)

# -------- LOAD DATA --------
df = pd.read_csv("data/transactions_with_text.csv")

records = df[["Row ID", "text"]].to_dict(orient="records")

# -------- CREATE CHUNKS --------
chunks_500 = chunk_rows_by_char_limit(records, chunk_size=500)

# -------- SAVE --------
save_chunks(chunks_500, "data/chunks_500.csv")

# -------- PRINT INFO --------
print(f"500-char chunks: {len(chunks_500)}")