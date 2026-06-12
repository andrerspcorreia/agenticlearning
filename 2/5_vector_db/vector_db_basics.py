import chromadb

# Create a client (like a db)
client = chromadb.Client()

# Create a collection (like a table)
collection = client.create_collection("news")

# Add documents to the collection (like rows)
# It will create embeddings for each of the documents and store them.
# Chromadb by default uses the all-MiniLM-L6-v2 embedding model (a sentence-transformer from huggingface)
collection.add(
    ids = ["id1", "id2", "id3", "id4"],
    documents=[
        "Apple is leading in a smart phone game with iPhone sales up by 35%",
        "Tesla booked a minor profit of 1 billion $ in Q2",
        "Apples are high in fiber, vitamin C, and various antioxidants",
        "SpaceX got NASA contract worth 10 billion $",
    ]
)

# Get and show the embeddings
data = collection.get(
    include = ["documents", "embeddings", "metadatas"]
)
for emb in data["embeddings"]:
    print(emb)

# Query the db (search through the meaning not keywords/words, we don't have any document with words Elon or Musk and it will know to fetch Tesla and SpaceX.)
results = collection.query(query_texts = ["This is a query related to Elon Musk."], n_results = 2)
print(results)
