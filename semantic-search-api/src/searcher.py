from src.embedder import embed
from src.indexer import get_collection


def search(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = embed(query)
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "tags": results["metadatas"][0][i]["tags"],
            "content": results["documents"][0][i],
            "similarity_score": round(1 - results["distances"][0][i], 4)
        })

    return hits
