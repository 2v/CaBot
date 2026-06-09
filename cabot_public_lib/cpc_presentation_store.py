"""
Presentation-of-case exemplar store for CaBot-Public.

This is the PUBLIC-release store: it is built from a single downloadable parquet
(``cpc_presentation_index_100.parquet``) that holds the 100 public CPCs together with their
precomputed presentation-of-case embeddings. The collection lives in memory (chromadb Ephemeral
client) — there is no large on-disk ChromaDB to ship.

NOTE: exemplar retrieval here searches ONLY the 100 public CPCs. The original CaBot retrieved over
the full (private) CPC corpus, so retrieved exemplars will differ from the paper's runs.

The query path embeds the search text with ``text-embedding-3-small`` — the same model used to build
the index — so the stored and query vectors are comparable.
"""
import chromadb
from tqdm import tqdm
import pyarrow.parquet as pq

EMBED_MODEL = "text-embedding-3-small"


def load_index_parquet(parquet_path):
    """Load the public CPC exemplar index parquet into a list of record dicts."""
    return pq.read_table(parquet_path).to_pylist()


class CPCPresentationStore:
    def __init__(self, openai_client, collection_name="cpc_presentations"):
        self.openai_client = openai_client
        self.collection_name = collection_name
        # In-memory collection — the parquet is the persistent artifact, not a ChromaDB dir.
        self.client = chromadb.EphemeralClient()
        self.collection = None

    def get_embeddings(self, texts, batch_size=30):
        """Embed query text with the same model used to build the index."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", disable=len(texts) <= batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai_client.embeddings.create(input=batch, model=EMBED_MODEL)
            all_embeddings.extend(d.embedding for d in response.data)
        return all_embeddings

    def build_from_records(self, records):
        """Build the in-memory collection from parquet records (precomputed embeddings, no re-embed)."""
        self.collection = self.client.create_collection(
            name=self.collection_name, embedding_function=None)

        ids, documents, embeddings, metadatas = [], [], [], []
        for r in records:
            case_id = r["id"]
            year = r.get("year")
            decade = r.get("decade")
            ids.append(case_id)
            documents.append(r.get("presentation_of_case", ""))
            embeddings.append(list(r["embedding"]))
            metadatas.append({
                # ChromaDB metadata must be str/int/float/bool (no None).
                "year": int(year) if year is not None else "unknown",
                "decade": int(decade) if decade is not None else "unknown",
                "title": r.get("title", "") or "",
                "publication_date": r.get("publication_date", "") or "",
                "case_id": case_id,
            })

        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return self

    def build_from_parquet(self, parquet_path):
        """Convenience: load the parquet and build the collection."""
        return self.build_from_records(load_index_parquet(parquet_path))

    def search_with_filters(self, query, k=5, year_min=None, year_max=None, decade=None, exclude_id=None):
        """Embed the query and search the collection with optional year/decade/exclude filters."""
        if self.collection is None:
            raise ValueError("Collection not built. Call build_from_parquet()/build_from_records() first.")

        query_embedding = self.get_embeddings([query])[0]

        where_conditions = []
        if year_min is not None:
            where_conditions.append({"year": {"$gte": year_min}})
        if year_max is not None:
            where_conditions.append({"year": {"$lte": year_max}})
        if decade is not None:
            where_conditions.append({"decade": {"$eq": decade}})
        if exclude_id is not None:
            where_conditions.append({"case_id": {"$ne": exclude_id}})

        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            if results["documents"] and len(results["documents"][0]) > 0:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                scores = [1 - dist for dist in distances]  # cosine distance -> similarity
                case_ids = [meta["case_id"] for meta in metadatas]
                return scores, case_ids, documents, metadatas
            return [], [], [], []
        except Exception as e:
            print(f"Search failed: {e}")
            return [], [], [], []
