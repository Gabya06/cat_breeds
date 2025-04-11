import chromadb


def create_chromadb(db_name: str, embedding_function, delete_exisiting: bool = True):
    """
    Create ChromaDB collection with given name and embedding function.

    Parameters
    ----------
    db_name : str
        Name of the ChromaDB collection.
    embedding_function : EmbeddingFunction
        Function to generate embeddings.
    delete_exisiting: Bool
        Whether to delete exisiting chromaDB before adding documents. Defaults to True
    """

    # create db and include metadata
    chroma_client = chromadb.Client()
    # delete existing collection
    if delete_exisiting:
        try:
            chroma_client.delete_collection(name=db_name)
        except Exception:
            pass
    # db = chroma_client.create_collection(name=db_name, embedding_function=embedding_function)
    db = chroma_client.get_or_create_collection(name=db_name, embedding_function=embedding_function)

    return db
