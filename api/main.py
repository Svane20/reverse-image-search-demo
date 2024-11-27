from fastapi import FastAPI, UploadFile, Request, Body
from fastapi.responses import Response
from contextlib import asynccontextmanager
from elasticsearch import Elasticsearch
import clip
import torch

from PIL import Image
from io import BytesIO
import os
import numpy as np
import base64
from typing import Tuple
from pydantic import BaseModel
import imagehash

ELASTIC_SEARCH_INDEX = "image_text_similarity"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if es.indices.exists(index=ELASTIC_SEARCH_INDEX):
        es.indices.delete(index=ELASTIC_SEARCH_INDEX)
    create_index(ELASTIC_SEARCH_INDEX)

    mapping = es.indices.get_mapping(index=ELASTIC_SEARCH_INDEX)
    print(f"INFO:     Mapping for index '{ELASTIC_SEARCH_INDEX}': {mapping['image_text_similarity']['mappings']}")

    yield


app = FastAPI(lifespan=lifespan)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
# https://github.com/openai/CLIP
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.to(device)

# Connect to elastic search
es_host = os.getenv("ES_HOST", "localhost")
es = Elasticsearch([f"http://{es_host}:9200"])


def create_index(index_name: str) -> None:
    """
    Create an Elasticsearch index with the given name if it does not exist.

    Args:
        index_name (str): Name of the index to create
    """
    if not es.indices.exists(index=index_name):
        # Define the mapping for the index
        mapping = {
            "mappings": {
                "properties": {
                    "image_path": {"type": "text"},
                    "thumbnail": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 512,  # Adjust dimensions as per the model,
                        "similarity": "cosine"
                    },
                    "phash": {
                        "type": "dense_vector",
                        "dims": 64,
                        "element_type": "bit",
                        "similarity": "l2_norm"
                    }
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
        print(f"INFO:     Index '{index_name}' created.")
    else:
        print(f"INFO:     Index '{index_name}' already exists.")


def extract_clip_embedding(image_bytes: bytes) -> np.ndarray:
    """
    Extract the CLIP embedding for an image.

    Args:
        image_bytes (bytes): Input image bytes

    Returns:
        np.ndarray: CLIP embedding for the image
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")  # Load image
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.inference_mode():
        # Generate the image embedding
        embedding = model.encode_image(input_tensor).cpu().numpy().flatten()

    return embedding


def generate_thumbnail(image_bytes: bytes, size: Tuple[int, int] = (128, 128)) -> str:
    """
    Generate a thumbnail from the image and return it as a base64-encoded string.

    Args:
        image_bytes (bytes): Input image bytes
        size (tuple): Thumbnail size (width, height). Default is (128, 128).

    Returns:
        str: Base64-encoded thumbnail image
    """
    # Load image and generate thumbnail
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail(size)

    # Save thumbnail to buffer
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    # Encode the thumbnail as base64
    thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return thumbnail_base64


def generate_phash_as_bit_array(image: Image) -> list:
    """
    Generate a 64-bit perceptual hash bit vector.
    """
    phash = imagehash.phash(image)  # Generate phash as a hex string
    pHash_bytes = bytes.fromhex(str(phash))  # Convert hex to bytes
    if len(pHash_bytes) != 8:  # Each byte represents 8 bits
        raise ValueError(f"Invalid pHash length. Expected 8 bytes, got {len(pHash_bytes)}.")

    bit_vector = []
    for byte in pHash_bytes:
        bit_vector.extend([int(bit) for bit in bin(byte)[2:].zfill(8)])  # Convert to binary
    if len(bit_vector) != 64:
        raise ValueError(f"Invalid bit vector length. Expected 64, got {len(bit_vector)}.")
    return bit_vector


@app.post("/index")
async def index_image(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Extract embedding and generate thumbnail
    embedding = extract_clip_embedding(image_bytes)
    thumbnail = generate_thumbnail(image_bytes)
    phash = generate_phash_as_bit_array(image)
    print(f"PHash: {phash}, Length: {len(phash)}")

    if not (isinstance(phash, list) and len(phash) == 64 and all(bit in [0, 1] for bit in phash)):
        raise ValueError(f"Invalid phash format. Expected 64 bits, got {len(phash)} bits: {phash}")

    # Use the correct index name
    doc = {
        "image_path": file.filename,
        "thumbnail": thumbnail,
        "embedding": embedding.tolist(),
        "phash": phash
    }
    print(f"Document Phash: {doc['phash']}, length: {len(doc['phash'])}")

    try:
        res = es.index(index=ELASTIC_SEARCH_INDEX, body=doc)
        return {"result": res}
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'info'):
            print(f"Elasticsearch response: {e.info}")
        raise e


@app.post("/search")
async def search_image(file: UploadFile, request: Request, top_k: int = 5):
    """
    Search for similar images using the input image.

    Args:
        file (UploadFile): The input image file to search for similar images.
        request (Request): FastAPI request object to construct the thumbnail URL.
        top_k (int): Number of top results to return.

    Returns:
        dict: Matching results with scores, image paths and thumbnails.
    """
    image_bytes = await file.read()

    # Extract image embedding
    query_embedding = extract_clip_embedding(image_bytes)

    # Convert embedding to list
    query_embedding = query_embedding.tolist()

    # PHash search
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    phash_bits = generate_phash_as_bit_array(image)
    p_response = search_by_phash(phash_bits, top_k)
    p_results = [
        {
            "score": hit["_score"]
        }
        for hit in p_response["hits"]["hits"]
    ]
    print(f"Phash results: {p_results}")

    # KNN search
    response = _knn_search_images(query_embedding, top_k)

    base_url = request.base_url
    results = [
        {
            "image_path": hit["_source"]["image_path"],
            "thumbnail": f"{base_url}thumbnail/{hit['_source']['image_path']}",
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]

    return {
        "total_hits": response["hits"]["total"]["value"],
        "results": results
    }


class TextQuery(BaseModel):
    query_text: str


@app.post("/query")
async def query_images(request: Request, job: TextQuery = Body(...), top_k: int = 5):
    """
    Query images using a text description.

    Args:
        job (TextQuery): The text query to match images.
        request (Request): FastAPI request object to construct the thumbnail URL.
        top_k (int): Number of top results to return.

    Returns:
        dict: Matching results with image paths and thumbnails.
    """
    query_text = job.query_text

    # Generate text embedding
    tokenized_text = clip.tokenize([query_text]).to(device)

    with torch.inference_mode():
        query_embedding = model.encode_text(tokenized_text).cpu().numpy().flatten()

    # KNN search
    response = _knn_search_images(query_embedding.tolist(), top_k)

    # Generate results with thumbnails
    base_url = request.base_url
    results = [
        {
            "image_path": hit["_source"]["image_path"],
            "thumbnail": f"{base_url}thumbnail/{hit['_source']['image_path']}",
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]

    return {
        "query_text": query_text,
        "total_hits": len(results),
        "results": results
    }


@app.get("/thumbnail/{image_path}")
async def get_thumbnail(image_path: str):
    """
    Retrieve and return the thumbnail for the given image_path.

    Args:
        image_path (str): Path of the image to retrieve the thumbnail for.

    Returns:
        Response: The thumbnail image as a binary response.
    """
    # Search for the document with the given image_path
    query = {
        "query": {
            "match": {
                "image_path": image_path
            }
        }
    }
    response = es.search(index=ELASTIC_SEARCH_INDEX, body=query)

    # If no document is found, return a 404 error
    if not response["hits"]["hits"]:
        return {"error": f"Thumbnail for '{image_path}' not found."}, 404

    # Get the base64-encoded thumbnail from the first hit
    thumbnail_base64 = response["hits"]["hits"][0]["_source"]["thumbnail"]

    # Decode the base64 thumbnail to binary
    thumbnail_binary = base64.b64decode(thumbnail_base64)

    # Return the thumbnail as an image response
    return Response(content=thumbnail_binary, media_type="image/jpeg")


def search_by_phash(phash_bits: list, top_k: int = 5):
    query = {
        "field": "phash",
        "query_vector": phash_bits,
        "k": top_k,
        "num_candidates": 10  # Adjust for performance vs. accuracy
    }

    response = es.knn_search(
        index=ELASTIC_SEARCH_INDEX,
        knn=query,
        source=True
    )
    return response


def _knn_search_images(dense_vector: list, knn: int):
    """
    Perform KNN search using the given dense vector.

    Args:
        dense_vector (list): Dense vector to search for similar images.
        knn (int): Number of nearest neighbors to return.

    Returns:
        dict: Elasticsearch response with matching images
    """
    source_fields = ["image_path", "thumbnail"]
    query = {
        "field": "embedding",
        "query_vector": dense_vector,
        "k": knn,
        "num_candidates": 10
    }

    response = es.search(
        index=ELASTIC_SEARCH_INDEX,
        fields=source_fields,
        knn=query,
        source=True
    )

    return response
