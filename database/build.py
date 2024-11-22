import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Record

import base64
from io import BytesIO
import os
import uuid
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

from constants.directories import CAT_DATA_TRAIN_DIRECTORY

load_dotenv()


def get_qdrant_client():
    return QdrantClient(url=os.getenv("QDRANT_DB_URL"), api_key=os.getenv("QDRANT_API_KEY"))


def get_collection(client: QdrantClient, collection_name: str = "animal_images"):
    collection_exists = client.collection_exists(collection_name)
    if not collection_exists:
        print(f"Collection {collection_name} does not exist. Creating...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1000,
                distance=Distance.COSINE
            )
        )

    return client.get_collection(collection_name)


def generate_thumbnail(image_path, size=(64, 64)):
    """Generate a 64x64 thumbnail and return it as a base64 string."""
    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail(size)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error generating thumbnail for {image_path}: {e}")
        return None


def build_records_for_collection(processor, model, image_paths, batch_size=8):
    records = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Building records"):
        # Load and preprocess images in batches
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        thumbnails = []
        for path in batch_paths:
            try:
                batch_images.append(Image.open(path).convert("RGB"))
                thumbnails.append(generate_thumbnail(path))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue  # Skip problematic images

        if not batch_images:
            continue  # Skip if no images could be loaded

        inputs = processor(batch_images, return_tensors="pt").to(model.device)

        # Generate embeddings
        with torch.inference_mode():
            embeddings = model(**inputs).logits.cpu().numpy()  # Convert to NumPy

        # Build Qdrant records
        for path, embedding, thumbnail in zip(batch_paths, embeddings, thumbnails):
            record = Record(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),  # Convert embedding to list
                payload={
                    "type": Path(path).parts[-2],
                    "image_url": path,
                    "thumbnail": thumbnail  # Include thumbnail in payload
                }
            )
            records.append(record)
    return records


if __name__ == "__main__":
    # Initialize Qdrant client and collection
    client = get_qdrant_client()
    collection_name = "animal_images"
    animals_collection = get_collection(client, collection_name)

    # Prepare the first 500 images
    image_paths = [str(path) for path in CAT_DATA_TRAIN_DIRECTORY.rglob("*.jpg")]
    images_path = image_paths[:500]

    # Device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    model_name = "microsoft/resnet-50"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ResNetForImageClassification.from_pretrained(model_name).to(device)

    # Build records and upload to Qdrant
    records = build_records_for_collection(processor, model, images_path)
    for record in tqdm(records, desc="Uploading Records"):
        client.upload_points(
            collection_name=collection_name,
            points=[record]
        )

    print(f"Uploaded {len(records)} records to Qdrant collection '{collection_name}'.")
