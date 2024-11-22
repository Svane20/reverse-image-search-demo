import streamlit as st

from qdrant_client import QdrantClient
from io import BytesIO
import base64

collection_name = "animal_images"

if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None


def set_selected_record(record):
    st.session_state.selected_record = record


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=st.secrets.get('qdrant_db_url'),
        api_key=st.secrets.get('qdrant_api_key'),
    )


def get_initial_records():
    client = get_qdrant_client()

    records, _ = client.scroll(
        collection_name=collection_name,
        with_vectors=False,
        limit=12,
    )

    return records


def get_similar_records():
    client = get_qdrant_client()

    if st.session_state.selected_record is not None:
        return client.recommend(
            collection_name=collection_name,
            positive=[st.session_state.selected_record.id],
            limit=12,
        )

    return records


def get_bytes_from_base64(base64_string):
    return BytesIO(base64.b64decode(base64_string))


records = get_similar_records() if st.session_state.selected_record is not None else get_initial_records()

if st.session_state.selected_record:
    images_bytes = get_bytes_from_base64(st.session_state.selected_record.payload["thumbnail"])
    st.header("Images similar to:")
    st.image(images_bytes)
    st.divider()

column = st.columns(3)

for idx, record in enumerate(records):
    col_idx = idx % 3
    images_bytes = get_bytes_from_base64(record.payload["thumbnail"])

    with column[col_idx]:
        st.image(images_bytes)

        st.button(
            label="Find similar images",
            key=record.id,
            on_click=set_selected_record,
            args=[record],
        )
