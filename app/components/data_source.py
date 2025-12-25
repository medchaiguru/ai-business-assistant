import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def json_data_to_langchain_docs(
    json_path: str,
    url_page_map: dict[str, Any]
) -> list[Document]:
    path = Path(json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in data:
        url = entry.get("metadata", {}).get("url", "")
        page_name = url_page_map.get(url, "Unknown")
        doc = Document(
            page_content=entry.get("page_content", ""),
            metadata={**entry.get("metadata", {}), "page_name": page_name},
        )
        docs.append(doc)
    return docs
