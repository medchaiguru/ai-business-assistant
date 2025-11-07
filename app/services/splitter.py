from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language="markdown",  # works well for Firecrawl output
    chunk_size=800,
    chunk_overlap=150,
)

