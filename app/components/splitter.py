from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,  # works well for Firecrawl output
    chunk_size=800,
    chunk_overlap=150,
)
