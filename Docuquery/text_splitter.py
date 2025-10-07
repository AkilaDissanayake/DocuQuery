# Create chunks
from typing import List, Tuple  #For type hints

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
         Split text into overlapping chunks.
         Returns a list of chunk strings.
    """
    if chunk_size <= overlap:    #Ensure chunk size is larger than overlap to avoid infinite loop
        raise ValueError("chunk_size must be larger than overlap")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:     #Loop until the entire text is processed
        end = min(start + chunk_size, text_len)  #Compute end index for current chunk
        chunk = text[start:end]                  #
        chunks.append(chunk)
        if end == text_len:     #End of the text
            break
        start = end - overlap   #Move start to get overlap
    return chunks
