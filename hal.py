import os
import json
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from rich.status import Status
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import readline
import pyperclip
import numpy as np
import pickle
from datetime import datetime
import re

# Global variables
cache = {}

# Set environment variable to suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Set up OpenAI API to use LM Studio
openai.api_base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
openai.api_key = "not-needed"  # LM Studio doesn't require an API key

# Initialize components
console = Console()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths for persistent storage
VECTOR_STORE_PATH = os.path.expanduser("~/hal_vector_store.faiss")
TEXT_CHUNKS_PATH = os.path.expanduser("~/hal_text_chunks.pkl")
USER_PROFILE_PATH = os.path.expanduser("~/hal_user_profile.txt")
CACHE_PATH = os.path.expanduser("~/hal_file_cache.json")

# Named collections
COLLECTIONS = {
    "journal": {
        "path": "~/Memorabilia/memorabilia/Journal",
        "date_pattern": r"(\d{4}\.\d{2}\.\d{2})",  # YYYY.MM.DD
        "keywords": ["my_journal"],
    },
    # "documents": {
    # "path": "/path/to/your/general/documents",
    # No date_pattern specified for this collection
    # "keywords": ["]
    # },
    # Add more collections as needed
}


def get_queried_collection(user_query):
    lower_query = user_query.lower()
    for collection, info in COLLECTIONS.items():
        if any(keyword in lower_query for keyword in info["keywords"]):
            return collection
    return None


def initialize_or_load_storage():
    global vector_store, text_chunks
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(TEXT_CHUNKS_PATH):
        vector_store = faiss.read_index(VECTOR_STORE_PATH)
        with open(TEXT_CHUNKS_PATH, "rb") as f:
            text_chunks = pickle.load(f)
        console.print(
            f"Loaded existing vector store with {vector_store.ntotal} vectors and {len(text_chunks)} text chunks",
            style="italic green",
        )
    else:
        vector_store = faiss.IndexFlatL2(
            384
        )  # 384 is the dimension of the chosen embedding model
        text_chunks = []
        console.print(
            "Initialized new vector store and text chunks", style="italic yellow"
        )

    load_file_cache()  # Load the file cache during initialization


def save_storage():
    faiss.write_index(vector_store, VECTOR_STORE_PATH)
    with open(TEXT_CHUNKS_PATH, "wb") as f:
        pickle.dump(text_chunks, f)
    console.print(
        f"Saved vector store with {vector_store.ntotal} vectors and {len(text_chunks)} text chunks",
        style="italic green",
    )

    save_file_cache()  # Save the file cache


def update_user_profile():
    if os.path.exists(USER_PROFILE_PATH):
        with open(USER_PROFILE_PATH, "r") as file:
            profile_content = file.read()

        # Remove existing profile embeddings
        global text_chunks
        text_chunks = [
            chunk for chunk in text_chunks if chunk[1].get("type") != "user_profile"
        ]

        # Add updated profile embeddings
        add_to_vector_store(
            profile_content, {"type": "user_profile", "source": USER_PROFILE_PATH}
        )
        console.print("User profile updated", style="italic green")
    else:
        console.print(
            f"User profile not found at {USER_PROFILE_PATH}", style="italic yellow"
        )


def check_lm_studio():
    try:
        openai.Model.list()
        return True
    except Exception:
        return False


def load_file_cache():
    global cache
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    console.print(f"Loaded file cache with {len(cache)} entries", style="italic blue")


def save_file_cache():
    global cache
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    console.print(f"Saved file cache with {len(cache)} entries", style="italic green")


def scan_collections():
    console.print("Starting to scan collections...", style="bold yellow")
    global cache
    load_file_cache()  # Load the cache before scanning
    for collection, info in COLLECTIONS.items():
        console.print(f"Scanning collection: {collection}", style="italic blue")
        scan_collection(
            collection, os.path.expanduser(info["path"]), info.get("date_pattern")
        )
    save_file_cache()  # Save the updated cache after scanning
    console.print("Finished scanning collections.", style="bold green")


def scan_collection(collection_name, directory, date_pattern=None):
    console.print(f"Scanning directory: {directory}", style="italic blue")
    files_processed = 0
    files_skipped = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".txt", ".md")):
                file_path = os.path.join(root, file)
                if file_needs_processing(file_path):
                    date = None
                    if date_pattern:
                        date_match = re.search(date_pattern, file)
                        if date_match:
                            date = datetime.strptime(
                                date_match.group(1), "%Y.%m.%d"
                            ).date()
                    result = add_document(
                        file_path, collection=collection_name, date=date
                    )
                    console.print(result, style="italic green")
                    cache[file_path] = os.path.getmtime(file_path)
                    files_processed += 1
                else:
                    files_skipped += 1

    console.print(
        f"Collection {collection_name}: Processed {files_processed} files, Skipped {files_skipped} unchanged files",
        style="bold blue",
    )


def file_needs_processing(file_path):
    global cache
    if file_path not in cache:
        return True
    return os.path.getmtime(file_path) > cache[file_path]


def remember(user_input, system_response):
    conversation_chunk = f"User: {user_input}\nSystem: {system_response}"
    add_to_vector_store(
        conversation_chunk, {"type": "conversation", "date": datetime.now().date()}
    )


def respond(user_query):
    queried_collection = get_queried_collection(user_query)
    is_latest_query = "!latest" in user_query.lower()

    if is_latest_query and queried_collection:
        latest_entry = get_latest_entry(queried_collection)
        if latest_entry:
            context = f"Latest {queried_collection} entry (date: {latest_entry[1]['date']}):\n{latest_entry[0]}"
        else:
            context = f"No entries found for {queried_collection}."
    else:
        relevant_chunks = search_vector_store(
            user_query, collection=queried_collection, k=5
        )
        context = (
            "\n".join(chunk for chunk, _ in relevant_chunks)
            if relevant_chunks
            else "No relevant context found."
        )

    messages = [
        {
            "role": "system",
            "content": "You are HAL, an AI assistant. Use the following context to inform your responses, but don't explicitly mention or quote it unless directly relevant to the user's query. Pay special attention to any user profile information provided.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nUser query: {user_query}"},
    ]

    try:
        with Status("[bold green]HAL is thinking...", spinner="dots") as status:
            response = openai.ChatCompletion.create(
                model="local-model",
                messages=messages,
            )
        return response.choices[0].message["content"]
    except Exception as e:
        console.print(f"Error: {str(e)}", style="bold red")
        return "I'm sorry, I encountered an error while processing your request. Please try again."


def add_document(file_path, collection=None, date=None):
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    with open(file_path, "r") as file:
        content = file.read()

    # Remove existing chunks for this file
    global text_chunks
    text_chunks = [
        chunk for chunk in text_chunks if chunk[1].get("source") != file_path
    ]

    # Split content into smaller chunks (e.g., paragraphs)
    chunks = content.split("\n\n")
    chunk_count = 0

    for chunk in chunks:
        if chunk.strip():  # Ignore empty chunks
            metadata = {
                "type": "document",
                "source": file_path,
                "collection": collection,
                "date": date or datetime.now().date(),
            }
            add_to_vector_store(chunk, metadata)
            chunk_count += 1

    return f"Added {chunk_count} chunks from {file_path} to the knowledge base"


def add_to_vector_store(text, metadata):
    embedding = embedding_model.encode([text])[0]
    vector_store.add(np.array([embedding]))
    text_chunks.append((text, metadata))
    console.print(
        f"Added chunk to vector store. Total chunks: {len(text_chunks)}",
        style="italic green",
    )


def search_vector_store(query, collection=None, k=5):
    if vector_store.ntotal == 0:
        return []  # Return an empty list if the vector store is empty

    query_vector = embedding_model.encode([query])[0]
    _, indices = vector_store.search(
        np.array([query_vector]), k * 2
    )  # Search for more results initially

    results = []
    for i in indices[0]:
        if i < len(text_chunks):
            chunk, metadata = text_chunks[i]
            if collection is None or metadata.get("collection") == collection:
                results.append((chunk, metadata))
                if len(results) == k:
                    break

    return results


def get_latest_entry(collection):
    entries = [
        chunk for chunk in text_chunks if chunk[1].get("collection") == collection
    ]
    if entries:
        return max(entries, key=lambda x: x[1].get("date"))
    return None


def multiline_input():
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def chat_loop():
    if not check_lm_studio():
        console.print(
            "Error: Unable to connect to LM Studio. Please make sure it's running and the API is accessible.",
            style="bold red",
        )
        return

    update_user_profile()  # Update profile at the start of the conversation

    console.print(
        Panel.fit("Welcome to HAL! Type 'exit' to end the conversation.", title="HAL")
    )
    console.print("Special commands:")
    console.print("  [bold]!copy[/bold] - Copy last system response to clipboard")
    console.print("  [bold]!paste[/bold] - Paste from clipboard as your input")
    console.print("  [bold]!multi[/bold] - Enter multiline input mode")
    console.print(
        "  [bold]!add [file_path][/bold] - Add a specific document to the knowledge base"
    )
    console.print(
        "  [bold]!scan[/bold] - Rescan all specified directories for new documents"
    )
    console.print("  [bold]!update_profile[/bold] - Manually update your user profile")

    last_response = ""

    while True:
        user_input = Prompt.ask("You").strip()

        if user_input.lower() == "exit":
            console.print("Goodbye!", style="bold red")
            save_storage()  # Save before exiting
            break
        elif user_input == "!copy":
            pyperclip.copy(last_response)
            console.print("Last response copied to clipboard", style="italic green")
            continue
        elif user_input == "!paste":
            user_input = pyperclip.paste()
            console.print(f"Pasted: {user_input}", style="italic blue")
        elif user_input == "!multi":
            console.print(
                "Enter your multiline message (press Enter twice to finish):",
                style="bold yellow",
            )
            user_input = multiline_input()
        elif user_input.startswith("!add "):
            file_path = user_input[5:].strip()
            result = add_document(file_path)
            console.print(result, style="bold magenta")
            save_storage()  # Save after adding a document
            continue
        elif user_input == "!scan":
            with Status(
                "[bold yellow]Scanning collections for new documents...", spinner="dots"
            ) as status:
                scan_collections()
            console.print("Scan complete!", style="bold green")
            save_storage()  # Save after scanning
            continue

        elif user_input == "!update_profile":
            update_user_profile()
            save_storage()  # Save after updating profile
            continue

        system_response = respond(user_input)
        last_response = system_response  # Store for !copy command

        console.print(Panel(system_response, title="HAL", border_style="green"))

        remember(user_input, system_response)
        save_storage()  # Save after each interaction


if __name__ == "__main__":
    initialize_or_load_storage()
    scan_collections()  # Initial scan of collections (will only process new or modified documents)
    console.print(
        f"After scanning, vector store has {vector_store.ntotal} vectors and {len(text_chunks)} text chunks",
        style="bold magenta",
    )
    save_storage()  # Save the updated vector store and file cache
    chat_loop()
