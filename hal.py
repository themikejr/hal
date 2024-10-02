import os
import json
import re
import pickle
import readline
import pyperclip
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import openai
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status

# Set environment variable to suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Initialize OpenAI API to use LM Studio
openai.api_base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
openai.api_key = "not-needed"  # LM Studio doesn't require an API key

# Initialize console
console = Console()

# Constants for paths
HOME_DIR = os.path.expanduser("~")
VECTOR_STORE_PATH = os.path.join(HOME_DIR, "hal_vector_store.faiss")
TEXT_CHUNKS_PATH = os.path.join(HOME_DIR, "hal_text_chunks.pkl")
USER_PROFILE_PATH = os.path.join(HOME_DIR, "hal_user_profile.txt")
CACHE_PATH = os.path.join(HOME_DIR, "hal_file_cache.json")

# Collections configuration
COLLECTIONS = {
    "journal": {
        "path": "~/Memorabilia/memorabilia/Journal",
        "date_pattern": r"(\d{4}\.\d{2}\.\d{2})",  # YYYY.MM.DD
        "keywords": ["my_journal"],
    },
    # Add more collections as needed
}


class HALAssistant:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = None
        self.text_chunks = []
        self.cache = {}
        self.last_response = ""
        self.initialize_or_load_storage()
        self.check_lm_studio()

    def initialize_or_load_storage(self):
        if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(TEXT_CHUNKS_PATH):
            self.vector_store = faiss.read_index(VECTOR_STORE_PATH)
            with open(TEXT_CHUNKS_PATH, "rb") as f:
                self.text_chunks = pickle.load(f)
            console.print(
                f"Loaded existing vector store with {self.vector_store.ntotal} vectors and {len(self.text_chunks)} text chunks",
                style="italic green",
            )
        else:
            self.vector_store = faiss.IndexFlatL2(384)  # Embedding dimension
            self.text_chunks = []
            console.print(
                "Initialized new vector store and text chunks", style="italic yellow"
            )
        self.load_file_cache()

    def save_storage(self):
        faiss.write_index(self.vector_store, VECTOR_STORE_PATH)
        with open(TEXT_CHUNKS_PATH, "wb") as f:
            pickle.dump(self.text_chunks, f)
        console.print(
            f"Saved vector store with {self.vector_store.ntotal} vectors and {len(self.text_chunks)} text chunks",
            style="italic green",
        )
        self.save_file_cache()

    def load_file_cache(self):
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        console.print(
            f"Loaded file cache with {len(self.cache)} entries", style="italic blue"
        )

    def save_file_cache(self):
        with open(CACHE_PATH, "w") as f:
            json.dump(self.cache, f)
        console.print(
            f"Saved file cache with {len(self.cache)} entries", style="italic green"
        )

    def check_lm_studio(self):
        try:
            openai.Model.list()
        except Exception:
            console.print(
                "Error: Unable to connect to LM Studio. Please make sure it's running and the API is accessible.",
                style="bold red",
            )
            exit(1)

    def update_user_profile(self):
        if os.path.exists(USER_PROFILE_PATH):
            with open(USER_PROFILE_PATH, "r") as file:
                profile_content = file.read()

            # Remove existing profile embeddings
            self.text_chunks = [
                chunk
                for chunk in self.text_chunks
                if chunk[1].get("type") != "user_profile"
            ]

            # Add updated profile embeddings
            self.add_to_vector_store(
                profile_content, {"type": "user_profile", "source": USER_PROFILE_PATH}
            )
            console.print("User profile updated", style="italic green")
        else:
            console.print(
                f"User profile not found at {USER_PROFILE_PATH}", style="italic yellow"
            )

    def scan_collections(self):
        console.print("Starting to scan collections...", style="bold yellow")
        self.load_file_cache()
        for collection, info in COLLECTIONS.items():
            console.print(f"Scanning collection: {collection}", style="italic blue")
            self.scan_collection(
                collection, os.path.expanduser(info["path"]), info.get("date_pattern")
            )
        self.save_file_cache()
        console.print("Finished scanning collections.", style="bold green")

    def scan_collection(self, collection_name, directory, date_pattern=None):
        console.print(f"Scanning directory: {directory}", style="italic blue")
        files_processed = 0
        files_skipped = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".txt", ".md")):
                    file_path = os.path.join(root, file)
                    if self.file_needs_processing(file_path):
                        date = self.extract_date_from_filename(file, date_pattern)
                        result = self.add_document(
                            file_path, collection=collection_name, date=date
                        )
                        console.print(result, style="italic green")
                        self.cache[file_path] = os.path.getmtime(file_path)
                        files_processed += 1
                    else:
                        files_skipped += 1

        console.print(
            f"Collection {collection_name}: Processed {files_processed} files, Skipped {files_skipped} unchanged files",
            style="bold blue",
        )

    def file_needs_processing(self, file_path):
        return (
            file_path not in self.cache
            or os.path.getmtime(file_path) > self.cache[file_path]
        )

    def extract_date_from_filename(self, filename, date_pattern):
        if date_pattern:
            match = re.search(date_pattern, filename)
            if match:
                return datetime.strptime(match.group(1), "%Y.%m.%d").date()
        return None

    def add_document(self, file_path, collection=None, date=None):
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        with open(file_path, "r") as file:
            content = file.read()

        # Remove existing chunks for this file
        self.text_chunks = [
            chunk for chunk in self.text_chunks if chunk[1].get("source") != file_path
        ]

        # Split content into smaller chunks (e.g., paragraphs)
        chunks = content.split("\n\n")
        chunk_count = 0

        for chunk in chunks:
            if chunk.strip():
                metadata = {
                    "type": "document",
                    "source": file_path,
                    "collection": collection,
                    "date": date or datetime.now().date(),
                }
                self.add_to_vector_store(chunk, metadata)
                chunk_count += 1

        return f"Added {chunk_count} chunks from {file_path} to the knowledge base"

    def add_to_vector_store(self, text, metadata):
        embedding = self.embedding_model.encode([text])[0]
        self.vector_store.add(np.array([embedding]))
        self.text_chunks.append((text, metadata))
        console.print(
            f"Added chunk to vector store. Total chunks: {len(self.text_chunks)}",
            style="italic green",
        )

    def search_vector_store(self, query, collection=None, k=5):
        if self.vector_store.ntotal == 0:
            return []

        query_vector = self.embedding_model.encode([query])[0]
        _, indices = self.vector_store.search(np.array([query_vector]), k * 2)

        results = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                chunk, metadata = self.text_chunks[idx]
                if collection is None or metadata.get("collection") == collection:
                    results.append((chunk, metadata))
                    if len(results) == k:
                        break
        return results

    def get_latest_entry(self, collection):
        entries = [
            chunk
            for chunk in self.text_chunks
            if chunk[1].get("collection") == collection
        ]
        if entries:
            return max(entries, key=lambda x: x[1].get("date"))
        return None

    def remember(self, user_input, system_response):
        conversation_chunk = f"User: {user_input}\nSystem: {system_response}"
        self.add_to_vector_store(
            conversation_chunk, {"type": "conversation", "date": datetime.now().date()}
        )

    def respond(self, user_query):
        queried_collection = self.get_queried_collection(user_query)
        is_latest_query = "!latest" in user_query.lower()

        if is_latest_query and queried_collection:
            latest_entry = self.get_latest_entry(queried_collection)
            if latest_entry:
                context = f"(1) {latest_entry[0]}"
                sources = [latest_entry[1]]
            else:
                context = "No entries found."
                sources = []
        else:
            relevant_chunks = self.search_vector_store(
                user_query, collection=queried_collection, k=5
            )
            if relevant_chunks:
                context = "\n".join(
                    f"({i+1}) {chunk}" for i, (chunk, _) in enumerate(relevant_chunks)
                )
                sources = [metadata for _, metadata in relevant_chunks]
            else:
                context = "No relevant context found."
                sources = []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are HAL, an AI assistant. Use the following context to inform your responses. "
                    "When you use information from the context, cite the source by referring to the reference number provided."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nUser query: {user_query}",
            },
        ]

        try:
            with Status("[bold green]HAL is thinking...", spinner="dots"):
                response = openai.ChatCompletion.create(
                    model="local-model",
                    messages=messages,
                )
            assistant_response = response.choices[0].message["content"]

            # Append source references at the end
            if sources:
                source_references = "\n\nSources:\n" + "\n".join(
                    f"({i+1}) [{metadata.get('collection', 'Unknown')}] "
                    f"{os.path.basename(metadata.get('source', 'Unknown'))} "
                    f"(Date: {metadata.get('date', 'Unknown')})"
                    for i, metadata in enumerate(sources)
                )
                assistant_response += source_references

            return assistant_response
        except Exception as e:
            console.print(f"Error: {str(e)}", style="bold red")
            return "I'm sorry, I encountered an error while processing your request. Please try again."

    def get_queried_collection(self, user_query):
        lower_query = user_query.lower()
        for collection, info in COLLECTIONS.items():
            if any(keyword in lower_query for keyword in info["keywords"]):
                return collection
        return None

    def multiline_input(self):
        console.print(
            "Enter your multiline message (press Enter twice to finish):",
            style="bold yellow",
        )
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

    def chat_loop(self):
        self.update_user_profile()
        console.print(
            Panel.fit(
                "Welcome to HAL! Type 'exit' to end the conversation.", title="HAL"
            )
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
        console.print(
            "  [bold]!update_profile[/bold] - Manually update your user profile"
        )

        while True:
            user_input = Prompt.ask("You").strip()

            if user_input.lower() == "exit":
                console.print("Goodbye!", style="bold red")
                self.save_storage()
                break
            elif user_input == "!copy":
                pyperclip.copy(self.last_response)
                console.print("Last response copied to clipboard", style="italic green")
            elif user_input == "!paste":
                user_input = pyperclip.paste()
                console.print(f"Pasted: {user_input}", style="italic blue")
            elif user_input == "!multi":
                user_input = self.multiline_input()
            elif user_input.startswith("!add "):
                file_path = user_input[5:].strip()
                result = self.add_document(file_path)
                console.print(result, style="bold magenta")
                self.save_storage()
            elif user_input == "!scan":
                with Status(
                    "[bold yellow]Scanning collections for new documents...",
                    spinner="dots",
                ):
                    self.scan_collections()
                console.print("Scan complete!", style="bold green")
                self.save_storage()
            elif user_input == "!update_profile":
                self.update_user_profile()
                self.save_storage()
            else:
                system_response = self.respond(user_input)
                self.last_response = system_response
                console.print(Panel(system_response, title="HAL", border_style="green"))
                self.remember(user_input, system_response)
                self.save_storage()


def main():
    assistant = HALAssistant()
    assistant.scan_collections()
    console.print(
        f"After scanning, vector store has {assistant.vector_store.ntotal} vectors and {len(assistant.text_chunks)} text chunks",
        style="bold magenta",
    )
    assistant.save_storage()
    assistant.chat_loop()


if __name__ == "__main__":
    main()
