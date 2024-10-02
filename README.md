# HAL (Highly Adaptive Language) Assistant

HAL is a personal AI assistant that uses Retrieval-Augmented Generation (RAG) to provide context-aware responses. It integrates with local document collections and maintains a conversation history to offer personalized assistance.

## Features

- **Local Document Integration**: Scan and index local document collections for context-aware responses.
- **Conversation Memory**: Remember and learn from past interactions.
- **User Profiling**: Maintain and update a user profile for personalized assistance.
- **Multiple Collections**: Support for multiple named document collections (e.g., journal, documents).
- **Efficient Document Processing**: Use a caching mechanism to avoid reprocessing unchanged documents.
- **Local Language Model Integration**: Utilize LM Studio for local language model inference.
- **Rich Command-Line Interface**: Colorful and interactive CLI using the Rich library.

## Prerequisites

- Python 3.7+
- Poetry (for dependency management)
- LM Studio (for local language model inference)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/themikejr/hal.git
   cd hal-assistant
   ```

2. Install dependencies using Poetry:

   ```
   poetry install
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add:

   ```
   OPENAI_API_BASE=http://localhost:1234/v1
   ```

4. Configure your document collections:
   Edit the `COLLECTIONS` dictionary in `hal.py` to point to your local document directories.

5. Set up your user profile:
   Create a file at `~/hal_user_profile.txt` and add relevant information about yourself. See the "User Profile" section below for more details.

## Usage

Run HAL using Poetry:

```
poetry run python hal.py
```

Or set up an alias in your shell configuration file (~/.zshrc or ~/.bash_profile):

```
alias hal='(cd /path/to/hal-assistant && poetry run python hal.py)'
```

Then you can simply run:

```
hal
```

### Special Commands

- `!copy`: Copy last system response to clipboard
- `!paste`: Paste from clipboard as your input
- `!multi`: Enter multiline input mode
- `!add [file_path]`: Add a specific document to the knowledge base
- `!scan`: Rescan all specified directories for new documents
- `!update_profile`: Manually update your user profile

## User Profile

The user profile is a key feature of HAL that allows for personalized interactions. To set up your user profile:

1. Create a text file at `~/hal_user_profile.txt`.
2. Add relevant information about yourself to this file. This can include:
   - Your name, age, occupation
   - Your interests and hobbies
   - Your preferences (e.g., communication style, topics of interest)
   - Any other information you'd like HAL to know about you

Example user profile:

```
Name: Jane Doe
Age: 30
Occupation: Software Developer
Interests: Machine learning, hiking, science fiction
Preferences: Direct communication, deep technical discussions
Goals: Improve coding skills, learn about AI ethics
```

You can update your user profile at any time by editing the `~/hal_user_profile.txt` file and then using the `!update_profile` command in the HAL interface. This will re-index your profile information and incorporate it into future interactions.

## Configuration

- **Vector Store Path**: `~/hal_vector_store.faiss`
- **Text Chunks Path**: `~/hal_text_chunks.pkl`
- **User Profile Path**: `~/hal_user_profile.txt`
- **Cache Path**: `~/hal_file_cache.json`

Modify these paths in `hal.py` if needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your chosen license here]

## Acknowledgements

- OpenAI for the ChatGPT model
- LM Studio for local model inference
- Sentence Transformers for embeddings
- FAISS for vector storage and search
- Rich for the command-line interface
