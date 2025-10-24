# Daily Paper Automation

This project automates the process of extracting, summarizing, and sending research papers from sources like Google Scholar and arXiv. It uses `asyncio` for concurrency, `aiohttp` for asynchronous HTTP requests, and OpenAI APIs for summarization.

## Features

- **Google Scholar and arXiv Integration**: Extracts research papers based on specified queries.
- **PDF Text Extraction**: Downloads and extracts text from PDF files.
- **Summarization**: Summarizes extracted text using OpenAI's API.
- **Telegram Notifications**: Sends formatted paper summaries to a Telegram chat.
- **Rate Limiting**: Uses `asyncio-throttle` to control request rates.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/daily-paper.git
   cd daily-paper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   tg_bot_token=YOUR_TELEGRAM_BOT_TOKEN
   chat_id=YOUR_TELEGRAM_CHAT_ID
   openrouter_key=YOUR_OPENAI_API_KEY
   api_base_url=YOUR_OPENAI_API_BASE_URL (optional)
   ```

## Framework

```mermaid

flowchart TD

%% ====== DATA SOURCES ======
A1["(arXiv API)"]
A2["(Google Scholar HTML)"]

subgraph INPUT["Data Acquisition Layer"]
    A1 --> B1["extract_arxiv_papers()"]
    A2 --> B2["extract_google_scholar_papers()"]
    B1 & B2 --> C1["Paper Queue (new + backlog)"]
end

%% ====== MODEL MANAGEMENT ======
subgraph MODELS["Model Selection Layer"]
    M1["get_free_models()"]
    M2["score_model_for_summary(description)"]
    M1 --> M2
    M2 --> M3["Rank + Filter free models"]
    M3 --> M4["Round-robin fallback in generate_completion()"]
end

%% ====== SUMMARIZATION PIPELINE ======
subgraph PIPELINE["Summarization Pipeline"]
    P1["download_and_extract_pdf_text()"]
    P2["extract_structured_chunks()"]
    P3["build_chunk_prompt()"]
    P4["generate_completion()"]
    P5["parse_llm_response() + extract_json()"]
    P6["Refinement: build_refine_prompt()"]
    P7["generate_completion() (final summary)"]
    
    P1 --> P2 --> P3 --> P4 --> P5
    P5 -->|"append relevant notes"| P3
    P5 -->|"document finished"| P6 --> P7
end

%% ====== MEMORY LOOP ======
subgraph MEMORY["Context Memory System"]
    L1["Short Memory (current chunk notes)"]
    L2["Long Memory (deque of recent snippets)"]
    L1 --> L2
    L2 --> P6
end

%% ====== OUTPUT ======
subgraph OUTPUT["Delivery Layer"]
    O1["format_telegram_message()"]
    O2["send_telegram()"]
    O3["save_unprocessed() + update known_ids.json"]
end

%% ====== CONTROL + RESILIENCE ======
subgraph CTRL["Control Layer"]
    C2["asyncio_throttle: rate control"]
    C3["tenacity: retry + exponential backoff"]
    C4["AsyncOpenAI Client"]
end
P4 -.-> CTRL
MODELS --> P4

%% ====== CONNECTIONS ======
C1 -->|PDF link| P1
P7 --> O1 --> O2
P7 --> O3
```


## Usage

1. Run the script:
   ```bash
   python scraping.py
   ```

2. The script will:
   - Fetch papers from Google Scholar or arXiv based on predefined queries.
   - Summarize the papers using OpenAI's API.
   - Send the summaries to a Telegram chat.

## Configuration

- **Queries**: Modify the `queries` list in the `main` function to customize search terms.
- **Rate Limits**: Adjust the `Throttler` settings for query, summary, and Telegram message sending rates.
- **Maximum Papers**: Change the `MAX_PAGE` and `MAX_IDS` constants to control the number of papers processed.

## File Structure

```
daily_paper/
├── scraping.py         # Main script for automation
├── .env                # Environment variables (not included in the repo)
├── .data/              # Folder for storing cached data
└── requirements.txt    # Python dependencies
```

## Dependencies

- `aiohttp`: For asynchronous HTTP requests.
- `asyncio-throttle`: For rate limiting.
- `tenacity`: For retrying failed API calls.
- `feedparser`: For parsing RSS feeds (arXiv).
- `python-dotenv`: For loading environment variables.
- `beautifulsoup4`: For parsing HTML (Google Scholar).
- `PyMuPDF`: For extracting text from PDFs.

## Notes

- Model selection uses:
• description keyword scoring
• free-only filter
• highest context length first
end

- Maintains continuity:
short-term chunk → deque long-term
→ contextual refinement summary

- Ensure that your OpenAI API key has sufficient quota for summarization tasks.

- The script uses a retry mechanism for handling API rate limits and timeouts.


end

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.