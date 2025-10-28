# scholar_alert.py
# import requests
from pathlib import Path
from bs4 import BeautifulSoup
import json
import hashlib
from typing import Dict
import json5
import fitz
from openai import AsyncOpenAI
from openai._exceptions import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)
from typing import Deque, Optional
from router_utlis.get_free import get_free_models
from dotenv import load_dotenv
import os
from io import BytesIO
import pdb
from  datetime import datetime
import html
import re
import asyncio
from html import unescape
from textwrap import dedent
from typing import Tuple
from collections import deque
# import nest_asyncio
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    RetryCallState
)
from asyncio_throttle import Throttler
import aiohttp
import logging
import feedparser

"""
Assumption:
15 pages in 2-column format <= 16000 words
16000 words  * 4/3 <= 20000 tokens
-> 4 chunks, 5000 tokens per chunk
"""
# nest_asyncio.apply()

load_dotenv()

# define retryable exceptions
retryable_exceptions = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError
)

folder_name = ".data"
folder_path = Path(f"{folder_name}")

# Define a function to log retry errors
def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"OpenAI Error: {exception} Retrying: {retry_state.attempt_number} time(s)..."
    )
    

throttler_query = Throttler(rate_limit=3, period=15)
throttler_summary = Throttler(rate_limit=15, period=60) # 20 qpm
throttler_send = Throttler(rate_limit=1, period=1)


HEADERS = {"User-Agent": "Mozilla/5.0"}
LAST_WORKING_MODEL = None
THROUGHPUT = 20000
MAX_PAGE = 15
CHUNK_CHAR_LIMIT = 4000 
MAX_CHUNKS = 6
LONG_MEMORY_LIMIT = 5
DAILY_SUMMARY_LIMIT = 15
MAX_TOKENS = 500
UNPROCESSED_PATH = folder_path / "unprocessed_papers.json"



FREE_MODELS = get_free_models() # provider : [model1, model2,...]

papers_seen = set()

with open(folder_path/"known_ids.json", "r") as f:
    histories = json.load(f)
    for each in histories:
        papers_seen.add(each[1])

    
START_IDX = 0
MAX_IDS = 500


client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_KEY"),
    base_url=os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
)


def extract_json(payload: str) -> Dict:
    """Extract JSON from the model output, tolerating stray text if possible."""

    pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
    matches = re.findall(pattern, payload, flags=re.MULTILINE)
    if matches:
        payload = matches[-1]

    start = payload.find("{")
    end = payload.rfind("}") + 1
    if end == 0 and start != -1:
        payload = payload + "}"
        end = len(payload)
    jsonStr = payload[start:end] if start != -1 and end != 0 else ""
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)
    try:
        return json5.loads(jsonStr)
    except (json.JSONDecodeError, ValueError):
        error_str = "LLM outputted an invalid JSON. Please use a better structured model."
        print(error_str)
        print(payload)
        raise  
    except Exception as e:
        print(payload)
        raise Exception(f"An unexpected error occurred: {str(e)}")

def parse_llm_response(resp) -> str:
    """
    適配新版 openai.ChatCompletion 回傳物件（Pydantic 模型）
    """
    if not resp or not resp.choices:
        return "[Invalid response structure]"
    # pdb.set_trace()
    choice = resp.choices[0]

    # 優先: 標準 OpenAI 格式
    if choice.message and choice.message.content:
        content = choice.message.content.strip()
        if content:
            return content

    # 非標準: Chutes 等提供者可能用 reasoning / reasoning_details
    msg = choice.message

    if hasattr(msg, "reasoning") and msg.reasoning and msg.reasoning.strip():
        return msg.reasoning.strip()

    if hasattr(msg, "reasoning_details") and isinstance(msg.reasoning_details, list):
        for reason in msg.reasoning_details:
            if reason.get("text", "").strip():
                return reason["text"].strip()

    # 備用: 某些模型還是會用 text 屬性（通常在非 chat-completion）
    if hasattr(choice, "text") and choice.text.strip():
        return choice.text.strip()

    return "[No usable content in response]"


def locate_references_section(text: str) -> int:
    """
    Locate the position of the references / bibliography heading if present.
    Returns -1 when no heading is found.
    """
    normalized = text.replace("\r\n", "\n")
    patterns = [
        r"^\s*(?:\d+(\.\d+)*)?\s*references?\s*[:\-]?\s*$",
        r"^\s*(?:\d+(\.\d+)*)?\s*bibliography\s*[:\-]?\s*$",
        r"^\s*(?:\d+(\.\d+)*)?\s*reference\s+list\s*[:\-]?\s*$",
    ]

    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        match = regex.search(normalized)
        if match:
            return match.start()
    return -1


def trim_before_references(text: str) -> str:
    """
    Truncate the paper content once the References section is reached.
    If no references heading is detected, the original text is returned.
    """
    normalized = text.replace("\r\n", "\n")
    ref_idx = locate_references_section(normalized)
    if ref_idx >= 0:
        return normalized[:ref_idx]
    return normalized


def normalize_heading(heading: str) -> str:
    cleaned = re.sub(r"\s+", " ", heading).strip()
    cleaned = re.sub(r"^\d+(\.\d+)*\s*", "", cleaned)
    cleaned = cleaned.rstrip(":")
    return cleaned.title() if cleaned.isupper() else cleaned


def extract_structured_chunks(
    text: str,
    chunk_size: int = CHUNK_CHAR_LIMIT,
    max_chunks: int = MAX_CHUNKS,
) -> list[dict[str, str]]:
    """
    Build structured chunks grouped by section headings, bounded by References.
    """
    if not text:
        return []

    truncated = trim_before_references(text)
    paragraphs = re.split(r"\n{2,}", truncated)
    current_paras: list[str] = []

    for para in paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue

        current_paras.append(cleaned)


    structured_chunks: list[dict[str, str]] = []

    for paras in current_paras:
        part_idx = 1
        buffer: list[str] = []
        buffer_len = 0

        for para in paras:
            candidate_len = buffer_len + len(para) + (2 if buffer else 0)
            if candidate_len <= chunk_size:
                buffer.append(para)
                buffer_len = candidate_len
            else:
                if buffer:
                    structured_chunks.append(
                        {"content": "\n\n".join(buffer)}
                    )
                    if len(structured_chunks) >= max_chunks:
                        return structured_chunks[:max_chunks]
                    part_idx += 1
                buffer = [para]
                buffer_len = len(para)

        if buffer:
            structured_chunks.append({"content": "\n\n".join(buffer)})
            if len(structured_chunks) >= max_chunks:
                return structured_chunks[:max_chunks]

    return structured_chunks[:max_chunks]


def build_chunk_prompt(
    questions: str = '',
    previous_keynote: Optional[Deque[str]] = None,
    chunk: str = '',
) -> str:
    
    previous_keynote = "\n\n".join(previous_keynote) if previous_keynote else "None"
    return dedent(
       f"""
You are an expert research analyst extracting insights to answer the questions: 
"{questions}".

You are given:
1. The **keynote from the previous chunk**, summarizing what was discussed earlier.
2. The **current chunk** of the paper.

Your task:
- Use the previous keynote as contextual memory to maintain continuity and avoid redundancy.
- Extract from the current chunk **only the new or expanded information** that helps answer the research query.
- Focus on **technical details**, **methodology**, **datasets**, **experimental setups**, **quantitative findings**, and **limitations**.
- If the chunk mainly repeats information from the previous keynote, summarize only the *differences or elaborations*.
- Be concise but technically precise — prefer numeric or mechanistic descriptions over generic phrases.
- If the chunk contains no new relevant insight, simply set is_relevants to false.

Do not use nested layer in notes. Use plain text description.
Output JSON Format Only.

Example Output:
{{
    'is_relevant' : <true or false>,
    'notes': <a brief summary of this chunk>
}}

User Input:

Previous Keynote:
{previous_keynote}

Current Chunk:
{chunk}
""")


def build_refine_prompt( memory_snippets: list[str]) -> str:
    formatted_snippets = "\n\n".join(
        f"Snippet {i + 1}:\n{snippet}" for i, snippet in enumerate(memory_snippets)
    )
    return (
        f"Using the collected insights below, synthesize a hierarchical summary that answers follow the rules below. "
        "Focus on clarity and avoid redundant wording. Provide the output in the following structure:\n"
        "1. Identified Problems\n"
        "2. Solutions to the Problems\n"
        "3. Results and Findings\n"
        "4. Innovations and Contributions\n"
        "5. Philosophical or Physical Implications\n\n"
        
        "Each point should have no more than two sub-items."
        "Each point should have no more than 50 words."
        f"Collected insights:\n{formatted_snippets}"
    )


def toknow():
    prompt = """
    "1. Identified Problems\n"
    "2. Solutions to the Problems\n"
    "3. Results and Findings\n"
    "4. Innovations and Contributions\n"
    "5. Philosophical or Physical Implications\n\n"
    """
    return dedent(prompt)

@retry(
    wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
    retry=retry_if_exception_type(retryable_exceptions),
    after=log_retry_error,
    reraise=True,
)
async def generate_completion(prompt: str, max_tokens: int, throttler: Throttler, is_json = False) -> str:
    """
    Use the last known working model; if it fails, fallback to next available.
    Retry transient errors with exponential backoff (handled by tenacity).
    """
    global START_IDX, LAST_WORKING_MODEL

    providers = list(FREE_MODELS.keys())
    n_providers = len(providers)

    
    # Try cached working model first
    candidate_models = []
    if LAST_WORKING_MODEL:
        candidate_models.append(LAST_WORKING_MODEL)
    # Then all others in deterministic round-robin order
    for i in range(n_providers):
        provider = providers[(START_IDX + i) % n_providers]
        for m in FREE_MODELS[provider]:
            candidate_models.append(m['id'])

    for model_id in candidate_models:
        retry_for_each = 3
        while retry_for_each > 0:
            try:
                async with throttler:
                    resp = await client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                    )
            
                LAST_WORKING_MODEL = model_id
                START_IDX = (START_IDX + 1) % n_providers
                chunk_summary = parse_llm_response(resp)
                if is_json:
                    return extract_json(chunk_summary)
                else:
                    return chunk_summary

            except Exception as e:
                print(f"[WARN] {model_id} failed: {e}")
                retry_for_each -= 1
                await asyncio.sleep(1)  # small cooldown before next try
                continue
    return ""
    
@retry(
wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
after=log_retry_error,
)
async def download_and_extract_pdf_text(pdf_url:str):
    """Download a PDF file and extract its text content.

    Args:
        pdf_url (str): The URL of the PDF file to download.

    Returns:
        str: The extracted text content from the PDF or an error message.
    """
  

    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(pdf_url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    print("Failed to download PDF:", response.status)
                    return None
                else:
                    print("PDF downloaded successfully.")
                    content = await response.read()  # Ensure the response is fully read
                    pdf_file = BytesIO(content)
                    doc = fitz.open(stream=pdf_file, filetype="pdf")
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    return full_text.strip()
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            
        return ""


async def extract_google_scholar_papers(throttler=throttler_query, query: str="llm security")-> dict:
    """Extract papers from Google Scholar based on a query.

    Args:
        throttler (Throttler, optional): Throttler instance to limit request rate. Defaults to throttler_query.
        query (str, optional): Search query for Google Scholar. Defaults to llm security.

    Returns:
        dict: A dictionary containing extracted paper information.
    """
    global papers_seen, histories

    url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&scisbd=1"
    async with throttler:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=HEADERS) as resp:
                if resp.status != 200:
                    print(f"Failed to fetch papers: {resp.status}")
                    return {}
                
                soup = BeautifulSoup(await resp.text(), "html.parser")
                keynotes = {}
                results = soup.select(".gs_r")[:MAX_PAGE]

                for r in results:
                    # Title and main link
                    curr = {}
                    title_tag = r.select_one(".gs_rt a")
                    if title_tag:
                        title = title_tag.text.strip()
                    else:
                        continue
                    
                    # main_link = title_tag["href"] if title_tag else None
                    curr["title"] = title
                    p_id = hashlib.md5(curr["title"].encode("utf-8")).hexdigest()
                    
                    if p_id in papers_seen:
                        continue
                    # pdb.set_trace()
                    
                    papers_seen.add(p_id)
                    histories.append((datetime.now().strftime('%Y-%m-%d, %H:%M'), p_id))
                    # Abstract snippet
                    snippet_tag = r.select_one(".gs_rs")
                    snippet = snippet_tag.text.strip() if snippet_tag else "N/A"
                    curr["abstract"] = snippet
                    
                    # PDF link if available (右側 PDF 下載連結)
                    pdf_tag = r.select_one(".gs_or_ggsm a")
                    pdf_link = pdf_tag["href"] if pdf_tag else None
                    curr["pdf_link"] = pdf_link if pdf_link else "N/A"
                    curr["query"] = query
                    print(curr)
                    print("-" * 80)
                    keynotes[p_id] = curr
                    if len(keynotes)>= 2:
                        break
                    
            await asyncio.sleep(1)
    return keynotes


async def extract_arxiv_papers(throttler=throttler_query, query: str="llm", max_results: int=10) -> dict:
    """Extract papers from arXiv based on a query.

    Args:
        throttler (Throttler): Throttler instance to limit request rate.
        query (str): Search keyword.
        max_results (int): Number of results to return.

    Returns:
        dict: Dictionary of paper information.
    """
    global histories, papers_seen

    base_url = "http://export.arxiv.org/api/query?"
    query_params = (
        f"search_query=all:{query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    url = base_url + query_params

    keynotes = {}
    print("Collectiing.....")
    async with throttler:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"Failed to fetch arXiv papers: {resp.status}")
                    return {}

                text = await resp.text()
                feed = feedparser.parse(text)
                # pdb.set_trace()
                for entry in feed.entries:
                    title = re.sub(r'\s+', ' ', entry.title.strip())

                    print("title =", title)
                    p_id = hashlib.md5(title.encode("utf-8")).hexdigest()
                    
                    if p_id in papers_seen:
                        print(f"[Info] {title} has been seen.")
                        continue
                    print(f"[Update] '{title}' added to watchlist.")
                    papers_seen.add(p_id)
                    histories.append((datetime.now().strftime('%Y-%m-%d, %H:%M'), p_id))

                    summary = entry.summary.strip()
                    pdf_link = ""
                    for link in entry.links:
                        if link.rel == "alternate" and link.type == "application/pdf":
                            pdf_link = link.href
                            break
                    
                    # fallback to entry.id as PDF (works for most arXiv papers)
                    if not pdf_link:
                        pdf_link = entry.id.replace("abs", "pdf") + ".pdf"

                    keynotes[p_id] = {
                        "title": title,
                        "summary": summary,
                        "pdf_link": pdf_link,
                        "query": query
                    }

                    print(keynotes[p_id])
                    print("-" * 80)

                await asyncio.sleep(1)

    return keynotes


async def load_and_summarize(pdf_link: str, max_tokens: int = 300) -> Tuple[str, list[str]]:
    """Load a PDF from a link and summarize its content through Llms' apis.

    Args:
        pdf_link (str): The URL of the PDF file to load.
        query (str): The guiding question used to focus the summary.
        max_tokens (int, optional): The maximum number of tokens for the summary. Defaults to 250.

    Returns:
        Tuple[str, list[str]]: The refined summary and the long-memory snippets used to produce it.
    """

    if pdf_link != "N/A" and pdf_link is not None:
        print(f"PDF link: {pdf_link}")

        pdf_text = await download_and_extract_pdf_text(pdf_link)
        if not pdf_text:
            return "", []

        scoped_text = pdf_text[:THROUGHPUT]
        text_chunks = extract_structured_chunks(scoped_text, CHUNK_CHAR_LIMIT, MAX_CHUNKS)
        if not text_chunks:
            return "", []

        memory_queue: deque[str] = deque(maxlen=LONG_MEMORY_LIMIT)
       
   
        for idx, chunk_info in enumerate(text_chunks):
            prompt = build_chunk_prompt(
                toknow(),
                memory_queue,
                chunk_info["content"]
            )
            try:
                json_keynotes = await generate_completion(prompt, max_tokens=300, throttler=throttler_summary, is_json=True)
                
            except Exception as e:
                print(f"Error summarizing chunk {idx + 1}: {e}")
                continue

            if json_keynotes["is_relevant"]:
                snippet = f"short summary:\n{json_keynotes['notes']}"
                memory_queue.appendleft(snippet)

        if not memory_queue:
            return "", []

        refine_prompt = build_refine_prompt(list(memory_queue))
        try:
            refined_summary = await generate_completion(refine_prompt, max_tokens=max_tokens, throttler=throttler_summary)
            print("[Info] Refined summary length:", len(refined_summary))
        except Exception as e:
            print(f"Error refining summary: {e}")
            refined_summary = "\n".join(list(memory_queue))

        return refined_summary, list(memory_queue)
    else:
        print("PDF link: Not available")
        return "", []
    

def load_unprocessed(path=UNPROCESSED_PATH) -> list:
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    return []


def save_unprocessed(entries: list, path=UNPROCESSED_PATH):
    with open(path, "w") as f:
        json.dump(entries, f, indent=4)

async def send_telegram(throttler, msg:str):
    """Send a message to a Telegram chat.

    Args:
        throttler (Throttler): Throttler instance to limit request rate.
        msg (str): The message content to send.
    """
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with throttler:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data={"chat_id": chat_id, "text": msg, "parse_mode": "MarkdownV2"}, timeout=10) as resp:
                if resp.status != 200:
                    print("Failed to send message:", resp.status, await resp.text())
                else:
                    print("Message sent successfully.")
                    await resp.text()
        

def escape_markdown_v2(text: str) -> str:
    """Escape special characters in text for Markdown V2 formatting."""
    text = html.unescape(text)  # 先還原 HTML 實體
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


def clean_telegram_text(text: str) -> str:
    """Clean and format text for Telegram messages. Currently, it escapes Markdown V2 special characters."""
    return escape_markdown_v2(text)

def format_telegram_message(paper:dict) -> str:
    """Format a paper's information into a Telegram message."""
    
    title = clean_telegram_text(paper.get("title", ""))
    summary = clean_telegram_text(paper.get("summary", ""))
    pdf_link = paper.get("pdf_link", "")
    summary = paper.get("summary")

    msg = f"📄 *Title:* {title}\n"
    msg += f"🔗 [PDF Link]({pdf_link})\n"

    if summary:
        msg += f"🧠 *Summary:* \n {clean_telegram_text(summary)}\n"

    return msg


async def main(queries=["ai agent"], path=folder_path/"known_ids.json"):
    global START_IDX

    START_IDX = 0
    tasks = []
    for query in queries:
        tasks.append(extract_arxiv_papers(throttler_query, query))
    parses = await asyncio.gather(*tasks)
    
    new_papers = []
    for query, parse in zip(queries, parses):
        for p_id, info in parse.items():
            if "query" not in info:
                info["query"] = query
            new_papers.append((p_id, info))
    
    unprocessed_entries = load_unprocessed()
    backlog_queue = []
    for entry in unprocessed_entries:
        if isinstance(entry, dict) and "id" in entry and "paper" in entry:
            backlog_queue.append((entry["id"], entry["paper"]))

    new_pdf_candidates = []
    new_non_pdf = []
    for item in new_papers:
        _, paper = item
        if paper.get("pdf_link") and paper.get("pdf_link") != "N/A":
            new_pdf_candidates.append(item)
        else:
            paper.setdefault("summary", "")
            paper.setdefault("memory_snippets", [])
            new_non_pdf.append(item)

    available_slots = DAILY_SUMMARY_LIMIT
    papers_to_process = []
    remaining_new_pdf = []
    for item in new_pdf_candidates:
        if available_slots > 0:
            papers_to_process.append(item)
            available_slots -= 1
        else:
            remaining_new_pdf.append(item)

    backlog_to_process = []
    backlog_remaining = []
    for entry in backlog_queue:
        if available_slots > 0:
            backlog_to_process.append(entry)
            available_slots -= 1
        else:
            backlog_remaining.append(entry)

    processing_targets = papers_to_process + backlog_to_process

    finalized_papers = {}
    finalized_order = []
    for p_id, paper in new_non_pdf:
        finalized_papers[p_id] = paper
        finalized_order.append(p_id)

    summary_targets = []
    for p_id, paper in processing_targets:
        if paper.get("pdf_link") and paper.get("pdf_link") != "N/A":
            summary_targets.append((p_id, paper))
        else:
            paper.setdefault("summary", "")
            paper.setdefault("memory_snippets", [])
            finalized_papers[p_id] = paper
            finalized_order.append(p_id)

    summary_results = []
    if summary_targets:
        summary_tasks = [
            load_and_summarize(
                paper["pdf_link"],
                max_tokens=MAX_TOKENS
            )
            for _, paper in summary_targets
        ]
        summary_results = await asyncio.gather(*summary_tasks, return_exceptions=True)

    failed_targets = []
    for (p_id, paper), result in zip(summary_targets, summary_results):
        if isinstance(result, Exception):
            print(f"Summary failed for paper {paper.get('title', 'N/A')}: {result}")
            failed_targets.append((p_id, paper))
            continue
        summary, _ = result
        paper["summary"] = summary
        finalized_papers[p_id] = paper
        finalized_order.append(p_id)

    pending_backlog_entries = backlog_remaining + remaining_new_pdf + failed_targets
    print("[Info] Saving unprocessed papers to backlog.")
    pending_backlog_entries = pending_backlog_entries[-MAX_IDS:]
    save_unprocessed([{"id": pid, "paper": paper} for pid, paper in pending_backlog_entries])

    known_ids = histories[-MAX_IDS:]
    with open(path, "w") as f:
        json.dump(known_ids, f, indent=4)

    send_tasks = []
    for p_id in finalized_order:
        paper = finalized_papers[p_id]
        formatted_msg = format_telegram_message(paper)
        send_tasks.append(send_telegram(throttler_send, formatted_msg))
    if send_tasks:
        await asyncio.gather(*send_tasks)
    

if __name__ == "__main__":
    queries = ['ai agent', 'fog ai']
    asyncio.run(main(queries))
