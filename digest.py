# scholar_alert.py
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import json
import hashlib
from datetime import datetime
import fitz
from openai import AsyncOpenAI
from openai._exceptions import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)
from router_utlis.get_free import get_free_models
from dotenv import load_dotenv
import os
from io import BytesIO
import pdb
import time
import html
import re
import os
import asyncio
from html import unescape
from typing import Tuple
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
throttler_summary = Throttler(rate_limit=3, period=600)
throttler_send = Throttler(rate_limit=1, period=1)



HEADERS = {"User-Agent": "Mozilla/5.0"}
THROUGHPUT = 20000
MAX_PAGE = 10



FREE_MODELS = get_free_models()
SELECTED_MODEL = FREE_MODELS[0]["id"]

try: 
    with open(folder_path/"known_ids.json", "r") as f:
        HISTORIES = json.load(f)
        HISTORIES = set(HISTORIES)
except:
    pdb.set_trace()
    HISTORIES = set()
    
START_IDX = 0
MAX_IDS = 500


client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_KEY"),
    base_url=os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
)

async def is_model_callable(model_id:str)-> bool:
    """Check if a model is callable by making a test API request.

    Args:
        model_id (str): The ID of the model to check.

    Returns:
        bool: True if the model is callable, False otherwise.
    """
    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=5,
            temperature=0
        )
        return True
    except Exception as e:
        print(f"❌ Cannot use model {model_id}: {str(e)}")
        return False
    
def load_useless(path=folder_path/"useless_models.json") -> dict:
    """
    Load a list of models that are not useful for summarization.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}
    # return ["google/gemini-2.5-pro-exp-03-25"]

async def update_good_models(useless_models)-> Tuple[list, dict]:    
    """Update the list of good models by checking their availability.
    Args:
        useless_models (dict): A dictionary of models that are not useful.  
    Returns:
        list: A list of available models that are not in the useless models.
    """
    cache = []
    count = 0
    
    for free_model in FREE_MODELS:
        if free_model["id"] not in useless_models or useless_models.get(free_model["id"], 0) < 3:
            if await is_model_callable(free_model["id"]):
                cache.append(free_model)
                count += 1
            else:
                if free_model["id"] not in useless_models:
                    useless_models[free_model["id"]] = 1
                else:
                    useless_models[free_model["id"]] += 1
        # pdb.set_trace()
        if count >= 3:
            break
    return cache, useless_models
    

def parse_llm_response(resp) -> str:
    """
    適配新版 openai.ChatCompletion 回傳物件（Pydantic 模型）
    """
    if not resp or not resp.choices:
        return "[❌ Invalid response structure]"
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

    return "[⚠️ No usable content in response]"

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
                    print("❌ Failed to download PDF:", response.status)
                    return None
                else:
                    print("✅ PDF downloaded successfully.")
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
    global HISTORIES

    url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&scisbd=1"
    async with throttler:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=HEADERS) as resp:
                if resp.status != 200:
                    print(f"❌ Failed to fetch papers: {resp.status}")
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
                    
                    if p_id in HISTORIES:
                        continue
                    # pdb.set_trace()
                    
                    HISTORIES.add(p_id)
                    
                    # Abstract snippet
                    snippet_tag = r.select_one(".gs_rs")
                    snippet = snippet_tag.text.strip() if snippet_tag else "N/A"
                    curr["abstract"] = snippet
                    
                    # PDF link if available (右側 PDF 下載連結)
                    pdf_tag = r.select_one(".gs_or_ggsm a")
                    pdf_link = pdf_tag["href"] if pdf_tag else None
                    curr["pdf_link"] = pdf_link if pdf_link else "N/A"
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
    global HISTORIES

    base_url = "http://export.arxiv.org/api/query?"
    query_params = (
        f"search_query=all:{query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    url = base_url + query_params

    keynotes = {}

    async with throttler:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"❌ Failed to fetch arXiv papers: {resp.status}")
                    return {}

                text = await resp.text()
                feed = feedparser.parse(text)
                # pdb.set_trace()
                for entry in feed.entries:
                    title = entry.title.strip()
                    p_id = hashlib.md5(title.encode("utf-8")).hexdigest()
                    
                    if p_id in HISTORIES:
                        continue
                    
                    HISTORIES.add(p_id)

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
                        "abstract": summary,
                        "pdf_link": pdf_link
                    }

                    print(keynotes[p_id])
                    print("-" * 80)

                await asyncio.sleep(1)

    return keynotes


async def load_and_summarize(pdf_link:str, max_tokens=250) -> str:
    """Load a PDF from a link and summarize its content through Llms' apis.

    Args:
        pdf_link (str): The URL of the PDF file to load.
        max_tokens (int, optional): The maximum number of tokens for the summary. Defaults to 250.

    Returns:
        str: The summarized content of the PDF or an error message.
    """

    if pdf_link != "N/A" and pdf_link is not None:
        print(f"📄 PDF link: {pdf_link}")

        pdf_text = await download_and_extract_pdf_text(pdf_link)
        # pdb.set_trace()
        
        @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
        )
        async def summarize(text, throttler=throttler_summary):
            global START_IDX
            if text:            
                prompt = f"""
                    Summarize the following text in a simple and concise manner and explain in physic meaning or philosophy. The summarization should make people understand the core concept in a succinct way,
                    no redundant prefix and suffix: \n\n{text}\n
                """
                try:
                    async with throttler:
                        resp = await client.chat.completions.create(
                            model=FREE_MODELS[START_IDX]["id"],
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens
                        )
                        res = parse_llm_response(resp)
                        # pdb.set_trace()
                        print("[Info] Summary length:", len(res))
                        return res
                except Exception as e:
                    print("Error summarizing PDF:", e)
                    START_IDX = (START_IDX + 1) % len(FREE_MODELS)
                return ""
        if pdf_text is not None and len(pdf_text) > THROUGHPUT:
            summarized_res = await summarize(pdf_text[:THROUGHPUT], throttler=throttler_summary)       
            return summarized_res
        else:
            return ""
    else:
        print("📄 PDF link: Not available")
        return ""
    
 

def load_old_articles(path=folder_path/"known_articles.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    
def load_old_ids(path=folder_path/"known_ids.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_articles(articles, path=folder_path/"known_articles.json"):
    with open(path, "w") as f:
        json.dump(articles, f)

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
                    print("❌ Failed to send message:", resp.status, await resp.text())
                else:
                    print("✅ Message sent successfully.")
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
    abstract = clean_telegram_text(paper.get("abstract", ""))
    pdf_link = paper.get("pdf_link", "")
    summary = paper.get("summary")

    msg = f"📄 *Title:* {title}\n"
    msg += f"📝 *Abstract:* {abstract}\n"
    msg += f"🔗 [PDF Link]({pdf_link})\n"

    if summary:
        msg += f"🧠 *Summary:* {clean_telegram_text(summary)}\n"

    return msg


async def main(queries=["llm security", "llm jailbreak", "ai agent"], path=folder_path/"known_ids.json"):
    global FREE_MODELS, SELECTED_MODEL
    import copy 
    previous_hist = copy.copy(HISTORIES)
    useless_models = load_useless()
    FREE_MODELS, useless_models = await update_good_models(useless_models)
    with open(folder_path/"useless_models.json", "w") as f:
        json.dump(useless_models, f, indent=4)
    SELECTED_MODEL = FREE_MODELS[0]["id"]
    queries = ["+(all:security+OR+all:jailbreak)"]
    tasks = []
    for query in queries:
        tasks.append(extract_arxiv_papers(throttler_query, query))
    parses = await asyncio.gather(*tasks)
    
    all_papers = {}
    
    # // no summary parses
    # for parse in parses:
    #     all_papers.update(parse)
    
    #// for summary queries
    summarize_tasks = []
    p_ids = []
    for parse in parses:
        all_papers.update(parse)
        for k, v in parse.items():
            if v.get("pdf_link") and v.get("pdf_link") != "N/A":
                summarize_tasks.append(load_and_summarize(v["pdf_link"]))
                p_ids.append(k)
    
    summaries = await asyncio.gather(*summarize_tasks)
    for p_id, summary in zip(p_ids, summaries):
        all_papers[p_id]["summary"] = summary
        
    all_ids = set()
    for id_ in all_papers.keys():
        all_ids.add(id_)
    all_ids = list(all_ids)
    new_ids = list(previous_hist) + all_ids
    if len(new_ids) > MAX_IDS:
        new_ids = new_ids[-MAX_IDS:]
    new_ids.sort()

    with open(path, "w") as f:
        json.dump(list(new_ids), f, indent=4)

    send_tasks = []
    for _,paper in all_papers.items():
        formatted_msg = format_telegram_message(paper)
        send_tasks.append(send_telegram(throttler_send, formatted_msg))
    await asyncio.gather(*send_tasks)
    

if __name__ == "__main__":
    asyncio.run(main())
