# -*- coding: utf-8 -*-
import gradio as gr
import json
import os
import requests
import logging
import sys
import time
import re
import io
import html
import traceback  # For detailed error logging
import subprocess  # For running the data processing script
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx
import random
import base64  # Import base64 module for embedding images
import matplotlib.font_manager as fm

# Configure Chinese fonts for matplotlib used by the app
def _configure_chinese_font_for_app(preferred_fonts=None):
    if preferred_fonts is None:
        preferred_fonts = [
            "Noto Sans CJK SC",
            "NotoSansCJKsc",
            "Microsoft YaHei",
            "MicrosoftYaHei",
            "PingFang SC",
            "SimHei",
            "SimSun",
            "Heiti SC",
            "STHeiti",
            "Arial Unicode MS",
        ]
    try:
        available_names = {f.name for f in fm.fontManager.ttflist}
        for name in preferred_fonts:
            if name in available_names:
                plt.rcParams['font.family'] = name
                logger.info(f"配置应用级中文字体: 使用 '{name}'")
                break
        else:
            for f in fm.fontManager.ttflist:
                lname = f.name.lower()
                if any(k in lname for k in ("noto", "cjk", "pingfang", "heiti", "sim", "msyh", "song")):
                    plt.rcParams['font.family'] = f.name
                    logger.info(f"配置应用级中文字体: 检测并使用 '{f.name}'")
                    break
            else:
                logger.warning("未检测到已知中文字体。图表中的中文可能无法正确显示。")
    except Exception as e:
        logger.warning(f"配置应用级中文字体时出错: {e}")


# (Will apply font configuration after logger is initialized)
# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s',
    handlers=[logging.FileHandler("arxiv_downloader.log", encoding='utf-8'),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
MAX_RESULTS = 50
PDF_BASE_URL = "https://yuanzhuo.bnu.edu.cn/downloads/hrh/"  # User-provided base URL
# Apply font configuration now that logger exists
try:
    _configure_chinese_font_for_app()
except Exception:
    # If logger isn't available for some reason, fall back silently
    print("Warning: failed to configure Chinese font for app.")
# Ensure minus sign rendered correctly
plt.rcParams['axes.unicode_minus'] = False
# --- ChatPDF API Specifics ---
CHATPDF_API_BASE_URL = "https://api.chatpdf.com/v1"
CHATPDF_ADD_URL_ENDPOINT = f"{CHATPDF_API_BASE_URL}/sources/add-url"
CHATPDF_MESSAGE_ENDPOINT = f"{CHATPDF_API_BASE_URL}/chats/message"
CHATPDF_DELETE_ENDPOINT = f"{CHATPDF_API_BASE_URL}/sources/delete"
CHATPDF_API_TIMEOUT_ADD = 60  # Timeout for adding a PDF
CHATPDF_API_TIMEOUT_INITIAL = 120  # Timeout for initial summary/questions
CHATPDF_API_TIMEOUT_STREAM = 180  # Timeout for streaming chat response
# --- ArxivDownloader Class ---
class ArxivDownloader:
    def __init__(self):
        self.metadata_file = "all_papers.json"
        try:
            self._check_initialization()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
    def _check_initialization(self):
        if not os.path.exists(self.metadata_file):
            logger.warning(f"Metadata file not found: {self.metadata_file}. Search will return no results.")
            try:
                with open(self.metadata_file, "w", encoding="utf-8") as f:
                    f.write("")  # Create empty file
                    logger.info(f"Created empty metadata file: {self.metadata_file}")
            except Exception as e:
                logger.error(f"Failed to create empty metadata file {self.metadata_file}: {e}")
        else:
            try:
                # Check if file is readable and not empty (at least contains [])
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    content = f.read(10)  # Read a few bytes
                logger.info(f"Metadata file check passed ('{self.metadata_file}' exists and readable).")
            except Exception as e:
                logger.error(f"Metadata file {self.metadata_file} check failed (cannot read): {e}")
        logger.info("ArxivDownloader initialized.")
    def load_metadata(self, query, max_papers=MAX_RESULTS):
        logger.info(f"Searching metadata file '{self.metadata_file}' for query: '{query}', max_results={max_papers}")
        papers = []
        query_lower = query.lower().strip() if query else ""
        if not query_lower:
            logger.warning("Search query is empty.")
            return []
        if not os.path.exists(self.metadata_file):
            logger.error(f"Metadata file '{self.metadata_file}' not found.")
            return []
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                matched_count = 0
                processed_count = 0
                for line_number, line in enumerate(f, 1):
                    processed_count += 1
                    if matched_count >= max_papers:
                        logger.info(f"Reached max_papers limit ({max_papers}).")
                        break
                    try:
                        if not line.strip():
                            continue
                        paper = json.loads(line)
                        # Ensure values are strings before lowercasing
                        title = str(paper.get("title", "") or "").lower()
                        abstract = str(paper.get("abstract", "") or "").lower()
                        authors_list = paper.get("authors", []) or []
                        authors = " ".join(map(str, authors_list)).lower()
                        paper_id = str(paper.get("id", "") or "").lower()
                        # Combine searchable fields
                        search_target = f"{title} {abstract} {authors} {paper_id}"
                        if query_lower in search_target:
                            papers.append(paper)
                            matched_count += 1
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {line_number}: {line[:100]}...")
                    except Exception as e:
                        logger.warning(f"Error processing line {line_number}: {e} - Line: {line[:100]}...", exc_info=False)  # Log less verbose errors for bad lines
            logger.info(f"Search completed. Found {len(papers)} papers matching '{query}' from {processed_count} lines processed.")
            return papers[:max_papers]
        except FileNotFoundError:
            logger.error(f"Metadata file '{self.metadata_file}' not found during search.")
            return []
        except Exception as e:
            logger.error(f"Failed to load or process metadata from {self.metadata_file}: {e}", exc_info=True)
            return []
# --- Helper Functions ---
def clean_title(title):
    if not isinstance(title, str):
        title = str(title) if title is not None else ""
    title = re.sub(r'[\\/*?:"<>|\n\r\t]+', "_", title)
    title = re.sub(r'[_ ]+', '_', title)
    title = title.strip('_')
    title = title[:100]  # Limit length
    return title if title else "Untitled_Paper"
def construct_pdf_url(paper_id):
    if not paper_id:
        return None
    # Basic sanitization: remove potential path traversal or unexpected chars
    safe_paper_id = os.path.basename(str(paper_id))
    # Ensure it ends with .pdf
    filename = f"{safe_paper_id}.pdf" if not safe_paper_id.lower().endswith('.pdf') else safe_paper_id
    base = PDF_BASE_URL.rstrip('/')
    full_url = f"{base}/{filename}"
    logger.info(f"Constructed PDF URL for {paper_id}: {full_url}")
    return full_url
# --- ChatPDF API Interaction Functions ---
def add_pdf_to_chatpdf(pdf_url, api_key):
    logger.info(f"Adding PDF to ChatPDF via URL: {pdf_url}")
    if not api_key or not api_key.startswith("sec_"):
        logger.error("ChatPDF API Key missing/invalid.")
        return None, "Config Error: Invalid or missing API Key."
    headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}
    data = {'url': pdf_url}
    try:
        response = requests.post(
            CHATPDF_ADD_URL_ENDPOINT,
            headers=headers,
            json=data,
            timeout=CHATPDF_API_TIMEOUT_ADD  # Use defined timeout
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        source_id = result.get('sourceId')
        if not source_id:
            logger.error(f"API Error: Missing sourceId in response for {pdf_url}. Response: {result}")
            return None, f"API Error: Missing sourceId. Details: {response.text[:200]}"
        logger.info(f"Successfully added PDF {pdf_url}. Source ID: {source_id}")
        return source_id, None
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out adding PDF {pdf_url}.")
        return None, "API Error: Request timed out."
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try:
            error_text = e.response.text[:200]  # Limit error text length
        except Exception:
            error_text = "(Could not read error text)"
        logger.error(f"API HTTP error adding PDF {pdf_url}: {status_code} {e.response.reason}. Details: {error_text}")
        if status_code == 400:
            error_msg = f"API Error ({status_code}): Invalid request (bad URL/file format?). Details: {error_text}"
        elif status_code == 401:
            error_msg = f"API Error ({status_code}): Invalid API Key."
        elif status_code == 402:
            error_msg = f"API Error ({status_code}): Payment required or quota exceeded."
        elif status_code == 429:
            error_msg = f"API Error ({status_code}): Rate limit exceeded."
        elif status_code >= 500:
            error_msg = f"API Error ({status_code}): ChatPDF server error. Details: {error_text}"
        else:
            error_msg = f"API Error ({status_code}): {e.response.reason}. Details: {error_text}"
        return None, error_msg
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed adding PDF {pdf_url}: {e}")
        return None, f"API Communication Error: {e}"
    except json.JSONDecodeError as e:
        logger.error(f"API response JSON decode error adding URL {pdf_url}: {e}. Response text: {response.text[:200]}")
        return None, "API Error: Invalid JSON response received."
    except Exception as e:
        logger.error(f"Unexpected error adding PDF {pdf_url}: {e}", exc_info=True)
        return None, f"Unexpected Error: {str(e)}"
def get_initial_chatpdf_message(source_id, api_key):
    logger.info(f"Requesting initial summary and questions for source ID: {source_id}")
    if not api_key or not api_key.startswith("sec_"):
        logger.error("API Key missing/invalid for initial message.")
        return None, "Config Error: Invalid or missing API Key."
    # Simple, clear prompt for summary and questions
    initial_prompt = (
"请对本文件进行简明扼要的总结（约150-200字）。在总结之后，提出三个与该文件相关的不同问题。请清晰编号问题（例如：1.、2.、3.）"
    )
    headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}
    # Include referenceSources here as well
    data = {
        "sourceId": source_id,
        "referenceSources": True,  # Enable references
        "messages": [
            {
                "role": "user",
                "content": initial_prompt
            }
        ]
    }
    try:
        logger.info(f"Sending request for initial message to {source_id}...")
        response = requests.post(
            CHATPDF_MESSAGE_ENDPOINT,
            headers=headers,
            json=data,
            timeout=CHATPDF_API_TIMEOUT_INITIAL  # Use defined timeout
        )
        response.raise_for_status()
        result = response.json()
        content = result.get('content')
        if content is None:  # Check for explicit None or missing key
            logger.error(f"API Error: No 'content' field in response for initial message {source_id}. Response: {result}")
            return None, f"API Error: No content received. Details: {response.text[:200]}"
        if not content.strip():  # Check for empty string content
            logger.warning(f"API Warning: Received empty 'content' for initial message {source_id}.")
            # Return a specific message instead of error, let user ask first question
            return "[System Note: Initial analysis from ChatPDF was empty. Please ask your first question.]", None
        logger.info(f"Successfully received initial response for {source_id}.")
        formatted_response = content.strip()
        # Optional: Add a note if questions seem present
        if re.search(r"\n\s*[1-3]\.", formatted_response):
            formatted_response += "\n\n*(Suggested questions above based on document content)*"
        return formatted_response, None
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out getting initial message for {source_id}.")
        return None, "API Error: Timeout getting initial summary."
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try:
            error_text = e.response.text[:200]
        except Exception:
            error_text = "(Could not read error text)"
        logger.error(f"API HTTP error getting initial message for {source_id}: {status_code} {e.response.reason}. Details: {error_text}")
        if status_code == 400:
            error_msg = f"API Error ({status_code}): Invalid request (bad Source ID?). Details: {error_text}"
        elif status_code == 401:
            error_msg = f"API Error ({status_code}): Invalid API Key."
        elif status_code == 402:
            error_msg = f"API Error ({status_code}): Payment required or quota exceeded."
        elif status_code == 422:
            error_msg = f"API Error ({status_code}): Unprocessable Entity (e.g., message format). Details: {error_text}"
        elif status_code == 429:
            error_msg = f"API Error ({status_code}): Rate limit exceeded."
        elif status_code >= 500:
            error_msg = f"API Error ({status_code}): ChatPDF server error. Details: {error_text}"
        else:
            error_msg = f"API Error ({status_code}): {e.response.reason}. Details: {error_text}"
        return None, error_msg
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed getting initial message for {source_id}: {e}")
        return None, f"API Communication Error: {e}"
    except json.JSONDecodeError as e:
        logger.error(f"API response JSON decode error getting initial message {source_id}: {e}. Response text: {response.text[:200]}")
        return None, "API Error: Invalid JSON response received."
    except Exception as e:
        logger.error(f"Unexpected error getting initial message for {source_id}: {e}", exc_info=True)
        return None, f"Unexpected Error: {str(e)}"
def chatpdf_follow_up_stream(source_id, history, api_key):
    """
    Sends chat history to ChatPDF and yields streamed response chunks.
    Uses referenceSources=True and stream=True.
    Handles potential errors during streaming.
    """
    logger.info(f"Initiating stream for source {source_id}. History length: {len(history)}")
    if not api_key or not api_key.startswith("sec_"):
        logger.error("API Key missing/invalid for streaming.")
        yield "[System Error: Invalid or missing API Key. Configure in API tab.]"
        return
    messages_for_api = []
    MAX_CHATPDF_MESSAGES = 6  # ChatPDF limit
    # Prepare messages from history (ensure structure is correct)
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and isinstance(content, str):
            messages_for_api.append({"role": role, "content": content})
        else:
            logger.warning(f"Skipping invalid message structure in history: {msg}")
    # Truncate if too many messages (keeping the most recent ones)
    if len(messages_for_api) > MAX_CHATPDF_MESSAGES:
        logger.warning(f"History length ({len(messages_for_api)}) exceeds ChatPDF limit ({MAX_CHATPDF_MESSAGES}). Truncating.")
        messages_for_api = messages_for_api[-MAX_CHATPDF_MESSAGES:]
    # Ensure the last message is from the user before sending
    if not messages_for_api or messages_for_api[-1].get("role") != "user":
        logger.error(f"Invalid history state for API call. Last message not from user: {messages_for_api[-1:]}")
        yield "[System Error: Invalid chat history state for API call.]"
        return
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json',
        # 'Accept': 'text/event-stream' # Often not needed, requests handles stream=True
    }
    payload = {
        "sourceId": source_id,        "messages": messages_for_api,
        "referenceSources": True,  # Enable references in response
        "stream": True  # Enable streaming response
    }
    # Log the payload being sent (useful for debugging)
    try:
        payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
        logger.debug(f"ChatPDF Payload for {source_id}:\n{payload_str}")  # Use DEBUG level
    except Exception as log_e:
        logger.error(f"Error logging payload: {log_e}")
    try:
        logger.info(f"Sending streaming request to ChatPDF for {source_id}...")
        response = requests.post(
            CHATPDF_MESSAGE_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=CHATPDF_API_TIMEOUT_STREAM,  # Use defined timeout
            stream=True  # Crucial for streaming
        )
        response.raise_for_status()  # Check for HTTP errors immediately
        # Process the stream using iter_content (more robust for potentially mixed encodings/chunks)
        # Based on ChatPDF Python example
        max_chunk_size = 1024  # Process in chunks
        received_content = False
        for chunk in response.iter_content(chunk_size=max_chunk_size):
            if chunk:
                try:
                    text = chunk.decode('utf-8')  # Decode using UTF-8
                    # SSE format is typically "data: <json_or_text>\n\n"
                    # Simple handling: yield the text directly if API sends raw text chunks
                    # If it sends SSE "data: " lines, you might need more parsing
                    # Assuming simple text chunks for now based on the API response description for non-stream
                    # If it _is_ SSE, the Gradio Chatbot might render "data: ..."
                    # Let's assume raw text chunks yield directly:
                    yield text
                    received_content = True
                    # logger.debug(f"Stream chunk received for {source_id}: {text[:50]}...") # Verbose
                except UnicodeDecodeError:
                    logger.warning(f"Unicode decode error in stream chunk for {source_id}. Chunk (raw): {chunk[:50]}...")
                    yield "[System Warning: Decoding error in stream]"
                except Exception as decode_e:
                    logger.error(f"Error processing stream chunk for {source_id}: {decode_e}")
                    yield f"[System Error: Error processing chunk: {str(decode_e)}]"
        if not received_content:
            logger.warning(f"Stream finished for {source_id} but resulted in an empty response.")
            # Don't yield an empty response message here, let the calling function handle it
        logger.info(f"Stream finished successfully for {source_id}.")
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out during streaming for {source_id}.")
        yield "\n\n[System Error: API request timed out during chat.]"
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_text = f"Streaming response details unavailable."
        try:
            # Try to get error text if available _before_ stream was consumed
            if not response.raw.closed and response.content:
                error_text = response.text[:200]
        except Exception:
            pass
        logger.error(f"API HTTP error during stream for {source_id}: {status_code} {e.response.reason}. Details: {error_text}")
        # Provide user-friendly error messages based on common codes
        if status_code == 401:
            error_msg = "[System Error: Invalid API Key.]"
        elif status_code == 402:
            error_msg = "[System Error: Payment required or quota exceeded.]"
        elif status_code == 429:
            error_msg = "[System Error: Rate limit exceeded. Please wait.]"
        elif status_code == 400:
            error_msg = f"[System Error: Bad Request ({status_code}). Check input. {error_text}]"
        elif status_code == 422:
            error_msg = f"[System Error: Unprocessable Entity ({status_code}). Check message format. {error_text}]"
        elif status_code >= 500:
            error_msg = f"[System Error: ChatPDF server error ({status_code}). Please try again later. {error_text}]"
        else:
            error_msg = f"[System Error: API Error ({status_code}). {error_text}]"
        yield f"\n\n{error_msg}"
    except requests.exceptions.RequestException as e:
        logger.error(f"API communication failed during streaming for {source_id}: {e}")
        yield f"\n\n[System Error: API communication failed. {str(e)}]"
    except GeneratorExit:
        logger.warning(f"Stream consumer closed connection for {source_id}.")  # User navigated away or stopped
    except Exception as e:
        logger.error(f"Unexpected error during streaming chat for {source_id}: {e}", exc_info=True)
        yield f"\n\n[System Error: Unexpected error processing chat. {str(e)}]"
# --- Gradio Event Handlers Definition ---
def search_and_display(query, num_papers):
    """Performs search and generates HTML for display with detailed logging."""
    logger.info(f"Starting search: '{query}', max results requested: {num_papers}")
    status = "Searching..."
    html_out = "<p>Enter a query and click 'Search' to see results.</p>"
    papers = []  # Initialize papers list
    try:
        # Validate num_papers
        try:
            max_papers = int(num_papers)
            if not 1 <= max_papers <= MAX_RESULTS * 2:
                logger.warning(f"Requested {max_papers} results, clamping to range [1, {MAX_RESULTS * 2}]. Using {MAX_RESULTS}.")
                max_papers = MAX_RESULTS
            else:
                logger.info(f"Search limit set to {max_papers}.")
        except (ValueError, TypeError):
            logger.warning(f"Invalid num_papers value '{num_papers}', defaulting to {MAX_RESULTS}.")
            max_papers = MAX_RESULTS
        # Perform Search
        downloader = ArxivDownloader()
        papers = downloader.load_metadata(query, max_papers)  # Get papers list
        # Generate HTML Output
        html_parts = []
        if papers:  # Only build parts if papers were found
            logger.info(f"Generating HTML for {len(papers)} found papers.")
            for idx, p in enumerate(papers, 1):
                try:
                    # Extract and sanitize data with fallback for uppercase keys
                    paper_id = str(p.get("id") or p.get("ID") or f"UnknownID_{idx}")
                    title = str(p.get("title") or p.get("Title") or "No Title Available")
                    # 处理作者字段：如果是列表则 join，如果是字符串则直接使用
                    authors_raw = p.get("authors") or p.get("Authors")
                    if isinstance(authors_raw, list):
                        authors = ", ".join(map(str, authors_raw)) if authors_raw else "N/A"
                    elif isinstance(authors_raw, str):
                        authors = authors_raw.strip() or "N/A"
                    else:
                        authors = "N/A"
                    # 处理类别字段同上
                    categories_raw = p.get("categories") or p.get("Categories")
                    if isinstance(categories_raw, list):
                        categories = ", ".join(map(str, categories_raw)) if categories_raw else "N/A"
                    elif isinstance(categories_raw, str):
                        categories = categories_raw.strip() or "N/A"
                    else:
                        categories = "N/A"
                    abstract = str(p.get("abstract") or p.get("Abstract") or "No Abstract Available")
                    # 对于日期，尝试多个可能的键名
                    pub_date_raw = (p.get("publication_date") or p.get("Publication_Date") or
                                    p.get("submittedDate") or p.get("SubmittedDate") or
                                    p.get("update_date") or p.get("UpdateDate"))
                    pub_date = str(pub_date_raw).strip() if pub_date_raw and str(pub_date_raw).strip() else "N/A"
                    doi = p.get("doi") or p.get("DOI")
                    # Escape data for HTML
                    title_esc = html.escape(title)
                    paper_id_esc = html.escape(paper_id)
                    authors_esc = html.escape(authors)
                    cats_esc = html.escape(categories)
                    pub_date_esc = html.escape(pub_date)
                    abstract_short = html.escape(abstract[:500] + ("..." if len(abstract) > 500 else "")).replace('\n', '<br>')
                    # Format DOI link
                    doi_display = "N/A"
                    if isinstance(doi, str) and doi.strip() and doi.strip().lower() != 'n/a':
                        doi_esc = html.escape(doi.strip())
                        doi_display = f"<a href='https://doi.org/{doi_esc}' target='_blank'>{doi_esc}</a>"
                    # Construct PDF URL and link
                    pdf_file_url = construct_pdf_url(paper_id)
                    if pdf_file_url:
                        pdf_link_html = f"<a href='{html.escape(str(pdf_file_url))}' target='_blank' class='action-button blue-button'>Download PDF</a>"
                    else:
                        pdf_link_html = "<span style='color: red;'>PDF URL Error</span>"
                    # Prepare paper_id for JS function (ensure proper quoting)
                    paper_id_js = json.dumps(paper_id)
                    # Build HTML item string
                    item = f"""
<div class='search-result-item'>
    <div class='result-title'><strong>{idx}. {title_esc}</strong></div>
    <div class='result-details'>
        <strong>ID:</strong> {paper_id_esc}<br>
        <strong>Authors:</strong> {authors_esc}<br>
        <strong>Categories:</strong> {cats_esc}<br>
        <strong>Date:</strong> {pub_date_esc}<br>
        <strong>DOI:</strong> {doi_display}
    </div>
    <div class='result-abstract'><strong>Abstract:</strong> {abstract_short}</div>
    <div class='result-actions'>
        {pdf_link_html}
        <button class='action-button blue-button chat-button' onclick='activateChatPanel({paper_id_js})'>Chat</button>
    </div>
</div>
"""
                    html_parts.append(item)
                except Exception as e:
                    current_id = p.get('id', f'ErrorItem_{idx}')
                    logger.error(f"Error processing paper data for display (ID: {current_id}): {e}", exc_info=True)
                    html_parts.append(f"<div class='search-result-item error'><p>Error displaying paper ID: {html.escape(str(current_id))}. Check logs for details.</p></div>")
        if papers and not html_parts:
            logger.error("CRITICAL: Papers list has items, but HTML parts list is empty! Check item processing loop.")
            html_out = "<p style='color:red;'>Internal error: Failed to generate result items. Please check application logs.</p>"
            status = "Error generating results"
        elif not papers:
            status = f"No papers found matching query: '{query}'"
            logger.info(status)
            html_out = f"<p>No papers found matching your query: '{html.escape(query)}'.</p>"
            logger.info(f"RETURN NO RESULTS: HTML (len:{len(html_out)}): {html_out[:200]}...")
        else:
            html_out = "\n".join(html_parts)
            status = f"Found {len(papers)} paper(s) matching query: '{query}'."
            logger.info(f"Search successful: {status}")
            logger.info(f"RETURN SUCCESS: HTML (len:{len(html_out)}): {html_out[:500]}...")
        return gr.update(value=html_out), status
    except Exception as e:
        logger.error(f"Unexpected error in search_and_display: {e}", exc_info=True)
        return gr.update(value=f"<p style='color:red;'>Unexpected error: {html.escape(str(e))}</p>"), "Error"
def handle_chat_activation(paper_id, current_api_key):
    """Handles the 'Chat' button click: adds PDF to ChatPDF and gets initial summary."""
    start_time = time.time()
    activation_source_id = None  # Keep track of source ID for this activation
    # Initial loading message
    yield (paper_id, None, [{"role": "system", "content": f"Loading context for paper: {html.escape(str(paper_id))}..."}],
           "Status: Loading PDF...", f"Loading: **{html.escape(str(paper_id))}**...")
    if not paper_id or not isinstance(paper_id, str):
        logger.error(f"Invalid paper_id received for chat activation: {paper_id} (type: {type(paper_id)})")
        yield (paper_id, None, [{"role": "system", "content": "Error: Invalid Paper ID received."}],
               "Status: Error - Invalid ID", "Chatting with: Error - Invalid ID")
        return
    logger.info(f"Activating chat for paper ID: {paper_id}")
    # Check API Key early
    if not current_api_key or not current_api_key.startswith("sec_"):
        logger.warning(f"ChatPDF API key missing or invalid during activation for {paper_id}.")
        yield (paper_id, None, [{"role": "system",
                                 "content": "Error: ChatPDF API Key is missing or invalid. Please configure it in the 'API Configuration' tab."}],
               "Status: Error - API Key", f"**{html.escape(paper_id)}** (API Key Error)")
        return
    # Construct PDF URL
    pdf_url = construct_pdf_url(paper_id)
    if not pdf_url:
        logger.error(f"Could not construct PDF URL for paper ID: {paper_id}")
        yield (paper_id, None, [{"role": "system",
                                 "content": f"Error: Could not determine the PDF URL for paper {html.escape(paper_id)}."}],
               "Status: Error - PDF URL", f"**{html.escape(paper_id)}** (URL Error)")
        return
    # Add PDF to ChatPDF
    logger.info(f"Adding PDF URL to ChatPDF for {paper_id}: {pdf_url}")
    yield (paper_id, None, [{"role": "system", "content": f"Adding PDF for {html.escape(paper_id)} to ChatPDF..."}],
           "Status: Adding PDF...", f"Processing: **{html.escape(paper_id)}**...")
    source_id, error_msg = add_pdf_to_chatpdf(pdf_url, current_api_key)
    if error_msg or not source_id:
        logger.error(f"Failed to add PDF {paper_id} (URL: {pdf_url}) to ChatPDF: {error_msg}")
        error_display = error_msg or "Unknown error adding PDF."
        yield (paper_id, None, [{"role": "system",
                                 "content": f"Error adding PDF {html.escape(paper_id)} to ChatPDF: {html.escape(error_display)}<br>Please check the PDF URL is accessible, your API key is valid, and you have available quota."}],
               f"Status: Error - {error_display}", f"**{html.escape(paper_id)}** (Add PDF Error)")
        return
    activation_source_id = source_id  # Store the source ID for this session
    logger.info(f"PDF added successfully for {paper_id}. Source ID: {activation_source_id}. Requesting initial analysis.")
    # Get Initial Summary/Questions from ChatPDF
    yield (paper_id, activation_source_id,
           [{"role": "system", "content": f"PDF added (Source ID: {activation_source_id}). Analyzing document content..."}],
           "Status: Analyzing PDF...", f"Analyzing: **{html.escape(paper_id)}** (Source: {activation_source_id})")
    initial_llm_response, error_msg = get_initial_chatpdf_message(activation_source_id, current_api_key)
    # Prepare final state based on initial analysis outcome
    final_history = []
    final_status = ""
    final_display_text = f"Chatting with: **{html.escape(paper_id)}** (Source: {activation_source_id})"  # Default success display
    if error_msg:
        logger.error(f"Failed to get initial message/summary for {activation_source_id}: {error_msg}")
        final_history = [{"role": "system",
                          "content": f"Error getting initial analysis for {html.escape(paper_id)} ({activation_source_id}):<br>{html.escape(error_msg)}"}]
        final_status = f"Status: Error - {error_msg}"
        final_display_text = f"**{html.escape(paper_id)}** (Analysis Error)"  # Update display on error
    elif initial_llm_response:
        logger.info(f"Successfully received initial response for {activation_source_id}. Displaying.")
        final_history = [{"role": "assistant",
                          "content": initial_llm_response}]  # Start chat with assistant's summary/questions
        final_status = "Status: PDF Analyzed. Ready for questions."
    else:
        # This case should be handled by get_initial_chatpdf_message returning a specific note now
        logger.warning(f"Initial analysis for {activation_source_id} returned empty content (handled).")
        final_history = [{"role": "system",
                          "content": f"Context loaded for {html.escape(paper_id)} ({activation_source_id}). Ask your first question."}]
        final_status = "Status: Ready. Ask a question."
    elapsed_time = time.time() - start_time
    logger.info(f"Chat activation for {paper_id} (Source: {activation_source_id}) completed in {elapsed_time:.2f}s. Final Status: {final_status}")
    # Yield the final state
    yield (paper_id, activation_source_id, final_history, final_status, final_display_text)
def send_chat_message(active_id, source_id, message, history, current_api_key):
    """Handles sending a user message and streaming the response."""
    current_history = history if isinstance(history, list) else []
    # Validate input message
    if not message or not message.strip():
        logger.warning("Empty message submitted by user.")
        # Do not modify history, just update status and clear input
        yield current_history, gr.update(value=""), "Status: Please enter a question or message."
        return
    logger.info(f"User message received for paper {active_id} (Source: {source_id}): '{message[:50]}...'")
    # Append user message to history
    current_history.append({"role": "user", "content": message})
    # Update UI immediately to show user message and clear input
    yield current_history, gr.update(value=""), "Status: Sending message..."
    # --- Pre-API Call Checks ---
    if not source_id:
        logger.error(f"Chat context error: Missing Source ID for active paper {active_id}.")
        current_history.append({"role": "system",
                                "content": "[System Error] Chat context (Source ID) is missing. Cannot send message. Try clicking 'Chat' on the paper again."})
        yield current_history, "", "Status: Error - Context missing."
        return
    if not current_api_key or not current_api_key.startswith("sec_"):
        logger.warning(f"API key missing/invalid when sending chat message for {active_id} ({source_id}).")
        current_history.append({"role": "system", "content": "[System Error] API Key missing or invalid. Configure in API tab."})
        yield current_history, "", "Status: Error - API Key."
        return
    # --- Call Streaming API ---
    logger.info(f"Calling chatpdf_follow_up_stream for source {source_id}")
    # Add placeholder for the assistant's response (will be filled by stream)
    current_history.append({"role": "assistant", "content": ""})
    full_assistant_response = ""
    stream_error_occurred = False
    system_error_prefix = "[System Error"
    system_warning_prefix = "[System Warning"
    try:
        stream_generator = chatpdf_follow_up_stream(source_id, current_history[:-1],
                                                     current_api_key)  # Pass history _before_ placeholder
        for chunk in stream_generator:
            if chunk:
                full_assistant_response += chunk
                # Update the content of the last message (assistant placeholder)
                current_history[-1]["content"] = full_assistant_response.strip()  # Use strip() for cleaner look
                # Check for error messages yielded by the stream function itself
                if chunk.strip().startswith(system_error_prefix) or chunk.strip().startswith(system_warning_prefix):
                    logger.warning(f"System error/warning detected in stream for {source_id}: {chunk.strip()}")
                    stream_error_occurred = True
                    yield current_history, "", "Status: Error during response stream."  # Update status
                else:
                    # Normal chunk received, update UI
                    yield current_history, "", "Status: Receiving response..."
    except Exception as e:
        logger.error(f"Error iterating through stream generator for {source_id}: {e}", exc_info=True)
        error_message = f"\n\n[System Error] Failed to process response stream: {str(e)}"
        # Append error to the assistant message placeholder
        current_history[-1]["content"] = (full_assistant_response + error_message).strip()
        stream_error_occurred = True
        yield current_history, "", "Status: Error processing stream."
        return  # Stop processing on generator error
    # --- Post-Stream Processing ---
    final_status = "Status: Response received."
    if stream_error_occurred:
        final_status = "Status: Error occurred during response."
        # Error message should already be in the history from the stream loop
    elif not full_assistant_response.strip():
        logger.warning(f"Stream finished for {source_id} but resulted in an empty response.")
        final_status = "Status: Received empty response from API."
        # Update the placeholder with a note about the empty response
        current_history[-1]["content"] = "[System Note: Received empty response from ChatPDF.]"
    else:
        # Stream succeeded and got content
        logger.info(f"Stream processing finished successfully for {source_id}.")
    # Final update to UI after stream completes or errors out
    yield current_history, "", final_status
def clear_chat(active_source_id, current_api_key):
    """Clears the chat UI and optionally deletes the source from ChatPDF."""
    logger.info("Clearing chat panel requested.")
    # --- Optional: Delete source from ChatPDF ---
    # Set delete_source_on_clear to True if you want to remove the PDF from ChatPDF servers when clearing.
    # Be cautious: this means you'd need to re-upload/re-add it next time.
    delete_source_on_clear = False
    if delete_source_on_clear and active_source_id:
        if not current_api_key or not current_api_key.startswith("sec_"):
            logger.warning(f"Cannot delete source {active_source_id}: API key missing or invalid.")
        else:
            logger.info(f"Attempting to delete ChatPDF source: {active_source_id}")
            headers = {'x-api-key': current_api_key, 'Content-Type': 'application/json'}
            data = {'sources': [active_source_id]}
            try:
                response = requests.post(CHATPDF_DELETE_ENDPOINT, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                logger.info(f"Successfully deleted ChatPDF source: {active_source_id}")
            except requests.exceptions.RequestException as e:
                # Log error but proceed with clearing UI
                logger.error(f"Failed to delete ChatPDF source {active_source_id}: {e}", exc_info=True)
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Delete response status: {e.response.status_code}, text: {e.response.text[:200]}")
            except Exception as e:
                logger.error(f"Unexpected error deleting source {active_source_id}: {e}", exc_info=True)
    elif delete_source_on_clear:
        logger.info("Clear chat called, but no active source ID to delete.")
    # --- End Optional Deletion ---
    # Reset Gradio components related to chat
    return (
        None,  # active_paper_id_state
        None,  # active_source_id_state
        [],  # chatbot history
        "Status: Chat cleared.",  # chat_status
        "Click 'Chat' on a paper to begin."  # current_paper_display
    )
def run_aggregate_analysis(query_string=None):
    """Runs the aggregate_data.py script and returns the analysis results and visualizations."""
    try:
        query_terms = [term.strip() for term in query_string.split(",")] if query_string else None
        logger.info(f"Running aggregate analysis with query terms: {query_terms}")
        # Construct the command to execute the aggregation script
        command = ["python", "aggregate_data.py"]
        if query_terms:
            command.extend(["--query", ",".join(query_terms)])  # Pass query terms to the script
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Log the output and error messages
        logger.info(f"Aggregate script output:\n{result.stdout}")
        if result.stderr:
            logger.error(f"Aggregate script error:\n{result.stderr}")
        # Load the analysis results from the file
        try:
            with open("aggregate_analysis.json", "r", encoding="utf-8") as f:
                analysis_results = json.load(f)
            logger.info("Successfully loaded aggregate analysis results.")
            # 提取各部分HTML
            concept_cooccurrence_chart = analysis_results.get("concept_cooccurrence_chart", "")
            theory_acceptance_chart = analysis_results.get("theory_acceptance_chart", "")
            collaboration_network_graph = analysis_results.get("collaboration_network_graph", "")
            # 构造HTML img 标签
            concept_cooccurrence_html = f'<img src="data:image/png;base64,{concept_cooccurrence_chart}" alt="Concept Co-occurrence Chart" style="max-width:100%;">' if concept_cooccurrence_chart else "<p>No concept co-occurrence data available.</p>"
            theory_acceptance_html = f'<img src="data:image/png;base64,{theory_acceptance_chart}" alt="Theory Acceptance Chart" style="max-width:100%;">' if theory_acceptance_chart else "<p>No theory acceptance data available.</p>"
            collaboration_network_html = f'<img src="data:image/png;base64,{collaboration_network_graph}" alt="Collaboration Network Graph" style="max-width:100%;">' if collaboration_network_graph else "<p>No collaboration network data available.</p>"
            # 组合所有信息
            visualization_html = f"""
                <div>
                    <h3>Concept Co-occurrence</h3>
                    {concept_cooccurrence_html}
                </div>
                <div>
                    <h3>Top Theories by Acceptance</h3>
                    {theory_acceptance_html}
                </div>
                <div>
                    <h3>Collaboration Network</h3>
                    {collaboration_network_html}
                </div>
                <p>分析状态：{analysis_results.get("message", "")}</p>
            """
            return gr.update(value=visualization_html), "分析完成"  # 分析成功，返回结果和状态
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from aggregate_analysis.json: {e}")
            return gr.update(value=f"分析出错：无法读取分析结果文件 (JSON 错误): {e}"), "分析出错"  # 分析失败，返回错误
        except FileNotFoundError:
            logger.error("aggregate_analysis.json not found.")
            return gr.update(value="分析出错：找不到分析结果文件"), "分析出错"  # 分析失败
    except subprocess.CalledProcessError as e:
        logger.error(f"Aggregate analysis script failed with error: {e}")
        return gr.update(value=f"分析出错：聚合分析脚本执行失败: {str(e)}"), "分析出错"  # 分析失败
    except Exception as e:
        logger.error(f"Unexpected error running aggregate analysis: {e}", exc_info=True)
        return gr.update(value=f"分析出错：发生意外错误: {str(e)}"), "分析出错"  # 分析失败
# --- Global variables ---
# Load API key from environment variable or use placeholder
# Placeholder key will likely fail, user MUST configure via UI or ENV
api_key_global = os.environ.get("CHATPDF_API_KEY", "sec_xG3xmhaUPXOHZiqAEBl5aXfwSYCM0NBZ")  # Default to empty string
# --- Gradio Interface ---
with gr.Blocks(title="Paper Search & Chat (ChatPDF)", theme=gr.themes.Soft()) as demo:
    # --- State Management ---
    # Use gr.State to hold session-specific values like API key and active chat context
    api_key_state = gr.State(api_key_global)
    active_paper_id_state = gr.State(None)
    active_source_id_state = gr.State(None)
    search_results_state = gr.State([]) # Store search results
    # Hidden components for JS interaction
    js_paper_id_trigger = gr.Textbox(visible=False, label="JS Paper ID Trigger", elem_id="hidden_paper_id_trigger")
    js_activate_panel_trigger = gr.Button("Activate Panel Trigger", visible=False, elem_id="hidden_activate_panel_trigger")
    # --- UI Structure ---
    gr.Markdown("# Prof. Huang Paper Search & Chat")
    gr.Markdown(
        "Search for papers using keywords, title, author, or ID. Click 'Chat' on a result to load its PDF into ChatPDF and start a conversation.")
    with gr.Tabs(elem_id="main_app_tabs") as main_tabs:
        # --- Tab 1: Search & Chat ---
        with gr.TabItem("Search & Chat", id=0):
            # <<< ADJUSTED COLUMN SCALES (e.g., 3:2 ratio) >>>
            with gr.Row(equal_height=False, variant='compact', elem_id="search-chat-row"):
                # --- Left Column: Search Area ---
                # <<< SCALE ADJUSTED >>>
                with gr.Column(scale=3, min_width=600, elem_id="search-column"):
                    # --- Wrapper specifically for controls styling ---
                    with gr.Blocks(elem_id="search-controls-wrapper"):
                        gr.Markdown("### Search Papers")
                        with gr.Row():
                            query = gr.Textbox(label="Search Query", placeholder="Enter keywords...", elem_id="search-query-input", scale=3)
                            num_papers = gr.Slider(minimum=1, maximum=MAX_RESULTS, value=10, step=1, label="# Results",
                                                   elem_id="search-num-results", scale=2)
                        with gr.Row():
                            search_button = gr.Button("Search", variant="primary", elem_id="search-button", scale=1)
                            aggregate_button = gr.Button("Aggregate Analysis", elem_id="aggregate-button", scale=1)
                            # Add the theme toggle button
                    with gr.Blocks(elem_id = "visualizations-wrapper"):
                        gr.Markdown("### Visualization Results")
                        visualization_results_html = gr.HTML(
                            """<p>Click 'Aggregate Analysis' to see visualizations.</p>""",
                            elem_id="visualization-results-area"
                        )
                        status_output = gr.Textbox(label="Search Status", interactive=False, lines=1,
                                                   elem_id="search-status-output")
                    # --- End Controls Wrapper ---
                    # --- Search Results Section ---
                    gr.Markdown("### Search Results", elem_id="search-results-header")
                    search_results_html = gr.HTML("<p>Enter a query and click 'Search' to see results.</p>",
                                                  elem_id="search-results-area")
                # --- Right Column: Chat Panel (Sticky) ---
                # <<< SCALE & MIN-WIDTH ADJUSTED >>>
                with gr.Column(scale=2, min_width=600, elem_id="chat-panel-column"):  # Increased min-width
                    gr.Markdown("### Chat Panel")
                    current_paper_display = gr.Markdown("Click 'Chat' on a paper to begin.", elem_id="chat-paper-display")
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        # <<< REMOVED EXPLICIT HEIGHT PARAMETER >>>
                        elem_id="chatbot-display",
                        show_label=False,
                        bubble_full_width=False,
                        render_markdown=True,
                        show_copy_button=True,
                        type='messages'
                    )
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about the selected paper...",
                        elem_id="chat-input-message",
                        show_label=False,
                        lines=2
                    )
                    with gr.Row(elem_id="chat-button-row"):
                        send_btn = gr.Button("Send", variant="primary", elem_id="chat-send-button")
                        clear_chat_btn = gr.Button("Clear Chat", elem_id="chat-clear-button")
                    chat_status = gr.Textbox(label="Chat Status", interactive=False, elem_id="chat-status-textbox",
                                             lines=1)
        # --- Tab 2: API Configuration ---
        with gr.TabItem("API Configuration", id=1):
            gr.Markdown("### Configure ChatPDF API Key")
            gr.Markdown(
                "Enter your ChatPDF API Key (starting with `sec_`). This key is required for the chat functionality and is stored in memory only for your current session.")
            # Use a lambda to get the current value from state for the input
            api_key_input = gr.Textbox(
                label="ChatPDF API Key",
                value=lambda: api_key_state.value,  # Dynamically get value from state
                type="password",
                placeholder="Enter your key (e.g., sec_xxxxxxxx...)"
            )
            def save_config(api_key_val):
                """Saves API key to state and updates global variable"""
                global api_key_global  # Allow modification of global
                if not api_key_val or not isinstance(api_key_val, str) or not api_key_val.startswith("sec_"):
                    logger.warning(f"Invalid API Key format provided: '{str(api_key_val)[:5]}...'")
                    # Return error message and DO NOT update the state
                    return "Error: Invalid API Key format. Must be a string starting with 'sec_'.", gr.update()  # gr.update() keeps previous state value
                else:
                    # Key looks valid, update state and global variable
                    api_key_global = api_key_val
                    logger.info(f"ChatPDF API Key updated in session state (Prefix: '{api_key_val[:5]}...').")
                    # Return success message and the valid key to update the state
                    return "Configuration Saved Successfully for this session!", api_key_val
            save_button = gr.Button("Save Configuration", variant="primary")
            config_status = gr.Textbox(label="Status", interactive=False)
            # Connect save button: input is the textbox, outputs update status and the state
            save_button.click(
                fn=save_config,
                inputs=[api_key_input],
                outputs=[config_status, api_key_state]  # Update status AND the api_key_state
            )
            # Optional: If state changes (e.g., via save_config), update the global var too
            # This might be redundant if save_config already does it, but ensures consistency
            # api_key_state.change(lambda x: globals().update(api_key_global=x), inputs=api_key_state)
    # --- Wire UI elements to Python functions ---
    # Search Actions
    search_button.click(
        fn=search_and_display,
        inputs=[query, num_papers],
        outputs=[search_results_html, status_output]
    )
    query.submit(  # Allow pressing Enter in query box to search
        fn=search_and_display,
        inputs=[query, num_papers],
        outputs=[search_results_html, status_output]
    )
    # Aggregate Analysis Action
    aggregate_button.click(
        fn=run_aggregate_analysis,
        inputs=[query],  # Pass the query string to filter analysis
        outputs=[visualization_results_html, status_output]
    )
    # Chat Activation (triggered by JavaScript)
    js_activate_panel_trigger.click(
        fn=handle_chat_activation,
        inputs=[js_paper_id_trigger, api_key_state],  # Pass API key from state
        outputs=[
            active_paper_id_state,  # Update active paper ID state
            active_source_id_state,  # Update active source ID state
            chatbot,  # Update chat history
            chat_status,  # Update chat status message
            current_paper_display  # Update display showing current paper
        ],
        show_progress="minimal"  # Show minimal progress indicator
    )
    # Send Chat Message Actions
    send_btn.click(
        fn=send_chat_message,
        inputs=[active_paper_id_state, active_source_id_state, msg, chatbot, api_key_state],  # Pass state
        outputs=[chatbot, msg, chat_status]  # Update chat, clear input msg, update status
    )  # .then(lambda: None, None, None, js = "_ => { setTimeout(() => { const chatbox = document.querySelector('#chatbot-display .wrap'); if(chatbox) chatbox.scrollTop = chatbox.scrollHeight; }, 100); }")
    # Removed JS scroll due to potential complexity / might not work reliably with streaming
    msg.submit(  # Allow pressing Enter in message box to send
        fn=send_chat_message,
        inputs=[active_paper_id_state, active_source_id_state, msg, chatbot, api_key_state],  # Pass state
        outputs=[chatbot, msg, chat_status]  # Update chat, clear input msg, update status
    )  # .then(lambda: None, None, None, js = "_ => { setTimeout(() => { const chatbox = document.querySelector('#chatbot-display .wrap'); if(chatbox) chatbox.scrollTop = chatbox.scrollHeight; }, 100); }")
    # Removed JS scroll
    # Clear Chat Action
    clear_chat_btn.click(
        fn=clear_chat,
        inputs=[active_source_id_state, api_key_state],  # Pass state needed for optional delete
        outputs=[
            active_paper_id_state,  # Reset state
            active_source_id_state,  # Reset state
            chatbot,  # Clear chat history
            chat_status,  # Update status
            current_paper_display  # Reset display
        ]
    )
# --- CSS ---
# Updated CSS V5: Chat panel width/visibility, search results position fix
css = """
/* --- General Styles --- */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    font-size: 16px;
    /* 移除 transform 缩放，保持默认100% */
    transform: none;
}
/* 保证容器加载全屏 */
.gradio-container {
    width: 100%;
    max-width: 100% !important;
    padding: 1rem;
    margin: 0 auto;
    box-sizing: border-box;
}
/* --- 搜索结果项的美化 --- */
.search-result-item {
    background-color: #fff;    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.search-result-item:not(:last-child) {
    border-bottom: 1px dashed #ddd;
    /* 横线隔开 */
    padding-bottom: 20px;
}
/* --- 内部内容的间距调整 --- */
.search-result-item .result-title {
    margin-bottom: 10px;
    font-size: 1.2em;
    font-weight: bold;
}
.search-result-item .result-details,
.search-result-item .result-abstract,
.search-result-item .result-actions {
    margin-bottom: 10px;
    line-height: 1.5;
}
/* --- 保证 Footer 显示 --- */
footer {
    display: block;
    width: 100%;
    padding: 1rem;
    text-align: center;
    font-size: 0.9em;
    color: #666;
}
/* 其余样式保持不变 */
#search-chat-row {
    display: flex !important;
    flex-wrap: nowrap;
    align-items: flex-start;
    gap: 25px;
    width: 100%;
    box-sizing: border-box;
}
#search-results-area {
    margin-top: 0;
    /* 使结果区域紧跟标题下方 */
    flex-grow: 1;
    width: 100%;
    max-height: 500px;
    /* 根据需要的高度调整 */
    box-sizing: border-box;
    overflow-y: auto;
    /* 开启垂直滚动 */
    min-height: 200px;
    padding-right: 10px;
}
.action-button.blue-button {
    background-color: #3498db;
    color: white !important;
    border: none;
    padding: 8px 14px;
    border-radius: 5px;
    cursor: pointer;
    text-decoration: none !important;
    display: inline-block;
    font-size: 14px;
    transition: background-color 0.2s ease, box-shadow 0.2s ease;
    line-height: 1.2;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.action-button.blue-button:hover {
    background-color: #2980b9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
}
"""
# --- JavaScript ---
# JavaScript for activating chat panel (remains the same, should still work)
javascript = """
<script>
    function activateChatPanel(paperId) {
        console.log(`--- activateChatPanel called. Paper ID: ${paperId} ---`);
        // Function to find Gradio's shadow root or use document if no shadow DOM
        function getGradioApp() {
            const grApp = document.querySelector("gradio-app");
            // Check if shadowRoot exists and is attached. If not, fall back to document.
            return (grApp && grApp.shadowRoot) ? grApp.shadowRoot : document;
        }
        try {
            const app = getGradioApp();
            if (!app) {
                console.error("Gradio root element not found.");
                return;
            }
            // Find the hidden textbox container and the actual textarea within it
            const textboxContainer = app.querySelector("#hidden_paper_id_trigger");
            // Gradio structure might change, target textarea specifically
            const paperIdTextarea = textboxContainer ? textboxContainer.querySelector("textarea") : null;
            if (!paperIdTextarea) {
                console.error("Could not find the textarea within #hidden_paper_id_trigger. Structure might have changed.");
                // Fallback: Try finding by elem_id directly if structure is flat
                const directTextarea = app.querySelector("textarea[elem_id='hidden_paper_id_trigger']");
                if (!directTextarea) {
                    console.error("Fallback textarea search failed.");
                    return;
                }
                paperIdTextarea = directTextarea; // Use fallback if found
            }
            // Set value and dispatch events to notify Gradio backend
            paperIdTextarea.value = paperId;
            paperIdTextarea.dispatchEvent(new Event('input', {bubbles: true}));
            paperIdTextarea.dispatchEvent(new Event('change', {bubbles: true})); // Change event might be important
            console.log(`Set hidden textarea value to: "${paperId}".`);
            // Find the hidden button container and the actual button within it
            const buttonElementOrContainer = app.querySelector("#hidden_activate_panel_trigger");
            // Gradio structure might involve a wrapper, target the button specifically
            let activateButton = buttonElementOrContainer ? (buttonElementOrContainer.tagName === 'BUTTON' ? buttonElementOrContainer : buttonElementOrContainer.querySelector("button")) : null;
            if (!activateButton) {
                console.error("Cannot find button within #hidden_activate_panel_trigger container.");
                // Fallback: Try finding button by elem_id directly
                const directButton = app.querySelector("button[elem_id='hidden_activate_panel_trigger']");
                if (!directButton) {
                    console.error("Fallback button search failed.");
                    return;
                }
                activateButton = directButton; // Use fallback if found
            }
            activateButton.click();
            console.log("Clicked hidden activation button.");
            // --- Try to focus input and scroll chat ---
            // Use setTimeout to allow Gradio to potentially re-render elements after click
            setTimeout(() => {
                try {
                    const app = getGradioApp(); // Re-query root inside timeout
                    const chatPanelColumn = app.querySelector("#chat-panel-column"); // The sticky column
                    const chatbotDisplayWrapper = app.querySelector("#chatbot-display"); // Chatbot outer container
                    // Find the scrollable area within the chatbot component (often a div with class 'wrap' or similar)
                    // Inspect element in browser to confirm the actual scrollable element selector
                    const chatbotScrollableArea = chatbotDisplayWrapper ? chatbotDisplayWrapper.querySelector(".wrap") : chatbotDisplayWrapper; // Fallback to wrapper if .wrap not found
                    const messageInput = app.querySelector("#chat-input-message textarea");
                    // Scroll chat history to the top after activation
                    if (chatbotScrollableArea) {
                        chatbotScrollableArea.scrollTop = 0; // Scroll to top
                        console.log("Scrolled chat history to top.");
                    } else if (chatPanelColumn) {
                        chatPanelColumn.scrollTop = 0; // Fallback: scroll the whole panel top
                        console.log("Scrolled chat panel column to top (fallback).");
                    }
                    // Focus the message input field
                    if (messageInput) {
                        messageInput.focus();
                        console.log("Focused chat input message area.");
                    }
                } catch (e) {
                    console.warn("Minor error during post-activation scroll/focus:", e);
                }
            }, 300); // Delay allows UI updates
        } catch (error) {
            console.error("JavaScript error in activateChatPanel:", error);
            // Alert user if something goes wrong in JS activation logic
            alert("A JavaScript error occurred while trying to activate the chat panel. Please check the browser console for details.");
        }
    }
</script>
"""
# Inject CSS and JS into the Gradio app's head
demo.head = getattr(demo, 'head', '') + f"<style>{css}</style>" + javascript
# --- Launch the App ---
if __name__ == "__main__":
    # Initial Checks on Launch
    if not os.path.exists("all_papers.json"):
        logger.warning("----------------------------------------------------")
        logger.warning("WARNING: 'all_papers.json' not found.")
        logger.warning("Search functionality will not return any results.")
        logger.warning(
            "Please ensure the file exists in the same directory and contains JSON objects (one per line).")
        try:  # Attempt to create an empty file
            with open("all_papers.json", "w", encoding="utf-8") as f:
                f.write("")  # Write empty string
            logger.info("Created an empty 'all_papers.json'. You need to populate it with data.")
        except Exception as e:
            logger.error(f"Failed to create empty 'all_papers.json': {e}")
        logger.warning("----------------------------------------------------")
    # Check initial API Key status
    current_key = api_key_state.value  # Get initial key from state
    if not current_key or not current_key.startswith("sec_"):
        logger.warning("----------------------------------------------------")
        logger.warning("WARNING: ChatPDF API Key is missing or invalid (must start with 'sec_').")
        logger.warning(
            "The key might be missing from the environment variable 'CHATPDF_API_KEY' or hasn't been configured yet.")
        logger.warning(
            "Chat functionality will fail until a valid key is provided via the 'API Configuration' tab.")
        logger.warning(f"Current key status: {'Not set' if not current_key else 'Invalid format'}")
        logger.warning("----------------------------------------------------")
    else:
        logger.info(f"ChatPDF API Key loaded initially (Prefix: '{current_key[:5]}...'). Key will be managed by session state.")
    logger.info("Starting Paper Search & Chat application...")
    try:
        # queue() is important for handling multiple users and streaming responses
        demo.queue().launch(
            server_name="0.0.0.0",  # Listen on all network interfaces
            server_port=7865,
            share=False,  # Set to True to get a public Gradio link (use with caution)
            debug=False  # Set to True for Gradio's debug mode (more verbose console output)
        )
    except Exception as e:
        logger.critical(f"Gradio application failed to launch: {e}", exc_info=True)
        sys.exit(1)  # Exit if launch fails