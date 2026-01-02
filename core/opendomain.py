import requests
import json
import logging
import time
from uuid import uuid4
from . import config

logger = logging.getLogger(__name__)


def _post_with_retries(url: str, headers: dict, payload: dict, timeout: int, attempts: int = 3, backoff: float = 0.5):
    """Post request with retry logic."""
    last_exc: Exception | None = None
    session = requests.Session()
    for attempt in range(1, attempts + 1):
        try:
            start = time.perf_counter()
            resp = session.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info('POST %s attempt=%d status=%s elapsed_ms=%.1f', url, attempt, resp.status_code, elapsed_ms)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            # Log response body for debugging
            try:
                error_detail = exc.response.text
                logger.warning('POST %s failed attempt=%d status=%s error=%s', url, attempt, exc.response.status_code, error_detail)
            except:
                logger.warning('POST %s failed attempt=%d error=%s', url, attempt, exc)
            if attempt < attempts:
                time.sleep(backoff * attempt)
        except Exception as exc:
            last_exc = exc
            logger.warning('POST %s failed attempt=%d error=%s', url, attempt, exc)
            if attempt < attempts:
                time.sleep(backoff * attempt)
    raise last_exc


def make_request(question_text: str = "What is the capital of Texas?") -> requests.Response:
    """Call the Groq API with a question and return the response object.
    
    This function uses the Groq API for fast LLM inference.
    """
    cid = str(uuid4())
    logger.info(f"cid={cid} Making Groq LLM request for question: {question_text}")
    
    # Check if API key is configured
    if not config.get('GROQ_API_KEY'):
        raise RuntimeError('GROQ_API_KEY must be set in the environment')
    
    url = config['GROQ_API_URL']
    
    payload = {
        "model": config['GROQ_MODEL'],
        "messages": [
            {
                "role": "user",
                "content": question_text
            }
        ],
        "max_tokens": config['GROQ_MAX_TOKENS'],
        "temperature": config['GROQ_TEMPERATURE']
    }
    
    headers = {
        'Authorization': f'Bearer {config["GROQ_API_KEY"]}',
        'Content-Type': 'application/json'
    }
    
    response = _post_with_retries(url, headers=headers, payload=payload, timeout=30)
    logger.info(f"cid={cid} Groq API response status: {response.status_code}")
    return response


def ask_with_context(question_text: str, contexts: list[str], system_prompt: str | None = None) -> str:
    """Call the Groq API with retrieved context chunks and return text answer.
    
    Args:
        question_text: The user's question
        contexts: List of context chunks retrieved from vector search
        system_prompt: Optional custom system prompt
        
    Returns:
        The generated answer as a string
    """
    cid = str(uuid4())
    logger.info(
        f"cid={cid} Calling ask_with_context question_len=%d contexts=%d",
        len(question_text or ""),
        len(contexts)
    )
    
    # Check if API key is configured
    if not config.get('GROQ_API_KEY'):
        raise RuntimeError('GROQ_API_KEY must be set in the environment')
    
    url = config['GROQ_API_URL']
    
    # Build context block from retrieved chunks
    context_block = "\n\n".join(contexts)
    
    # Default system prompt for RAG
    sys_msg = system_prompt or (
        "You are a helpful AI assistant answering questions based ONLY on the provided context chunks. "
        "Rules:\n"
        "1. Do not invent facts or use information outside the provided context.\n"
        "2. If the context is insufficient to answer the question, clearly state that you don't have enough information.\n"
        "3. Keep answers concise and relevant (2-5 sentences unless more detail is needed).\n"
        "4. When multiple chunks support your answer, cite them inline as [C1], [C2], etc., using their order in the context.\n"
        "5. Be accurate and precise in your responses."
    )
    
    # Build user message with question and context
    user_text = (
        f"Question: {question_text}\n\n"
        f"Context chunks (C1 through C{len(contexts)}):\n"
        f"{context_block}\n\n"
        "Please provide a clear answer based on the context above, and cite relevant chunks using [C1], [C2], etc."
    )
    
    payload = {
        "model": config['GROQ_MODEL'],
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": config['GROQ_MAX_TOKENS'],
        "temperature": config['GROQ_TEMPERATURE']
    }
    
    headers = {
        'Authorization': f'Bearer {config["GROQ_API_KEY"]}',
        'Content-Type': 'application/json'
    }
    
    try:
        logger.debug("cid=%s Groq payload: %s", cid, json.dumps(payload)[:2000])
        resp = _post_with_retries(url, headers=headers, payload=payload, timeout=45)
        data = resp.json()
        logger.info(f"cid={cid} Groq context API response status: {resp.status_code}")
        
        try:
            logger.debug("cid=%s Groq raw response (truncated): %s", cid, json.dumps(data)[:2000])
        except Exception:
            logger.debug("cid=%s Groq raw response not JSON-decodable", cid)
        
        # Extract the answer from Groq's response
        # Groq API follows OpenAI-compatible format
        try:
            if 'choices' in data and len(data['choices']) > 0:
                first_choice = data['choices'][0]
                if 'message' in first_choice:
                    message = first_choice['message']
                    if 'content' in message:
                        return message['content']
            
            # Fallback: return the entire response as JSON if structure is unexpected
            logger.warning(f"cid={cid} Unexpected Groq response structure")
            return json.dumps(data, indent=2)
            
        except Exception as e:
            logger.error(f"cid={cid} Error extracting Groq answer: {e}")
            return json.dumps(data, indent=2)
            
    except Exception as e:
        logger.error(f"cid={cid} Error in ask_with_context: {e}")
        raise


if __name__ == '__main__':
    """Test the Groq API integration."""
    try:
        print("Testing Groq API...")
        resp = make_request("What is machine learning?")
        print(f"Status: {resp.status_code}")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to set GROQ_API_KEY in your .env file")

# Made with Bob
