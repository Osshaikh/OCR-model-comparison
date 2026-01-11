"""
OCR Model Comparison Backend Server
Handles API calls to multiple OCR models and returns results with metrics
"""

import os
import base64
import time
import re
import logging
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from dotenv import load_dotenv

# Setup logging to console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# Rate limiting: 4 requests per minute per IP address
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],  # No default limit on all routes
    storage_uri="memory://",  # In-memory storage (resets on restart)
)


@app.errorhandler(429)
def ratelimit_handler(e):
    """Custom error response for rate limit exceeded"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "You can only make 4 OCR requests per minute. Please wait and try again.",
        "retry_after": e.description
    }), 429


@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)


@app.route('/api/check-deployments', methods=['GET'])
def check_deployments():
    """Check available Azure deployments for debugging"""
    endpoint = os.getenv("MISTRAL_ENDPOINT")
    api_key = os.getenv("MISTRAL_API_KEY")
    
    # Try to list deployments
    try:
        url = f"{endpoint}/openai/deployments?api-version=2024-05-01-preview"
        headers = {"api-key": api_key}
        response = requests.get(url, headers=headers, timeout=30)
        return jsonify({
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# Model configurations
MODELS = {
    "mistral-large-3": {
        "name": "Mistral Large 3",
        "endpoint": "https://sbi-llama.openai.azure.com/openai/v1/chat/completions",
        "api_key": os.getenv("GPT52_API_KEY"),
        "model_name": "Mistral-Large-3",
        "pricing": {
            "input_per_1k": 0.002,   # $ per 1K input tokens
            "output_per_1k": 0.006   # $ per 1K output tokens
        }
    },
    "deepseek-ocr-gpu": {
        "name": "DeepSeek OCR (Azure GPU)",
        "endpoint": os.getenv("DEEPSEEK_OCR_GPU_ENDPOINT", "https://deepseek-ocr-gpu-app.calmriver-e92c6207.westus.azurecontainerapps.io") + "/api/generate",
        "pricing": {
            "input_per_1k": 0.002,  # Azure Container Apps A100 GPU cost estimate
            "output_per_1k": 0.002   # Azure Container Apps A100 GPU cost estimate
        }
    },
    "gpt-5.2": {
        "name": "GPT-5.2",
        "endpoint": "https://sbi-llama.openai.azure.com/openai/deployments/gpt-5.2/chat/completions?api-version=2024-10-21",
        "api_key": os.getenv("GPT52_API_KEY"),
        "pricing": {
            "input_per_1k": 0.01,    # $ per 1K input tokens
            "output_per_1k": 0.03    # $ per 1K output tokens
        }
    },
    "gemini-3-flash": {
        "name": "Gemini 3 Flash",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "pricing": {
            "input_per_1k": 0.00001,  # $ per 1K input tokens (very cheap)
            "output_per_1k": 0.00004  # $ per 1K output tokens
        }
    },
    "azure-doc-intelligence": {
        "name": "Azure Doc Intelligence",
        "endpoint": os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", ""),
        "api_key": os.getenv("AZURE_DOC_INTELLIGENCE_KEY", ""),
        "pricing": {
            "input_per_1k": 0.001,  # $ per page (approximated to tokens)
            "output_per_1k": 0.0    # No output token cost - extraction based
        }
    }
}

# OCR System prompt
OCR_SYSTEM_PROMPT = """You are an OCR (Optical Character Recognition) assistant. 
Extract ALL text from the provided image accurately. 
Preserve the original formatting, line breaks, and structure as much as possible.
If the image contains tables, try to maintain the table structure.
If text is unclear or partially visible, indicate with [unclear] or [partial].
Return ONLY the extracted text without any additional commentary."""


def estimate_tokens(text):
    """Estimate token count (roughly 4 characters per token)"""
    return len(text) // 4


def calculate_quality_score(text):
    """
    Calculate a quality score based on various text metrics
    Returns a score from 0-100
    """
    if not text or len(text.strip()) == 0:
        return 0
    
    score = 0
    
    # Check for echoed system prompt / instruction text (MAJOR PENALTY)
    prompt_indicators = [
        'preserve the original formatting',
        'line breaks, and structure',
        'if the image contains tables',
        'maintain the table structure',
        'indicate with [unclear]',
        'without any additional commentary',
        'optical character recognition',
        'extract all text from',
        'you are an ocr',
        'as an ai',
        'i cannot',
        'i\'m unable to',
        'i am unable to',
        'sorry, but',
        'i don\'t see',
        'there is no image',
        'no image provided',
        'please provide',
        'i cannot process',
        # New indicators for instruction-like output
        'use a consistent font',
        'throughout the transcription',
        'include any annotations',
        'notes in the margins',
        'ensure accuracy',
        'double-check',
        'proofread',
        'make sure to',
        'remember to',
        'don\'t forget',
        'be careful',
        'pay attention',
        'here are some tips',
        'follow these steps',
        'instructions:',
        'guidelines:',
        'note:',
        'important:',
        'tip:',
    ]
    
    text_lower = text.lower()
    prompt_echo_count = sum(1 for indicator in prompt_indicators if indicator in text_lower)
    if prompt_echo_count >= 2:
        # Severe penalty for echoing prompts/instructions
        return max(5, 20 - (prompt_echo_count * 5))
    
    # 1. Text length score (up to 25 points)
    text_length = len(text)
    if text_length > 0:
        length_score = min(25, text_length / 40)  # 1000 chars = 25 points
        score += length_score
    
    # 2. Word count score (up to 25 points)
    words = text.split()
    word_count = len(words)
    if word_count > 0:
        word_score = min(25, word_count / 8)  # 200 words = 25 points
        score += word_score
    
    # 3. Average word length (up to 15 points) - indicates real words vs gibberish
    if word_count > 0:
        avg_word_length = sum(len(w) for w in words) / word_count
        if 3 <= avg_word_length <= 10:  # Reasonable word length
            score += 15
        elif 2 <= avg_word_length <= 12:
            score += 10
        else:
            score += 5
    
    # 4. Punctuation presence (up to 10 points) - indicates structured text
    punctuation_count = len(re.findall(r'[.,!?;:\-\(\)]', text))
    if punctuation_count > 0:
        punct_ratio = punctuation_count / max(1, word_count)
        if 0.05 <= punct_ratio <= 0.3:
            score += 10
        elif punct_ratio > 0:
            score += 5
    
    # 5. Line structure (up to 10 points)
    lines = text.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    if len(non_empty_lines) > 1:
        score += min(10, len(non_empty_lines))
    
    # 6. No error indicators (up to 15 points)
    error_indicators = ['[unclear]', '[partial]', 'error', 'cannot read', 'unable to']
    has_errors = any(indicator.lower() in text.lower() for indicator in error_indicators)
    if not has_errors:
        score += 15
    else:
        score += 5
    
    # Penalty for single prompt echo indicator found
    if prompt_echo_count == 1:
        score = max(10, score - 30)
    
    return min(100, round(score))


def call_mistral_ocr(image_base64, image_type):
    """Call Mistral Large 3 API via Azure for OCR"""
    debug_logs = []
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Starting Mistral Large 3 OCR request")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Image size: {len(image_base64)} chars (~{len(image_base64) * 3 // 4 // 1024} KB)")
    
    logger.info("[Mistral Large 3] Starting OCR request...")
    config = MODELS["mistral-large-3"]
    
    headers = {
        "Content-Type": "application/json",
        "api-key": config["api_key"]
    }
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Endpoint: {config['endpoint']}")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Model: {config.get('model_name', 'Mistral-Large-3')}")
    
    payload = {
        "model": config.get("model_name", "Mistral-Large-3"),
        "messages": [
            {
                "role": "system",
                "content": OCR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract all text from this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.1
    }
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Sending POST request...")
    start_time = time.time()
    logger.info(f"[Mistral Large 3] Sending request to {config['endpoint']}...")
    try:
        response = requests.post(config["endpoint"], headers=headers, json=payload, timeout=120)
    except requests.exceptions.Timeout:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Request timed out after 120 seconds")
        logger.error("[Mistral Large 3] Request timed out after 120 seconds")
        raise Exception("Mistral API request timed out")
    except Exception as e:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}")
        logger.error(f"[Mistral Large 3] Request failed: {str(e)}")
        raise
    response_time = time.time() - start_time
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response received in {response_time:.2f}s")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] HTTP Status: {response.status_code}")
    logger.info(f"[Mistral Large 3] Response received in {response_time:.2f}s, status: {response.status_code}")
    
    if response.status_code != 200:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: API returned {response.status_code}")
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response: {response.text[:500]}")
        logger.error(f"[Mistral Large 3] API error: {response.status_code} - {response.text[:500]}")
        raise Exception(f"Mistral API error: {response.status_code} - {response.text}")
    
    result = response.json()
    text = result["choices"][0]["message"]["content"]
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text length: {len(text)} chars")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Input tokens: {result.get('usage', {}).get('prompt_tokens', 'N/A')}")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Output tokens: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
    preview = text[:200].replace('\n', ' ') if text else "(empty)"
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text preview: {preview}...")
    
    return {
        "text": text,
        "response_time": response_time,
        "input_tokens": result.get("usage", {}).get("prompt_tokens", estimate_tokens(image_base64) // 10),
        "output_tokens": result.get("usage", {}).get("completion_tokens", 0),
        "debug_logs": debug_logs
    }


def call_deepseek_ocr(image_base64, image_type):
    """Call DeepSeek OCR via Ollama (Azure or Local)"""
    logger.info("[DeepSeek OCR] Starting OCR request...")
    config = MODELS["deepseek-ocr"]
    
    payload = {
        "model": "deepseek-ocr",
        "prompt": OCR_SYSTEM_PROMPT + "\n\nPlease extract all text from this image.",
        "images": [image_base64],
        "stream": False
    }
    
    start_time = time.time()
    logger.info(f"[DeepSeek OCR] Sending request to {config['endpoint']}...")
    try:
        response = requests.post(config["endpoint"], json=payload, timeout=300)
    except requests.exceptions.Timeout:
        logger.error("[DeepSeek OCR] Request timed out after 300 seconds")
        raise Exception("DeepSeek OCR request timed out")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[DeepSeek OCR] Connection error: {str(e)}")
        raise Exception(f"Cannot connect to DeepSeek OCR endpoint. Error: {str(e)}")
    except Exception as e:
        logger.error(f"[DeepSeek OCR] Request failed: {str(e)}")
        raise
    response_time = time.time() - start_time
    logger.info(f"[DeepSeek OCR] Response received in {response_time:.2f}s, status: {response.status_code}")
    
    if response.status_code != 200:
        logger.error(f"[DeepSeek OCR] API error: {response.status_code} - {response.text[:500]}")
        raise Exception(f"DeepSeek OCR error: {response.status_code} - {response.text}")
    
    result = response.json()
    text = result.get("response", "")
    
    return {
        "text": text,
        "response_time": response_time,
        "input_tokens": estimate_tokens(image_base64) // 10,
        "output_tokens": estimate_tokens(text)
    }


def call_deepseek_ocr_gpu(image_base64, image_type):
    """Call DeepSeek OCR via Ollama on Azure Container Apps with A100 GPU"""
    debug_logs = []
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Starting DeepSeek OCR GPU request")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Image size: {len(image_base64)} chars (~{len(image_base64) * 3 // 4 // 1024} KB)")
    
    logger.info("[DeepSeek OCR GPU] Starting OCR request...")
    config = MODELS["deepseek-ocr-gpu"]
    
    # Simple prompt that produces cleanest output
    prompt_text = "OCR:"
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Prompt: {prompt_text}")
    
    payload = {
        "model": "deepseek-ocr",
        "prompt": prompt_text,
        "images": [image_base64],
        "stream": False,
        "options": {
            "num_predict": 4096,
            "temperature": 0.2
        }
    }
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Endpoint: {config['endpoint']}")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Model: deepseek-ocr")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Options: num_predict=4096, temperature=0.2")
    
    start_time = time.time()
    logger.info(f"[DeepSeek OCR GPU] Sending request to {config['endpoint']}...")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Sending POST request...")
    
    try:
        response = requests.post(config["endpoint"], json=payload, timeout=300)
    except requests.exceptions.Timeout:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Request timed out after 300 seconds")
        logger.error("[DeepSeek OCR GPU] Request timed out after 300 seconds")
        raise Exception("DeepSeek OCR GPU request timed out after 300 seconds. Large images may take longer to process.", debug_logs)
    except requests.exceptions.ConnectionError as e:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Connection error - {str(e)}")
        logger.error(f"[DeepSeek OCR GPU] Connection error: {str(e)}")
        raise Exception(f"Cannot connect to DeepSeek OCR GPU endpoint. Error: {str(e)}", debug_logs)
    except Exception as e:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}")
        logger.error(f"[DeepSeek OCR GPU] Request failed: {str(e)}")
        raise
        
    response_time = time.time() - start_time
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response received in {response_time:.2f}s")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] HTTP Status: {response.status_code}")
    logger.info(f"[DeepSeek OCR GPU] Response received in {response_time:.2f}s, status: {response.status_code}")
    
    if response.status_code != 200:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: API returned {response.status_code}")
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response body: {response.text[:500]}")
        logger.error(f"[DeepSeek OCR GPU] API error: {response.status_code} - {response.text[:500]}")
        raise Exception(f"DeepSeek OCR GPU error: {response.status_code} - {response.text}")
    
    result = response.json()
    text = result.get("response", "")
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response keys: {list(result.keys())}")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text length: {len(text)} chars")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Word count: {len(text.split())} words")
    
    # Log raw API response metadata
    if "total_duration" in result:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Ollama total_duration: {result.get('total_duration', 0) / 1e9:.2f}s")
    if "load_duration" in result:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Ollama load_duration: {result.get('load_duration', 0) / 1e9:.2f}s")
    if "prompt_eval_count" in result:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Ollama prompt_eval_count: {result.get('prompt_eval_count', 0)}")
    if "eval_count" in result:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Ollama eval_count (output tokens): {result.get('eval_count', 0)}")
    if "done_reason" in result:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Done reason: {result.get('done_reason', 'unknown')}")
    
    # Preview of extracted text
    preview = text[:200].replace('\n', ' ') if text else "(empty)"
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text preview: {preview}...")
    
    return {
        "text": text,
        "response_time": response_time,
        "input_tokens": estimate_tokens(image_base64) // 10,
        "output_tokens": estimate_tokens(text),
        "debug_logs": debug_logs
    }


def call_gpt52_ocr(image_base64, image_type):
    """Call GPT-5.2 API via Azure OpenAI"""
    debug_logs = []
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Starting GPT-5.2 OCR request")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Image size: {len(image_base64)} chars (~{len(image_base64) * 3 // 4 // 1024} KB)")
    
    logger.info("[GPT-5.2] Starting OCR request...")
    config = MODELS["gpt-5.2"]
    
    headers = {
        "Content-Type": "application/json",
        "api-key": config["api_key"]
    }
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Endpoint: {config['endpoint']}")
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": OCR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract all text from this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_completion_tokens": 4096,
        "temperature": 0.1
    }
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Sending POST request...")
    start_time = time.time()
    logger.info(f"[GPT-5.2] Sending request to {config['endpoint']}...")
    try:
        response = requests.post(config["endpoint"], headers=headers, json=payload, timeout=120)
    except requests.exceptions.Timeout:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Request timed out after 120 seconds")
        logger.error("[GPT-5.2] Request timed out after 120 seconds")
        raise Exception("GPT-5.2 API request timed out")
    except Exception as e:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}")
        logger.error(f"[GPT-5.2] Request failed: {str(e)}")
        raise
    response_time = time.time() - start_time
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response received in {response_time:.2f}s")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] HTTP Status: {response.status_code}")
    logger.info(f"[GPT-5.2] Response received in {response_time:.2f}s, status: {response.status_code}")
    
    if response.status_code != 200:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: API returned {response.status_code}")
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response: {response.text[:500]}")
        logger.error(f"[GPT-5.2] API error: {response.status_code} - {response.text[:500]}")
        raise Exception(f"GPT-5.2 API error: {response.status_code} - {response.text}")
    
    result = response.json()
    text = result["choices"][0]["message"]["content"]
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text length: {len(text)} chars")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Input tokens: {result.get('usage', {}).get('prompt_tokens', 'N/A')}")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Output tokens: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
    preview = text[:200].replace('\n', ' ') if text else "(empty)"
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text preview: {preview}...")
    
    return {
        "text": text,
        "response_time": response_time,
        "input_tokens": result.get("usage", {}).get("prompt_tokens", 0),
        "output_tokens": result.get("usage", {}).get("completion_tokens", 0),
        "debug_logs": debug_logs
    }


def call_gemini_ocr(image_base64, image_type):
    """Call Gemini 3 Flash API for OCR"""
    debug_logs = []
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Starting Gemini 3 Flash OCR request")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Image size: {len(image_base64)} chars (~{len(image_base64) * 3 // 4 // 1024} KB)")
    
    logger.info("[Gemini 3 Flash] Starting OCR request...")
    config = MODELS["gemini-3-flash"]
    
    # Gemini uses different API format
    url = f"{config['endpoint']}?key={config['api_key']}"
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Endpoint: {config['endpoint']}")
    
    payload = {
        "contents": [{
            "parts": [
                {"text": OCR_SYSTEM_PROMPT + "\n\nPlease extract all text from this image."},
                {"inline_data": {
                    "mime_type": image_type,
                    "data": image_base64
                }}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096
        }
    }
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Sending POST request...")
    start_time = time.time()
    logger.info(f"[Gemini 3 Flash] Sending request to {config['endpoint']}...")
    try:
        response = requests.post(url, json=payload, timeout=120)
    except requests.exceptions.Timeout:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Request timed out after 120 seconds")
        logger.error("[Gemini 3 Flash] Request timed out after 120 seconds")
        raise Exception("Gemini API request timed out")
    except Exception as e:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}")
        logger.error(f"[Gemini 3 Flash] Request failed: {str(e)}")
        raise
    response_time = time.time() - start_time
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response received in {response_time:.2f}s")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] HTTP Status: {response.status_code}")
    logger.info(f"[Gemini 3 Flash] Response received in {response_time:.2f}s, status: {response.status_code}")
    
    if response.status_code != 200:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: API returned {response.status_code}")
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response: {response.text[:500]}")
        logger.error(f"[Gemini 3 Flash] API error: {response.status_code} - {response.text[:500]}")
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
    
    result = response.json()
    
    # Extract text from Gemini response format
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    
    # Gemini provides usage metadata differently
    usage = result.get("usageMetadata", {})
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text length: {len(text)} chars")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Input tokens: {usage.get('promptTokenCount', 'N/A')}")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Output tokens: {usage.get('candidatesTokenCount', 'N/A')}")
    preview = text[:200].replace('\n', ' ') if text else "(empty)"
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text preview: {preview}...")
    
    return {
        "text": text,
        "response_time": response_time,
        "input_tokens": usage.get("promptTokenCount", estimate_tokens(image_base64) // 10),
        "output_tokens": usage.get("candidatesTokenCount", estimate_tokens(text)),
        "debug_logs": debug_logs
    }


def call_azure_doc_intelligence(image_base64, image_type):
    """Call Azure AI Document Intelligence for OCR using API key authentication"""
    debug_logs = []
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Starting Azure Doc Intelligence OCR request")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Image size: {len(image_base64)} chars (~{len(image_base64) * 3 // 4 // 1024} KB)")
    
    logger.info("[Azure Doc Intelligence] Starting OCR request...")
    config = MODELS["azure-doc-intelligence"]
    
    endpoint = config["endpoint"].rstrip('/')
    api_key = config["api_key"]
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Endpoint: {endpoint}")
    
    if not endpoint or "your-resource-name" in endpoint:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Endpoint not configured")
        raise Exception("Azure Document Intelligence not configured. Please set AZURE_DOC_INTELLIGENCE_ENDPOINT in .env file")
    
    if not api_key:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: API key not configured")
        raise Exception("Azure Document Intelligence API key not configured. Please set AZURE_DOC_INTELLIGENCE_KEY in .env file")
    
    # Use the prebuilt-read model for general OCR
    analyze_url = f"{endpoint}/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30"
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Using prebuilt-read model")
    
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Send base64 image
    payload = {
        "base64Source": image_base64
    }
    
    start_time = time.time()
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Sending POST request...")
    logger.info(f"[Azure Doc Intelligence] Sending request to {analyze_url}...")
    
    try:
        # Start the analysis (async operation)
        response = requests.post(analyze_url, headers=headers, json=payload, timeout=30)
    except requests.exceptions.Timeout:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Initial request timed out after 30 seconds")
        logger.error("[Azure Doc Intelligence] Initial request timed out after 30 seconds")
        raise Exception("Azure Document Intelligence request timed out")
    except Exception as e:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}")
        logger.error(f"[Azure Doc Intelligence] Request failed: {str(e)}")
        raise
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Initial response: HTTP {response.status_code}")
    
    if response.status_code != 202:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Expected 202, got {response.status_code}")
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Response: {response.text[:500]}")
        logger.error(f"[Azure Doc Intelligence] API error: {response.status_code} - {response.text[:500]}")
        raise Exception(f"Azure Document Intelligence error: {response.status_code} - {response.text}")
    
    # Get the operation location to poll for results
    operation_location = response.headers.get("Operation-Location")
    if not operation_location:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: No operation location returned")
        raise Exception("Azure Document Intelligence did not return operation location")
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Analysis started, polling for results...")
    logger.info(f"[Azure Doc Intelligence] Analysis started, polling for results...")
    
    # Poll for results using same API key header
    poll_headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }
    
    max_retries = 30  # 30 seconds max
    for i in range(max_retries):
        time.sleep(1)
        poll_response = requests.get(operation_location, headers=poll_headers, timeout=30)
        
        if poll_response.status_code != 200:
            debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Poll returned {poll_response.status_code}")
            logger.error(f"[Azure Doc Intelligence] Poll error: {poll_response.status_code}")
            raise Exception(f"Azure Document Intelligence poll error: {poll_response.status_code}")
        
        result = poll_response.json()
        status = result.get("status")
        
        if status == "succeeded":
            debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Analysis completed after {i+1} seconds")
            logger.info(f"[Azure Doc Intelligence] Analysis completed after {i+1} seconds")
            break
        elif status == "failed":
            error = result.get("error", {})
            debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Analysis failed - {error.get('message', 'Unknown')}")
            raise Exception(f"Azure Document Intelligence analysis failed: {error.get('message', 'Unknown error')}")
        # status is "running" - continue polling
    else:
        debug_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Polling timed out after 30 seconds")
        raise Exception("Azure Document Intelligence analysis timed out after 30 seconds")
    
    response_time = time.time() - start_time
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Total response time: {response_time:.2f}s")
    logger.info(f"[Azure Doc Intelligence] Response received in {response_time:.2f}s")
    
    # Extract text from the result
    analyze_result = result.get("analyzeResult", {})
    content = analyze_result.get("content", "")
    
    # Count pages for cost estimation
    pages = analyze_result.get("pages", [])
    page_count = len(pages) if pages else 1
    
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Pages detected: {page_count}")
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text length: {len(content)} chars")
    preview = content[:200].replace('\n', ' ') if content else "(empty)"
    debug_logs.append(f"[{time.strftime('%H:%M:%S')}] Text preview: {preview}...")
    
    return {
        "text": content,
        "response_time": response_time,
        "input_tokens": page_count * 1000,  # Approximate as 1000 tokens per page for cost
        "output_tokens": estimate_tokens(content),
        "debug_logs": debug_logs
    }


def calculate_cost(model_id, input_tokens, output_tokens):
    """Calculate the cost for a model based on token usage"""
    pricing = MODELS[model_id]["pricing"]
    input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
    return round(input_cost + output_cost, 6)


@app.route('/api/models', methods=['GET'])
def get_models():
    """Return list of available models"""
    return jsonify({
        "models": [
            {"id": model_id, "name": config["name"], "pricing": config["pricing"]}
            for model_id, config in MODELS.items()
        ]
    })


@app.route('/api/ocr', methods=['POST'])
@limiter.limit("4 per minute")
def process_ocr():
    """Process image with selected OCR models (rate limited: 4 req/min per IP)"""
    logger.info("="*60)
    logger.info("Received OCR request")
    data = request.json
    image_base64 = data.get('image')
    image_type = data.get('image_type', 'image/png')
    selected_models = data.get('models', list(MODELS.keys()))
    
    logger.info(f"Image type: {image_type}")
    logger.info(f"Image size: {len(image_base64) if image_base64 else 0} chars")
    logger.info(f"Selected models: {selected_models}")
    
    if not image_base64:
        logger.error("No image provided in request")
        return jsonify({"error": "No image provided"}), 400
    
    # Remove data URL prefix if present
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    results = {}
    
    # Process with each selected model
    model_functions = {
        "mistral-large-3": call_mistral_ocr,
        "deepseek-ocr-gpu": call_deepseek_ocr_gpu,
        "gpt-5.2": call_gpt52_ocr,
        "gemini-3-flash": call_gemini_ocr,
        "azure-doc-intelligence": call_azure_doc_intelligence
    }
    
    for model_id in selected_models:
        if model_id not in model_functions:
            logger.warning(f"Unknown model: {model_id}, skipping")
            continue
        
        logger.info(f"\n>>> Processing with model: {model_id}")
        try:
            result = model_functions[model_id](image_base64, image_type)
            logger.info(f"<<< {model_id} completed successfully")
            
            # Calculate metrics
            quality_score = calculate_quality_score(result["text"])
            cost = calculate_cost(model_id, result["input_tokens"], result["output_tokens"])
            
            results[model_id] = {
                "model_name": MODELS[model_id]["name"],
                "text": result["text"],
                "metrics": {
                    "response_time": round(result["response_time"], 2),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["input_tokens"] + result["output_tokens"],
                    "cost_usd": cost,
                    "quality_score": quality_score,
                    "char_count": len(result["text"]),
                    "word_count": len(result["text"].split())
                },
                "debug_logs": result.get("debug_logs", []),
                "status": "success"
            }
        except Exception as e:
            error_msg = str(e)
            # Extract debug logs if they were passed with the exception
            debug_logs = []
            if hasattr(e, 'args') and len(e.args) > 1:
                debug_logs = e.args[1] if isinstance(e.args[1], list) else []
                error_msg = e.args[0]
            
            logger.error(f"<<< {model_id} FAILED: {error_msg}")
            results[model_id] = {
                "model_name": MODELS[model_id]["name"],
                "text": "",
                "metrics": {
                    "response_time": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0,
                    "quality_score": 0,
                    "char_count": 0,
                    "word_count": 0
                },
                "debug_logs": debug_logs,
                "status": "error",
                "error": error_msg
            }
    
    # Calculate comparative scores
    if results:
        # Normalize scores for comparison
        max_quality = max((r["metrics"]["quality_score"] for r in results.values() if r["status"] == "success"), default=1)
        min_time = min((r["metrics"]["response_time"] for r in results.values() if r["status"] == "success" and r["metrics"]["response_time"] > 0), default=1)
        min_cost = min((r["metrics"]["cost_usd"] for r in results.values() if r["status"] == "success" and r["metrics"]["cost_usd"] > 0), default=0.001)
        
        for model_id, result in results.items():
            if result["status"] == "success":
                # Calculate overall score (weighted)
                quality_normalized = (result["metrics"]["quality_score"] / max_quality * 100) if max_quality > 0 else 0
                speed_normalized = (min_time / result["metrics"]["response_time"] * 100) if result["metrics"]["response_time"] > 0 else 0
                cost_normalized = (min_cost / result["metrics"]["cost_usd"] * 100) if result["metrics"]["cost_usd"] > 0 else 0
                
                # Weighted overall score: 50% quality, 30% speed, 20% cost
                overall_score = (quality_normalized * 0.5) + (speed_normalized * 0.3) + (cost_normalized * 0.2)
                result["metrics"]["overall_score"] = round(overall_score, 1)
    
    return jsonify({"results": results})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    print("Starting OCR Comparison Server...")
    print("Available models:", list(MODELS.keys()))
    app.run(debug=True, port=5000, use_reloader=False)
