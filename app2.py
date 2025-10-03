"""
OS Study Bot - Enhanced Flask Backend
====================================

A comprehensive AI-powered study assistant for Operating Systems course.
Features:
- Google Gemini AI integration for intelligent responses
- Question parsing and categorization by units
- Advanced page tracking and reference system
- Test evaluation with detailed feedback
- Q&A caching and recommendation system
- Multi-question support with differentiation tables

Author: OS Study Bot Team
Version: 2.0
"""

from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from google.generativeai import types
import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('os_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simple API Key Rotator
class SimpleAPIRotator:
    def __init__(self, config_file="api_keys_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.current_gemini_index = 0
        self.current_openai_index = 0
        self.gemini_keys = self.config.get("gemini_keys", [])
        self.openai_keys = self.config.get("openai_keys", [])
        logger.info(f"API Rotator initialized with {len(self.gemini_keys)} Gemini keys and {len(self.openai_keys)} OpenAI keys")
    
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading API config: {e}")
            return {"gemini_keys": [], "openai_keys": []}
    
    def get_next_gemini_key(self):
        if not self.gemini_keys:
            return None
        key_data = self.gemini_keys[self.current_gemini_index % len(self.gemini_keys)]
        self.current_gemini_index += 1
        return key_data["api_key"]
    
    def get_next_openai_key(self):
        if not self.openai_keys:
            return None
        key_data = self.openai_keys[self.current_openai_index % len(self.openai_keys)]
        self.current_openai_index += 1
        return key_data["api_key"]
    
    def get_status(self):
        return {
            "gemini_keys": len(self.gemini_keys),
            "openai_keys": len(self.openai_keys),
            "current_gemini_index": self.current_gemini_index,
            "current_openai_index": self.current_openai_index
        }

# Initialize API rotator
try:
    api_rotator = SimpleAPIRotator()
    API_ROTATOR_AVAILABLE = True
    logger.info("Simple API Rotator loaded successfully")
except Exception as e:
    logger.warning(f"API Rotator not available: {e}")
    API_ROTATOR_AVAILABLE = False
    api_rotator = None

def clean_markdown_formatting(text: str) -> str:
    """
    Remove all Markdown formatting and create clean, ChatGPT-style responses.
    
    Args:
        text (str): Input text with potential markdown formatting
        
    Returns:
        str: Cleaned text without markdown symbols, formatted like ChatGPT
    """
    if not text:
        return ""
    
    # Remove all markdown formatting
    cleaned = text
    
    # Remove headers (# ## ### etc.)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    
    # Remove bold and italic (**text** or *text*)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)  # **bold** -> bold
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)      # *italic* -> italic
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)      # __bold__ -> bold
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)        # _italic_ -> italic
    
    # Remove code blocks (```code```)
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)        # `code` -> code
    
    # Remove links [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    
    # Remove strikethrough ~~text~~ -> text
    cleaned = re.sub(r"~~([^~]+)~~", r"\1", cleaned)
    
    # Remove blockquotes > text -> text
    cleaned = re.sub(r"^>\s*", "", cleaned, flags=re.MULTILINE)
    
    # Remove horizontal rules (--- or ***)
    cleaned = re.sub(r"^[-*_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    
    # Clean up bullet points - convert various markdown bullets to simple dashes
    cleaned = re.sub(r"^[\s]*[•▪▫‣⁃]\s*", "- ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[\s]*\d+\.\s*", "- ", cleaned, flags=re.MULTILINE)  # Convert numbered lists to bullets
    cleaned = re.sub(r"^\s*\*\s*", "- ", cleaned, flags=re.MULTILINE)  # Convert asterisk bullets to dashes
    
    # Remove any remaining markdown symbols
    cleaned = re.sub(r"[*_#>`~]", "", cleaned)
    
    # Normalize whitespace but preserve structure
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)  # Multiple newlines to double newline
    cleaned = re.sub(r"^\s+|\s+$", "", cleaned, flags=re.MULTILINE)  # Trim each line
    cleaned = re.sub(r"[ \t]+", " ", cleaned)  # Multiple spaces/tabs to single space
    
    # Remove excessive punctuation
    cleaned = re.sub(r"[!]{2,}", "!", cleaned)
    cleaned = re.sub(r"[?]{2,}", "?", cleaned)
    cleaned = re.sub(r"\.{3,}", "...", cleaned)
    
    return cleaned.strip()

# Initialize Flask app with enhanced configuration
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # Preserve JSON key order
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# -------- Enhanced Gemini Configuration --------
API_KEY = "AIzaSyBFjCRQtXAop7Uh8Amjb2jPmT872PIFWFM"
genai.configure(api_key=API_KEY)

# Model configuration with fallback options
PRIMARY_MODEL = "gemini-2.0-flash-exp"
FALLBACK_MODEL = "gemini-1.5-pro"
CURRENT_MODEL = PRIMARY_MODEL

# Initialize Gemini client with enhanced settings
try:
    client = genai.GenerativeModel(CURRENT_MODEL)
    logger.info(f"Successfully initialized Gemini with model: {CURRENT_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {e}")
    client = None

# -------- Enhanced Content Loading System --------
FILE_PATH = r"C:\users\user\OneDrive\Desktop\OS_BOT\OS SYLLABUS INCOMPLETE.txt"

def load_syllabus_content(file_path: str) -> str:
    """
    Load syllabus content with comprehensive error handling and validation.
    
    Args:
        file_path (str): Path to the syllabus file
        
    Returns:
        str: Loaded content or fallback content
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Syllabus file not found at: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            raise ValueError("Syllabus file is empty")
        
        logger.info(f"Successfully loaded syllabus content ({len(content)} characters)")
        return content
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return get_fallback_content()
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error: {e}")
        return get_fallback_content()
    except Exception as e:
        logger.error(f"Unexpected error loading syllabus: {e}")
        return get_fallback_content()

def get_fallback_content() -> str:
    """Provide comprehensive fallback content for OS topics."""
    return """
    OPERATING SYSTEMS FUNDAMENTALS
    =============================
    
    1. INTRODUCTION TO OPERATING SYSTEMS
    An Operating System (OS) is system software that manages computer hardware and software resources.
    
    2. OPERATING SYSTEM SERVICES
    - Process Management
    - Memory Management
    - File System Management
    - Device Management
    - Security and Protection
    
    3. TYPES OF OPERATING SYSTEMS
    - Batch Operating Systems
    - Multiprogramming Systems
    - Time-Sharing Systems
    - Real-Time Systems
    - Distributed Systems
    
    4. PROCESS MANAGEMENT
    A process is a program in execution. Process management includes:
    - Process Creation and Termination
    - Process Scheduling
    - Process Synchronization
    - Inter-Process Communication
    
    5. MEMORY MANAGEMENT
    Memory management handles:
    - Memory Allocation
    - Virtual Memory
    - Paging and Segmentation
    - Memory Protection
    
    Reference: Operating Systems Course Material
    """

# Load syllabus content with enhanced error handling
SUBJECT_CONTENT = load_syllabus_content(FILE_PATH)

# --------- Enhanced Content Parsing with Advanced Page Tracking ---------
def parse_content_with_pages(text: str) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Enhanced content parser with comprehensive page tracking and section detection.
    
    Args:
        text (str): The syllabus content to parse
        
    Returns:
        Tuple[Dict[str, List[str]], Dict[str, int]]: (content_sections, page_mapping)
    """
    content_sections = {}
    page_mapping = {}
    
    lines = text.splitlines()
    current_page = 1
    current_unit = None
    current_section = ""
    section_depth = 0
    
    logger.info(f"Parsing {len(lines)} lines of content")
    
    for line_num, line in enumerate(lines, 1):
        original_line = line
        line = line.strip()
        
        if not line:
            continue
            
        # Enhanced page marker detection
        page_patterns = [
            r"Pg(\d+)END",           # Pg1END, Pg2END, etc.
            r"page\s*(\d+)",         # page 1, page 2
            r"p\.?\s*(\d+)",         # p. 1, p 2
            r"pg\.?\s*(\d+)",        # pg. 1, pg 2
            r"^(\d+)$",              # standalone numbers
            r"Page\s*(\d+)",         # Page 1, Page 2
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match and len(line) < 30:  # Short lines more likely to be page markers
                try:
                    new_page = int(match.group(1))
                    if new_page > current_page:  # Only advance page numbers
                        current_page = new_page
                        logger.debug(f"Found page marker: {line} -> Page {current_page}")
                    continue
                except (ValueError, IndexError):
                    pass
        
        # Enhanced unit and section detection
        unit_patterns = [
            r"^UNIT\s*(\d+)",                    # UNIT 1, UNIT 2
            r"^(\d+)\.?\s*DIFFERENT TYPES OF",   # 2. DIFFERENT TYPES OF...
            r"^(\d+)\.?\s*INTRODUCTION TO",      # 1. INTRODUCTION TO...
        ]
        
        for pattern in unit_patterns:
            unit_match = re.match(pattern, line, flags=re.IGNORECASE)
        if unit_match:
            unit_num = unit_match.group(1)
            current_unit = f"UT{unit_num}"
            current_section = line
            content_sections.setdefault(current_unit, [])
            page_mapping[line] = current_page
            logger.debug(f"Found unit header: {line} -> {current_unit}")
            break
        
        # Section depth tracking for better organization
        if line.startswith(("1.", "2.", "3.", "4.", "5.")):
            section_depth = 1
        elif line.startswith(("1.1", "1.2", "2.1", "2.2")):
            section_depth = 2
        elif line.startswith(("1.1.1", "1.1.2", "2.1.1")):
            section_depth = 3
        
        # Store content with comprehensive page mapping
        if current_unit or section_depth > 0:
            if current_unit:
                content_sections[current_unit].append(line)
                                
            # Map content to current page
            page_mapping[line] = current_page
            
            # Map content snippets for faster lookup
            for snippet_length in [30, 50, 100]:
                if len(line) > snippet_length:
                    snippet = line[:snippet_length]
            page_mapping[snippet] = current_page
    
            # Map individual words for keyword-based lookup
            if len(line) > 10:
                words = line.split()[:5]  # First 5 words
                for word in words:
                    if len(word) > 3:  # Only meaningful words
                        page_mapping[word.lower()] = current_page
    
    logger.info(f"Parsing complete: {len(content_sections)} sections, {len(page_mapping)} mappings")
    return content_sections, page_mapping

# Parse content and create page mapping
CONTENT_SECTIONS, PAGE_MAPPING = parse_content_with_pages(SUBJECT_CONTENT)

def find_page_number(content_text: str) -> int:
    """
    Enhanced page number finder with multiple matching strategies.
    
    Args:
        content_text (str): Text content to find page number for
        
    Returns:
        int: Most likely page number (1 if no match found)
    """
    if not content_text or not content_text.strip():
        return 1
    
    content_text = content_text.strip()
    content_lower = content_text.lower()
    
    # Strategy 1: Direct exact match
    if content_text in PAGE_MAPPING:
        return PAGE_MAPPING[content_text]
    
    # Strategy 2: Case-insensitive direct match
    for mapped_content, page_num in PAGE_MAPPING.items():
        if mapped_content.lower() == content_lower:
                return page_num
    
    # Strategy 3: Substring matching (content contains or is contained in mapped content)
    for mapped_content, page_num in PAGE_MAPPING.items():
        if len(mapped_content) > 10 and len(content_text) > 5:
            if (content_lower in mapped_content.lower() or 
                mapped_content.lower() in content_lower):
                return page_num
    
    # Strategy 4: Word-based similarity scoring
    content_words = set(content_lower.split())
    best_match_page = 1
    best_match_score = 0
    
    for mapped_content, page_num in PAGE_MAPPING.items():
        if len(mapped_content) > 20:
            mapped_words = set(mapped_content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = content_words.intersection(mapped_words)
            union = content_words.union(mapped_words)
            
            if len(union) > 0:
                similarity = len(intersection) / len(union)
                if similarity > best_match_score and similarity > 0.1:  # Minimum threshold
                    best_match_score = similarity
                best_match_page = page_num
    
    # Strategy 5: Keyword density scoring
    if best_match_score == 0:
        keyword_scores = {}
        for mapped_content, page_num in PAGE_MAPPING.items():
            if len(mapped_content) > 30:
                mapped_lower = mapped_content.lower()
                score = 0
                for word in content_words:
                    if len(word) > 3:  # Only meaningful words
                        score += mapped_lower.count(word)
                
                if score > 0:
                    keyword_scores[page_num] = keyword_scores.get(page_num, 0) + score
        
        if keyword_scores:
            best_match_page = max(keyword_scores.items(), key=lambda x: x[1])[0]
    
    logger.debug(f"Page lookup for '{content_text[:50]}...' -> Page {best_match_page}")
    return best_match_page

def get_page_content(page_number: int, limit_lines: int = None, limit_words: int = None, end_at_text: str = None) -> dict:
    """
    Get content from a specific page with optional limits.
    
    Args:
        page_number (int): Page number to extract content from
        limit_lines (int): Maximum number of lines to return
        limit_words (int): Maximum number of words to return
        end_at_text (str): Stop at this text (case-insensitive)
        
    Returns:
        dict: Page content with metadata
    """
    try:
        lines = SUBJECT_CONTENT.splitlines()
        current_page = 1
        page_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for page markers
            page_patterns = [
                r"Pg(\d+)END",
                r"page\s*(\d+)",
                r"p\.?\s*(\d+)",
                r"pg\.?\s*(\d+)",
                r"^(\d+)$",
                r"Page\s*(\d+)",
            ]
            
            for pattern in page_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match and len(line) < 30:
                    try:
                        new_page = int(match.group(1))
                        if new_page > current_page:
                            current_page = new_page
                        break
                    except (ValueError, IndexError):
                        pass
            
            # Collect content for target page
            if current_page == page_number:
                page_content.append(line)
                
                # Check for end condition
                if end_at_text and end_at_text.lower() in line.lower():
                    break
                    
                # Check line limit
                if limit_lines and len(page_content) >= limit_lines:
                    break
        
        # Apply word limit if specified
        if limit_words:
            content_text = " ".join(page_content)
            words = content_text.split()
            if len(words) > limit_words:
                content_text = " ".join(words[:limit_words])
                page_content = [content_text]
        
        return {
            "page_number": page_number,
            "content": page_content,
            "total_lines": len(page_content),
            "total_words": sum(len(line.split()) for line in page_content),
            "found": len(page_content) > 0
        }
        
    except Exception as e:
        logger.error(f"Error getting page content: {e}")
        return {
            "page_number": page_number,
            "content": [],
            "total_lines": 0,
            "total_words": 0,
            "found": False,
            "error": str(e)
        }

def find_content_boundaries(question: str, page_number: int) -> dict:
    """
    Find the most relevant content boundaries for a question on a specific page.
    
    Args:
        question (str): The question being asked
        page_number (int): Page number to search
        
    Returns:
        dict: Content boundaries and suggestions
    """
    page_data = get_page_content(page_number)
    if not page_data["found"]:
        return {"suggestions": [], "full_page": False}
    
    question_words = set(question.lower().split())
    suggestions = []
    
    for i, line in enumerate(page_data["content"]):
        line_words = set(line.lower().split())
        common_words = question_words.intersection(line_words)
        
        if len(common_words) >= 2:  # At least 2 common words
            suggestions.append({
                "line_number": i + 1,
                "text": line[:100] + "..." if len(line) > 100 else line,
                "relevance_score": len(common_words),
                "stop_here": True
            })
    
    # Sort by relevance
    suggestions.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "suggestions": suggestions[:5],  # Top 5 suggestions
        "full_page": len(suggestions) == 0,
        "total_lines": page_data["total_lines"]
    }

# --------- Enhanced Question Parsing System ---------
QUESTION_PREFIX = re.compile(r"^\s*(?:Q\.?\s*\d+|[0-9]+\.[0-9]+|QUESTIONS?:|-\s*)\s*", re.IGNORECASE)

def parse_questions_by_unit(text: str) -> Dict[str, List[str]]:
    """
    Enhanced question parser with comprehensive pattern recognition.
    
    Args:
        text (str): Syllabus content to parse for questions
        
    Returns:
        Dict[str, List[str]]: Organized questions by unit and category
    """
    units = {}
    current_unit = None
    current_section = None
    question_count = 0
    
    logger.info("Starting question parsing...")
    
    lines = text.splitlines()
    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue

        # Enhanced unit detection patterns
        unit_patterns = [
            r"^UNIT\s*(\d+)",                    # UNIT 1, UNIT 2
            r"^(\d+)\.?\s*DIFFERENT TYPES OF",   # 2. DIFFERENT TYPES OF...
            r"^(\d+)\.?\s*INTRODUCTION TO",      # 1. INTRODUCTION TO...
            r"^(\d+)\.?\s*OPERATING SYSTEM",     # 1. OPERATING SYSTEM...
        ]
        
        for pattern in unit_patterns:
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match:
                unit_num = match.group(1)
                current_unit = f"UT{unit_num}"
                current_section = line
            units.setdefault(current_unit, [])
            logger.debug(f"Found unit {current_unit}: {line}")
            break

        # Enhanced question detection patterns
        question_patterns = [
            r"^\s*(?:Q\.?\s*\d+|[0-9]+\.[0-9]+)\s*",  # Q. 1, 1.1, etc.
            r"^\s*QUESTIONS?:",                         # QUESTIONS:
            r"^\s*-\s*",                               # - (bullet point)
            r"^\s*\d+\.\s*",                           # 1. (numbered list)
        ]
        
        is_question = False
        for pattern in question_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_question = True
                break
        
        # Additional heuristics for question detection
        question_indicators = [
            "what is", "define", "explain", "describe", "list",
            "differentiate", "compare", "advantages", "disadvantages",
            "types of", "components of", "functions of", "features of",
            "how does", "why", "when", "where"
        ]
        
        line_lower = line.lower()
        if not is_question and current_unit:
            # Check if line contains question indicators
            for indicator in question_indicators:
                if indicator in line_lower and len(line) > 10:
                    is_question = True
                    break

        # Process questions
        if current_unit and is_question:
            # Clean the question text
            clean_question = QUESTION_PREFIX.sub("", line).strip()
            if clean_question and len(clean_question) > 5:
                # Remove trailing punctuation that might be formatting
                clean_question = re.sub(r'[.,;]+$', '', clean_question)
                
                # Add to unit if not duplicate
                if clean_question not in units[current_unit]:
                    units[current_unit].append(clean_question)
                    question_count += 1

    # Build comprehensive question collections
    all_questions = []
    for unit_key in sorted([k for k in units.keys() if k is not None]):
        all_questions.extend(units[unit_key])
    
    units["ALL"] = all_questions
    
    # Create combined units
    if "UT1" in units and "UT2" in units:
        units["UT1_UT2"] = units["UT1"] + units["UT2"]
    else:
        units["UT1_UT2"] = all_questions
    
    # Create topic-based collections
    topic_collections = {
        "PROCESS_MANAGEMENT": [],
        "MEMORY_MANAGEMENT": [],
        "FILE_SYSTEMS": [],
        "SECURITY": [],
        "TYPES_OF_OS": []
    }
    
    # Categorize questions by topics
    for question in all_questions:
        q_lower = question.lower()
        if any(word in q_lower for word in ["process", "thread", "scheduling", "cpu"]):
            topic_collections["PROCESS_MANAGEMENT"].append(question)
        elif any(word in q_lower for word in ["memory", "virtual", "paging", "segmentation"]):
            topic_collections["MEMORY_MANAGEMENT"].append(question)
        elif any(word in q_lower for word in ["file", "directory", "storage", "disk"]):
            topic_collections["FILE_SYSTEMS"].append(question)
        elif any(word in q_lower for word in ["security", "protection", "access", "permission"]):
            topic_collections["SECURITY"].append(question)
        elif any(word in q_lower for word in ["batch", "multiprogramming", "time-sharing", "real-time", "distributed"]):
            topic_collections["TYPES_OF_OS"].append(question)
    
    # Add topic collections to units
    units.update(topic_collections)
    
    logger.info(f"Question parsing complete: {question_count} questions across {len(units)} categories")
    return units

QUESTIONS_BY_UNIT = parse_questions_by_unit(SUBJECT_CONTENT)

# Simple in-memory storage for saved Q&A (replace with database in production)
SAVED_QA = {}

# --------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test")
def test_page():
    return render_template("test.html")

@app.route("/questions", methods=["GET"])
def questions_api():
    return jsonify({
        "UT1": QUESTIONS_BY_UNIT.get("UT1", []),
        "UT2": QUESTIONS_BY_UNIT.get("UT2", []),
        "UT1_UT2": QUESTIONS_BY_UNIT.get("UT1_UT2", []),
        "ALL": QUESTIONS_BY_UNIT.get("ALL", []),
    })

@app.route("/ask", methods=["POST"])
def ask():
    """
    Enhanced AI-powered Q&A endpoint with comprehensive instruction system.
    
    Features:
    - Multi-question support (semicolon-separated)
    - Intelligent differentiation table generation
    - Advanced page number tracking
    - Comprehensive error handling
    - Detailed AI instructions for better responses
    """
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message") or data.get("query") or ""
        user_message = user_message.strip()
        
        if not user_message:
            return jsonify({"ok": False, "error": "No question received. Please provide a valid question."})

        logger.info(f"Processing question: {user_message[:100]}...")

        # Split multiple questions by semicolon or newline
        questions = []
        for delimiter in [';', '\n', '\r\n']:
            if delimiter in user_message:
                questions = [q.strip() for q in user_message.split(delimiter) if q.strip()]
                break
        
        if not questions:
            questions = [user_message]

        answers = []

        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                # Detect question type for specialized handling
                question_lower = question.lower().strip()
                is_differentiation = any(word in question_lower for word in [
                    "differentiate", "difference", "compare", "contrast", "distinguish", "diff", "differentiatre"
                ]) or re.search(r'\bdiff(erence)?\b', question_lower)
                
                is_definition = any(word in question_lower for word in [
                    "what is", "define", "definition", "explain", "describe"
                ])
                
                is_listing = any(word in question_lower for word in [
                    "list", "enumerate", "types of", "components of", "features of", "state any"
                ])

                # Generate specialized prompt based on question type
                if is_differentiation:
                    logger.info(f"Detected differentiation question: {question[:30]}...")
                    prompt = generate_differentiation_prompt(question)
                elif is_definition:
                    logger.info(f"Detected definition question: {question[:30]}...")
                    prompt = generate_definition_prompt(question)
                elif is_listing:
                    logger.info(f"Detected listing question: {question[:30]}...")
                    prompt = generate_listing_prompt(question)
                else:
                    logger.info(f"Using general prompt for: {question[:30]}...")
                    prompt = generate_general_prompt(question)

                # Log the prompt length for debugging
                logger.info(f"Prompt length: {len(prompt)} characters")
                
                # Generate AI response with enhanced error handling
                response = generate_ai_response(prompt, question)
                
                # If AI says it can't find the material, try a more flexible approach
                if "I cannot find that in the material" in response:
                    logger.info(f"AI couldn't find content, trying fallback approach for: {question[:30]}...")
                    fallback_response = generate_fallback_response(question)
                    if fallback_response and fallback_response != "I cannot find that in the material":
                        response = fallback_response
                        logger.info("Fallback response generated successfully")
                
                # Find page number and content boundaries for the answer
                page_number = find_page_number(question + " " + response)
                
                # Get page content and boundaries
                page_data = get_page_content(page_number)
                content_boundaries = find_content_boundaries(question, page_number)
                
                # Process response based on type
                if is_differentiation and response:
                    processed_answer = process_differentiation_response(response, page_number)
                else:
                    # Clean the response to remove markdown and format like ChatGPT
                    cleaned_response = clean_markdown_formatting(response)
                    processed_answer = {
                        "type": "normal", 
                        "answer": cleaned_response,
                        "page_number": page_number
                    }
                
                # Add detailed page information
                processed_answer.update({
                    "page_info": {
                        "page_number": page_number,
                        "total_lines": page_data["total_lines"],
                        "total_words": page_data["total_words"],
                        "content_found": page_data["found"]
                    },
                    "content_boundaries": {
                        "suggestions": content_boundaries["suggestions"],
                        "full_page_available": content_boundaries["full_page"],
                        "total_page_lines": content_boundaries["total_lines"]
                    },
                    "source_reference": f"Source: Page {page_number} of Operating Systems Course Material"
                })
                
                answers.append(processed_answer)
                    
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append({
                    "type": "normal", 
                    "answer": f"I encountered an error while processing your question. Please try rephrasing it or ask a different question.",
                    "page_number": 1
                })

        logger.info(f"Successfully processed {len(answers)} answers")
        return jsonify({"ok": True, "answers": answers})

    except Exception as e:
        logger.error(f"General error in /ask: {str(e)}")
        return jsonify({"ok": False, "error": f"Server error: {str(e)}"})

def generate_differentiation_prompt(question: str) -> str:
    """Generate specialized prompt for differentiation questions."""
    return f"""You are a tutor. Use ONLY the following subject text to answer questions.
If the answer is not in the text, reply exactly: "I cannot find that in the material."

--- SUBJECT MATERIAL START ---
{SUBJECT_CONTENT}
--- SUBJECT MATERIAL END ---

Question: {question}

INSTRUCTIONS (MUST FOLLOW):
1) Start with a line: DIFFERENTIATION
2) Next line: Topic: Difference between <Term A> and <Term B>
3) Then present a plain-text side-by-side table where columns are separated by the pipe character '|' (but do NOT use Markdown fences).
   First row must be headers single-line: Term A | Term B
   Each subsequent row is exactly one point of difference, new line each.
4) Do NOT output any other text before or after the table.
Example:
DIFFERENTIATION
Topic: Difference between X and Y

Term A | Term B
Point 1 for A | Point 1 for B
Point 2 for A | Point 2 for B"""

def generate_definition_prompt(question: str) -> str:
    """Generate specialized prompt for definition questions."""
    # Check if user specifically asked for bullet points
    bullet_points_requested = any(phrase in question.lower() for phrase in [
        "bullet points", "in points", "list", "enumerate", "points"
    ])
    
    if bullet_points_requested:
        return f"""You are a helpful tutor. Use ONLY the following subject text to answer questions.
If the answer is not in the text, reply exactly: "I cannot find that in the material."

--- SUBJECT MATERIAL START ---
{SUBJECT_CONTENT}
--- SUBJECT MATERIAL END ---

Question: {question}

INSTRUCTIONS:
1. Look for the most relevant information in the material
2. For section numbers like "1.1 intro to os", look for "1.1 INTRODUCTION TO OPERATING SYSTEM" or similar headings
3. Provide a structured definition with proper headings and bullet points
4. Use ONLY dash (-) for bullet points, NEVER use asterisk (*)
5. Write in plain text - NO markdown formatting (*, **, #, etc.)
6. Use simple, clear language
7. Be flexible with matching - if you find related content, use it even if it's not an exact match
8. End with: Reference: Operating Systems Course Material

FORMAT EXAMPLE:
Definition of [Term]

[Clear definition sentence 1]
[Clear definition sentence 2]
[Clear definition sentence 3]

Key Characteristics
- Characteristic 1: Brief explanation
- Characteristic 2: Brief explanation
- Characteristic 3: Brief explanation

Additional Details
- Detail 1: Brief explanation
- Detail 2: Brief explanation

Reference: Operating Systems Course Material"""
    else:
        return f"""You are a helpful tutor. Use ONLY the following subject text to answer questions.
If the answer is not in the text, reply exactly: "I cannot find that in the material."

--- SUBJECT MATERIAL START ---
{SUBJECT_CONTENT}
--- SUBJECT MATERIAL END ---

Question: {question}

INSTRUCTIONS:
1. Look for the most relevant information in the material
2. For section numbers like "1.1 intro to os", look for "1.1 INTRODUCTION TO OPERATING SYSTEM" or similar headings
3. Provide a clear definition and explanation like ChatGPT
4. Write in plain text - NO markdown formatting (*, **, #, etc.)
5. Use simple, clear language with proper paragraphs
6. Do NOT use bullet points unless specifically requested
7. Be flexible with matching - if you find related content, use it even if it's not an exact match
8. End with: Reference: Operating Systems Course Material

FORMAT EXAMPLE:
Based on the material, [term] is [clear definition]. [Explanation paragraph 1]. [Explanation paragraph 2].

[Additional relevant information in paragraph form].

Reference: Operating Systems Course Material"""

def generate_listing_prompt(question: str) -> str:
    """Generate specialized prompt for listing questions."""
    # Check if user specifically asked for bullet points
    bullet_points_requested = any(phrase in question.lower() for phrase in [
        "bullet points", "in points", "list", "enumerate", "points"
    ])
    
    if bullet_points_requested:
        return f"""You are a helpful tutor. Use ONLY the following subject text to answer questions.
If the answer is not in the text, reply exactly: "I cannot find that in the material."

--- SUBJECT MATERIAL START ---
{SUBJECT_CONTENT}
--- SUBJECT MATERIAL END ---

Question: {question}

INSTRUCTIONS:
1. Look for the most relevant information in the material
2. For section numbers like "1.1 intro to os", look for "1.1 INTRODUCTION TO OPERATING SYSTEM" or similar headings
3. Provide a structured list with proper headings and bullet points
4. Use ONLY dash (-) for bullet points, NEVER use asterisk (*)
5. Write in plain text - NO markdown formatting (*, **, #, etc.)
6. Use simple, clear language
7. Be flexible with matching - if you find related content, use it even if it's not an exact match
8. End with: Reference: Operating Systems Course Material

FORMAT EXAMPLE:
[Main Topic]

[Brief introduction sentence 1]
[Brief introduction sentence 2]

List of [items]
- Item 1: Brief description
- Item 2: Brief description
- Item 3: Brief description

Additional Details
- Detail 1: Brief explanation
- Detail 2: Brief explanation

Reference: Operating Systems Course Material"""
    else:
        return f"""You are a helpful tutor. Use ONLY the following subject text to answer questions.
If the answer is not in the text, reply exactly: "I cannot find that in the material."

--- SUBJECT MATERIAL START ---
{SUBJECT_CONTENT}
--- SUBJECT MATERIAL END ---

Question: {question}

INSTRUCTIONS:
1. Look for the most relevant information in the material
2. For section numbers like "1.1 intro to os", look for "1.1 INTRODUCTION TO OPERATING SYSTEM" or similar headings
3. Provide a clear list like ChatGPT
4. Write in plain text - NO markdown formatting (*, **, #, etc.)
5. Use simple, clear language with proper paragraphs
6. Do NOT use bullet points unless specifically requested
7. Be flexible with matching - if you find related content, use it even if it's not an exact match
8. End with: Reference: Operating Systems Course Material

FORMAT EXAMPLE:
Based on the material, here are the [items]: [Item 1], [Item 2], [Item 3]. [Explanation paragraph 1]. [Explanation paragraph 2].

[Additional relevant information in paragraph form].

Reference: Operating Systems Course Material"""

def generate_general_prompt(question: str) -> str:
    """Generate general prompt for other question types."""
    # Check if user specifically asked for bullet points
    bullet_points_requested = any(phrase in question.lower() for phrase in [
        "bullet points", "in points", "list", "enumerate", "points"
    ])
    
    if bullet_points_requested:
        return f"""You are a helpful tutor. Use ONLY the following subject text to answer questions.
If the answer is not in the text, reply exactly: "I cannot find that in the material."

--- SUBJECT MATERIAL START ---
{SUBJECT_CONTENT}
--- SUBJECT MATERIAL END ---

Question: {question}

INSTRUCTIONS:
1. Look for the most relevant information in the material
2. For section numbers like "1.1 intro to os", look for "1.1 INTRODUCTION TO OPERATING SYSTEM" or similar headings
3. Provide a structured response with proper headings and bullet points
4. Use ONLY dash (-) for bullet points, NEVER use asterisk (*)
5. Write in plain text - NO markdown formatting (*, **, #, etc.)
6. Use simple, clear language
7. Keep each bullet point concise (one idea per point)
8. Use bold headings to divide sections (Key Points, Additional Details)
9. Make headings stand out by putting them on separate lines
10. Be flexible with matching - if you find related content, use it even if it's not an exact match
11. End with: Reference: Operating Systems Course Material

FORMAT EXAMPLE:
[Main Topic Name]

[Brief introduction sentence 1]
[Brief introduction sentence 2]
[Brief introduction sentence 3]
[Brief introduction sentence 4]

Key Points
- Point 1: Brief explanation
- Point 2: Brief explanation  
- Point 3: Brief explanation

Additional Details
- Detail 1: Brief explanation
- Detail 2: Brief explanation

Reference: Operating Systems Course Material"""
    else:
        return f"""You are a helpful tutor. Use ONLY the following subject text to answer questions.
If the answer is not in the text, reply exactly: "I cannot find that in the material."

--- SUBJECT MATERIAL START ---
{SUBJECT_CONTENT}
--- SUBJECT MATERIAL END ---

Question: {question}

INSTRUCTIONS:
1. Look for the most relevant information in the material
2. For section numbers like "1.1 intro to os", look for "1.1 INTRODUCTION TO OPERATING SYSTEM" or similar headings
3. Provide a clear, conversational answer like ChatGPT
4. Write in plain text - NO markdown formatting (*, **, #, etc.)
5. Use simple, clear language with proper paragraphs
6. Do NOT use bullet points unless specifically requested
7. Be flexible with matching - if you find related content, use it even if it's not an exact match
8. End with: Reference: Operating Systems Course Material

FORMAT EXAMPLE:
Based on the material, [topic] is [definition]. [Explanation paragraph 1]. [Explanation paragraph 2].

[Additional relevant information in paragraph form].

Reference: Operating Systems Course Material"""

def generate_fallback_response(question: str) -> str:
    """Generate a fallback response using heuristic content extraction."""
    try:
        # Extract relevant sentences from the syllabus content
        question_words = set(question.lower().split())
        sentences = re.split(r'(?<=[.?!])\s+', SUBJECT_CONTENT)
        
        relevant_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence_words = set(sentence.lower().split())
            # Check for word overlap
            overlap = len(question_words.intersection(sentence_words))
            if overlap >= 2:  # At least 2 common words
                relevant_sentences.append((overlap, sentence.strip()))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if relevant_sentences:
            # Take top 8 sentences and format as bullet points
            top_sentences = [s for _, s in relevant_sentences[:8]]
            response = "\n".join(f"- {s}" for s in top_sentences)
            response += "\n\nReference: Operating Systems Course Material"
            return response
        else:
            return "I cannot find that in the material."
            
    except Exception as e:
        logger.error(f"Fallback response generation failed: {e}")
        return "I cannot find that in the material."

def generate_ai_response(prompt: str, question: str) -> str:
    """Generate AI response with enhanced error handling and fallback."""
    try:
        # Use API rotator if available, otherwise fallback to direct Gemini
        if API_ROTATOR_AVAILABLE and api_rotator:
            logger.info("Using API rotator for AI response")
            
            # Try Gemini first
            gemini_key = api_rotator.get_next_gemini_key()
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                    temp_client = genai.GenerativeModel("gemini-2.0-flash-exp")
                    response = temp_client.generate_content(prompt)
                    
                    if hasattr(response, 'text') and response.text:
                        return response.text.strip()
                except Exception as e:
                    logger.warning(f"Gemini API error with key rotation: {e}")
            
            # Try OpenAI as fallback
            openai_key = api_rotator.get_next_openai_key()
            if openai_key:
                try:
                    import openai
                    openai_client = openai.OpenAI(api_key=openai_key)
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4000,
                        temperature=0.7
                    )
                    
                    if response.choices and response.choices[0].message.content:
                        return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning(f"OpenAI API error: {e}")
            
            # If all keys fail, fallback to original method
            logger.warning("All API keys failed, using fallback")
            
        # Fallback to direct Gemini API
        if not client:
            return "AI service is currently unavailable. Please try again later."
        
        response = client.generate_content(prompt)
        
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            logger.warning(f"No text in AI response for question: {question[:50]}...")
            return "I apologize, but I couldn't generate a proper response for this question. Please try rephrasing your question."

    except Exception as e:
        logger.error(f"AI generation error: {str(e)}")
        return f"I encountered an error while processing your question. Please try again or contact support if the issue persists."

def process_differentiation_response(response: str, page_number: int) -> dict:
    """Process differentiation response and extract table data."""
    try:
        left, right = [], []
        lines = [ln.strip() for ln in response.splitlines() if "|" in ln]
        
        for ln in lines:
            if ln.lower().startswith(("term a", "differentiation", "topic:")):
                continue
            cells = [c.strip() for c in ln.split("|", 1)]  # Split only on first |
            if len(cells) >= 2:
                left.append(cells[0])
                right.append(cells[1])
        
        if left and right:
            return {
                "type": "differentiate", 
                "left": left, 
                "right": right,
                "page_number": page_number
            }
        else:
            return {
                "type": "normal", 
                "answer": response,
                "page_number": page_number
            }
    except Exception as e:
        logger.error(f"Error processing differentiation response: {str(e)}")
        return {
            "type": "normal", 
            "answer": response,
            "page_number": page_number
        }

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Enhanced AI-powered test evaluation system with comprehensive grading.
    
    Features:
    - Intelligent answer evaluation against course material
    - Detailed feedback and scoring
    - Key points analysis
    - Page number references
    - JSON-formatted responses
    """
    try:
        data = request.get_json(force=True)
        question = (data.get("question") or "").strip()
        student_answer = (data.get("student_answer") or "").strip()

        if not question or not student_answer:
            return jsonify({"ok": False, "error": "Both question and student answer are required for evaluation."}), 400

        logger.info(f"Evaluating answer for question: {question[:50]}...")

        # Enhanced grading prompt with detailed instructions
        grading_prompt = f"""
You are an expert Operating Systems course examiner. Your task is to evaluate the student's answer comprehensively and fairly.

EVALUATION CRITERIA:
1. Factual Accuracy: How correct is the information provided?
2. Completeness: Does the answer cover the main points?
3. Understanding: Does the student demonstrate conceptual understanding?
4. Clarity: Is the answer well-structured and clear?

REFERENCE MATERIAL (Use as the source of truth):
{SUBJECT_CONTENT}

QUESTION: {question}

STUDENT ANSWER: {student_answer}

INSTRUCTIONS:
- Grade based ONLY on the reference material provided
- Be fair and constructive in feedback
- Identify specific strengths and weaknesses
- Provide actionable suggestions for improvement
- If the answer is partially correct, acknowledge what's right

Provide your evaluation in this EXACT JSON format:
{{
  "score": [0-100 integer],
  "feedback": "Detailed constructive feedback (2-3 sentences)",
  "ideal_answer": ["Key point 1", "Key point 2", "Key point 3", "Key point 4"],
  "key_points_missed": ["Missing point 1", "Missing point 2"],
  "strengths": ["What the student got right"],
  "areas_for_improvement": ["Specific suggestions for improvement"],
  "found_in_material": true,
  "page_reference": [page number where topic is found]
}}

SCORING GUIDELINES:
- 90-100: Excellent - Comprehensive, accurate, well-explained
- 80-89: Good - Mostly correct with minor gaps
- 70-79: Satisfactory - Basic understanding with some errors
- 60-69: Needs Improvement - Partial understanding, several errors
- 0-59: Unsatisfactory - Major misconceptions or incomplete
"""

        try:
            if not client:
                return jsonify({
                    "ok": False,
                    "error": "AI evaluation service is currently unavailable."
                }), 503

            # Generate AI evaluation
            response = client.generate_content(grading_prompt)
            
            if not hasattr(response, 'text') or not response.text:
                raise ValueError("No response text from AI")

            raw_response = response.text.strip()
            logger.debug(f"Raw AI response: {raw_response[:200]}...")
            
            # Extract and parse JSON from response
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON found in AI response")
            
            json_text = raw_response[json_start:json_end]
            result = json.loads(json_text)
            
            # Validate and clean the result
            score = max(0, min(100, int(result.get("score", 0))))
            feedback = clean_markdown_formatting(str(result.get("feedback", "")))
            ideal_answer = [clean_markdown_formatting(item) for item in result.get("ideal_answer", [])]
            key_missed = [clean_markdown_formatting(item) for item in result.get("key_points_missed", [])]
            strengths = [clean_markdown_formatting(item) for item in result.get("strengths", [])]
            improvements = [clean_markdown_formatting(item) for item in result.get("areas_for_improvement", [])]
            found_in_material = bool(result.get("found_in_material", True))
            
            # Extract page number
            page_number = result.get("page_reference", None)
            if page_number:
                try:
                    page_number = int(page_number)
                except (ValueError, TypeError):
                    page_number = find_page_number(question + " " + " ".join(ideal_answer))
            else:
                page_number = find_page_number(question + " " + " ".join(ideal_answer))
            
            # Enhance feedback with page reference
            if "Page Number:" not in feedback and page_number:
                feedback += f" Reference: Page {page_number}"

            logger.info(f"Evaluation complete: Score {score}/100")

            return jsonify({
                "ok": True,
                "score": score,
                "feedback": feedback,
                "ideal_answer": ideal_answer,
                "key_points_missed": key_missed,
                "strengths": strengths,
                "areas_for_improvement": improvements,
                "found_in_material": found_in_material,
                "page_number": page_number,
            })

        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error: {je}")
            return jsonify({
                "ok": True,
                "score": 50,
                "feedback": "Evaluation completed with limited feedback due to processing constraints.",
                "ideal_answer": ["Please refer to course material for comprehensive answer"],
                "key_points_missed": ["Unable to determine specific gaps"],
                "strengths": ["Answer was processed successfully"],
                "areas_for_improvement": ["Ensure answer covers all aspects of the question"],
                "found_in_material": True,
                "page_number": find_page_number(question),
            })

    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({
            "ok": False,
            "error": f"Evaluation failed: {str(e)}",
        }), 500

# Additional routes for the frontend functionality
@app.route("/list_saved", methods=["GET"])
def list_saved():
    """List saved Q&A entries with page numbers"""
    folder = request.args.get('folder', '')
    folders = list(set(qa.get('folder', 'default') for qa in SAVED_QA.values()))
    
    if folder:
        entries = [{"question": qa.get("question", ""), "answer": qa["answer"], "path": key, "page_number": qa.get("page_number", 1)} 
                  for key, qa in SAVED_QA.items() if qa.get('folder') == folder]
    else:
        entries = [{"question": qa.get("question", ""), "answer": qa["answer"], "folder": qa.get('folder', 'default'), 
                   "path": key, "page_number": qa.get("page_number", 1)} 
                  for key, qa in SAVED_QA.items()]
    
    return jsonify({"ok": True, "folders": folders, "entries": entries})

@app.route("/save_answer", methods=["POST"])
def save_answer():
    """Save a Q&A pair with enhanced page information and content boundaries"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        folder = data.get('folder', 'default').strip()
        page_number = data.get('page_number', None)
        
        if not question or not answer:
            return jsonify({"ok": False, "error": "Question and answer required"})
        
        # Find page number if not provided
        if not page_number:
            page_number = find_page_number(question + " " + answer)
        
        # Get detailed page information
        page_data = get_page_content(page_number)
        content_boundaries = find_content_boundaries(question, page_number)
        
        # Create unique key with timestamp
        import time
        timestamp = int(time.time())
        key = f"{folder}_{timestamp}_{len(SAVED_QA)}"
        
        SAVED_QA[key] = {
            "question": question,
            "answer": answer,
            "folder": folder,
            "page_number": page_number,
            "timestamp": timestamp,
            "page_info": {
                "page_number": page_number,
                "total_lines": page_data["total_lines"],
                "total_words": page_data["total_words"],
                "content_found": page_data["found"]
            },
            "content_boundaries": {
                "suggestions": content_boundaries["suggestions"],
                "full_page_available": content_boundaries["full_page"],
                "total_page_lines": content_boundaries["total_lines"]
            },
            "source_reference": f"Source: Page {page_number} of Operating Systems Course Material"
        }
        
        logger.info(f"Saved Q&A: {question[:50]}... -> Page {page_number}")
        
        return jsonify({
            "ok": True, 
            "key": key,
            "page_number": page_number,
            "page_info": SAVED_QA[key]["page_info"],
            "content_boundaries": SAVED_QA[key]["content_boundaries"]
        })
    except Exception as e:
        logger.error(f"Error saving Q&A: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/recommend", methods=["POST"])
def recommend():
    """Recommend saved Q&A based on text similarity with enhanced page information"""
    try:
        data = request.get_json()
        search_text = data.get('text', '').lower()
        
        suggestions = []
        for key, qa in SAVED_QA.items():
            question = qa.get('question', '').lower()
            if search_text in question or any(word in question for word in search_text.split()):
                suggestions.append({
                    "key": key,
                    "question": qa.get('question', ''),
                    "answer": qa.get('answer', ''),
                    "folder": qa.get('folder', 'default'),
                    "page_number": qa.get('page_number', 1),
                    "page_info": qa.get('page_info', {}),
                    "content_boundaries": qa.get('content_boundaries', {}),
                    "source_reference": qa.get('source_reference', ''),
                    "timestamp": qa.get('timestamp', 0)
                })
        
        # Sort by relevance (exact match first, then by timestamp)
        suggestions.sort(key=lambda x: (search_text in x['question'].lower(), x.get('timestamp', 0)), reverse=True)
        
        return jsonify({"ok": True, "suggestions": suggestions[:5]})  # Limit to 5 suggestions
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/load_saved_qa", methods=["POST"])
def load_saved_qa():
    """
    Load a specific saved Q&A with full context and page information.
    
    Expected JSON:
    {
        "key": "folder_timestamp_index"
    }
    """
    try:
        data = request.get_json()
        key = data.get('key', '').strip()
        
        if not key:
            return jsonify({"ok": False, "error": "Q&A key is required"}), 400
        
        if key not in SAVED_QA:
            return jsonify({"ok": False, "error": "Saved Q&A not found"}), 404
        
        qa_data = SAVED_QA[key]
        
        # Get current page content to ensure it's up-to-date
        page_number = qa_data.get('page_number', 1)
        page_data = get_page_content(page_number)
        
        # Update page info if needed
        qa_data['page_info'] = {
            "page_number": page_number,
            "total_lines": page_data["total_lines"],
            "total_words": page_data["total_words"],
            "content_found": page_data["found"]
        }
        
        return jsonify({
            "ok": True,
            "qa_data": qa_data,
            "full_context": {
                "question": qa_data.get('question', ''),
                "answer": qa_data.get('answer', ''),
                "page_info": qa_data['page_info'],
                "content_boundaries": qa_data.get('content_boundaries', {}),
                "source_reference": qa_data.get('source_reference', ''),
                "folder": qa_data.get('folder', 'default'),
                "timestamp": qa_data.get('timestamp', 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Error loading saved Q&A: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/page_info", methods=["GET"])
def page_info():
    """Get page mapping information and saved Q&A stats for debugging"""
    return jsonify({
        "ok": True,
        "total_pages": max(PAGE_MAPPING.values()) if PAGE_MAPPING else 1,
        "content_sections": list(CONTENT_SECTIONS.keys()),
        "sample_mappings": dict(list(PAGE_MAPPING.items())[:10]),  # First 10 mappings
        "saved_qa_count": len(SAVED_QA),
        "saved_qa_keys": list(SAVED_QA.keys())[:5]  # Show first 5 keys
    })

@app.route("/get_page_content", methods=["POST"])
def get_page_content_api():
    """
    Get content from a specific page with optional limits.
    
    Expected JSON:
    {
        "page_number": 1,
        "limit_lines": 10,  // optional
        "limit_words": 100, // optional
        "end_at_text": "stop here" // optional
    }
    """
    try:
        data = request.get_json() or {}
        page_number = data.get("page_number")
        
        if not page_number:
            return jsonify({"ok": False, "error": "Page number is required"}), 400
        
        try:
            page_number = int(page_number)
        except (ValueError, TypeError):
            return jsonify({"ok": False, "error": "Page number must be an integer"}), 400
        
        limit_lines = data.get("limit_lines")
        limit_words = data.get("limit_words")
        end_at_text = data.get("end_at_text")
        
        # Get page content
        page_data = get_page_content(page_number, limit_lines, limit_words, end_at_text)
        
        if not page_data["found"]:
            return jsonify({
                "ok": True,
                "page_number": page_number,
                "content": [],
                "message": f"No content found for page {page_number}",
                "total_lines": 0,
                "total_words": 0
            })
        
        return jsonify({
            "ok": True,
            "page_number": page_number,
            "content": page_data["content"],
            "total_lines": page_data["total_lines"],
            "total_words": page_data["total_words"],
            "limits_applied": {
                "limit_lines": limit_lines,
                "limit_words": limit_words,
                "end_at_text": end_at_text
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_page_content_api: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/find_content_boundaries", methods=["POST"])
def find_content_boundaries_api():
    """
    Find content boundaries for a question on a specific page.
    
    Expected JSON:
    {
        "question": "What is an operating system?",
        "page_number": 1
    }
    """
    try:
        data = request.get_json() or {}
        question = data.get("question", "").strip()
        page_number = data.get("page_number")
        
        if not question:
            return jsonify({"ok": False, "error": "Question is required"}), 400
        
        if not page_number:
            return jsonify({"ok": False, "error": "Page number is required"}), 400
        
        try:
            page_number = int(page_number)
        except (ValueError, TypeError):
            return jsonify({"ok": False, "error": "Page number must be an integer"}), 400
        
        # Find content boundaries
        boundaries = find_content_boundaries(question, page_number)
        
        return jsonify({
            "ok": True,
            "question": question,
            "page_number": page_number,
            "suggestions": boundaries["suggestions"],
            "full_page": boundaries["full_page"],
            "total_lines": boundaries["total_lines"]
        })
        
    except Exception as e:
        logger.error(f"Error in find_content_boundaries_api: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/get_full_page", methods=["POST"])
def get_full_page_api():
    """
    Get complete page content for a given page number.
    
    Expected JSON:
    {
        "page_number": 1
    }
    """
    try:
        data = request.get_json() or {}
        page_number = data.get("page_number")
        
        if not page_number:
            return jsonify({"ok": False, "error": "Page number is required"}), 400
        
        try:
            page_number = int(page_number)
        except (ValueError, TypeError):
            return jsonify({"ok": False, "error": "Page number must be an integer"}), 400
        
        # Get full page content
        page_data = get_page_content(page_number)
        
        return jsonify({
            "ok": True,
            "page_number": page_number,
            "content": page_data["content"],
            "total_lines": page_data["total_lines"],
            "total_words": page_data["total_words"],
            "found": page_data["found"]
        })
        
    except Exception as e:
        logger.error(f"Error in get_full_page_api: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/test_save", methods=["GET"])
def test_save():
    """Test endpoint to add sample saved Q&A"""
    try:
        # Add some test data
        test_data = [
            {
                "question": "What is an operating system?",
                "answer": "An operating system is system software that manages computer hardware and software resources.",
                "folder": "test"
            },
            {
                "question": "What is process scheduling?",
                "answer": "Process scheduling is the method by which processes are assigned to run on the CPU.",
                "folder": "test"
            }
        ]
        
        for i, item in enumerate(test_data):
            key = f"test_{i}"
            SAVED_QA[key] = {
                "question": item["question"],
                "answer": item["answer"],
                "folder": item["folder"],
                "page_number": i + 1
            }
        
        return jsonify({
            "ok": True, 
            "message": f"Added {len(test_data)} test entries",
            "total_saved": len(SAVED_QA)
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# --------- Application Startup and Configuration ---------
def initialize_application():
    """Initialize the application with comprehensive setup."""
    logger.info("=" * 60)
    logger.info("OS STUDY BOT - Enhanced Flask Backend v2.0")
    logger.info("=" * 60)
    
    # Log system status
    logger.info(f"Flask App: Initialized")
    logger.info(f"Gemini Model: {CURRENT_MODEL}")
    logger.info(f"Syllabus Content: {len(SUBJECT_CONTENT)} characters")
    logger.info(f"Content Sections: {len(CONTENT_SECTIONS)}")
    logger.info(f"Page Mappings: {len(PAGE_MAPPING)}")
    logger.info(f"Questions Parsed: {len(QUESTIONS_BY_UNIT.get('ALL', []))}")
    
    # Log available endpoints
    logger.info("Available Endpoints:")
    logger.info("  GET  /                    - Main interface")
    logger.info("  GET  /test               - Test interface")
    logger.info("  GET  /questions          - Get questions by unit")
    logger.info("  POST /ask                - AI Q&A endpoint (enhanced)")
    logger.info("  POST /evaluate           - Test evaluation")
    logger.info("  POST /get_page_content   - Get page content with limits")
    logger.info("  POST /find_content_boundaries - Find content boundaries")
    logger.info("  POST /get_full_page      - Get complete page content")
    logger.info("  POST /save_answer        - Save Q&A with page info")
    logger.info("  POST /load_saved_qa      - Load saved Q&A with context")
    logger.info("  POST /recommend          - Recommend saved Q&A")
    logger.info("  GET  /page_info          - System information")
    logger.info("  GET  /api_status         - API key rotation status")
    
    logger.info("=" * 60)
    logger.info("Application ready for requests!")
    logger.info("=" * 60)

@app.route("/api_status", methods=["GET"])
def api_status():
    """Get API key rotation status and usage statistics."""
    try:
        if not API_ROTATOR_AVAILABLE or not api_rotator:
            return jsonify({
                "ok": True,
                "rotator_available": False,
                "message": "API rotator not available, using direct Gemini API"
            })
        
        status = api_rotator.get_status()
        return jsonify({
            "ok": True,
            "rotator_available": True,
            "status": status,
            "message": "Simple API rotator is active with key rotation"
        })
        
    except Exception as e:
        logger.error(f"Error getting API status: {e}")
        return jsonify({
            "ok": False,
            "error": f"Failed to get API status: {str(e)}"
        }), 500

# Initialize application on startup
initialize_application()

# --------- Main Application Entry Point ---------
if __name__ == "__main__":
    try:
        logger.info("Starting OS Study Bot server...")
        port = int(os.environ.get('PORT', 5001))
        debug_mode = os.environ.get('FLASK_ENV') != 'production'
        app.run(
            debug=debug_mode,
            host='0.0.0.0',
            port=port,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")
        raise