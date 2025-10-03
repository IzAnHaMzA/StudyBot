# app.py
"""
OS Study Bot - Flask backend
- Supports online (Gemini) or offline (llama_cpp) LLMs.
- Caches confirmed Q&A to disk as JSON files.
- Provides recommendation/search over saved Q&A.
- Provides a Test UI for UT1/UT2/all units and a grader endpoint.
- Uses the textbook/syllabus file (plain text) for "ground truth".
- Minimal external dependencies: flask, google.genai (optional), llama_cpp (optional).
"""
# runs in env using  .\env\Scripts\Activate.ps1
# path of llama model: "C:\Users\User\OneDrive\Desktop\models\Llama_off_bot\llama-2-7b.Q4_K_M.gguf"

import os
import re
import json
import time
import uuid
import math
import html
import shutil
import pathlib
import datetime
from typing import List, Dict, Tuple, Any

from flask import Flask, render_template, request, jsonify, send_from_directory

# --- Optional imports: google.genai and llama_cpp (we try to import but fall back if missing) ---
GEMINI_API_KEY = os.environ.get("AIzaSyDPwEGekUv4k2GVN19ev5D4GDzh7ahrvTk")  # set this in your environment to enable Gemini
try:
    if GEMINI_API_KEY:
        from google import genai
        from google.genai import types
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        genai_client = None
except Exception:
    genai_client = None

try:
    from llama_cpp import Llama  # local model (gguf) interface
    LLAMA_AVAILABLE = True
except Exception:
    Llama = None
    LLAMA_AVAILABLE = False

# You can set the path to your GGUF model via environment variable LLAMA_MODEL_PATH,
# or edit the default below.
LLAMA_MODEL_PATH = "C:\\Users\\User\\OneDrive\\Desktop\\models\\Llama_off_bot\\llama-2-7b.Q4_K_M.gguf"  # e.g. C:\...\.gguf

# If you have a local model and llama_cpp installed, we will load it lazily
_llama_instance = None

# ---- App and file constants ----
app = Flask(__name__, template_folder="templates", static_folder="static")
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVED_DIR = DATA_DIR / "saved"           # saved/<folder>/<uuid>.json
INDEX_FILE = DATA_DIR / "saved_index.json"
SYLLABUS_PATH = os.environ.get("SYLLABUS_PATH") or str(BASE_DIR / "OS SYLLABUS INCOMPLETE.txt")

# Ensure folders exist
os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---- Load subject content (syllabus / textbook) ----
def load_subject_content(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return ""

SUBJECT_CONTENT = load_subject_content(SYLLABUS_PATH)

# ---- Question parser (units) ----
QUESTION_PREFIX = re.compile(r"^\s*(?:Q\.?\s*\d+|[0-9]+\.[0-9]+)\s*", re.IGNORECASE)
UNIT_HEADER = re.compile(r"^UNIT\s*(\d+)", re.IGNORECASE)

def parse_questions_by_unit(text: str) -> Dict[str, List[str]]:
    units = {}
    current_unit = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = UNIT_HEADER.match(line)
        if m:
            current_unit = f"UT{m.group(1)}"
            units.setdefault(current_unit, [])
            continue
        if current_unit and QUESTION_PREFIX.match(line):
            q = QUESTION_PREFIX.sub("", line).strip()
            units[current_unit].append(q)
    # Build ALL
    all_qs = []
    for k in sorted(units.keys()):
        all_qs.extend(units[k])
    units["ALL"] = all_qs
    if "UT1" in units and "UT2" in units:
        units["UT1_UT2"] = units["UT1"] + units["UT2"]
    else:
        units["UT1_UT2"] = all_qs
    return units

QUESTIONS_BY_UNIT = parse_questions_by_unit(SUBJECT_CONTENT)

# ---- Utilities: normalization, tokenization, similarity ----
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    return [t for t in normalize_text(s).split() if t]

def jaccard_similarity(a: str, b: str) -> float:
    sa = set(tokens(a))
    sb = set(tokens(b))
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return len(inter) / len(union)

def overlap_ratio(a: str, b: str) -> float:
    # counts token overlap by min fraction
    ta = set(tokens(a))
    tb = set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta.intersection(tb)) / max(1, min(len(ta), len(tb)))

# ---- Saved index helpers ----
def load_index() -> Dict[str, Any]:
    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}

def save_index(index: Dict[str, Any]):
    with open(INDEX_FILE, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

# Initialize index if missing
INDEX = load_index()
if "entries" not in INDEX:
    INDEX["entries"] = []  # list of dict {id, question, answer, folder, path, timestamp, keywords}

def index_add_entry(entry: Dict[str, Any]):
    INDEX["entries"].append(entry)
    save_index(INDEX)

def index_find_exact(question: str) -> Dict[str, Any]:
    qn = normalize_text(question)
    for e in INDEX["entries"]:
        if normalize_text(e.get("question", "")) == qn:
            return e
    return {}

def index_search_similar(question: str, top_n: int = 5) -> List[Dict[str, Any]]:
    scores = []
    for e in INDEX["entries"]:
        q = e.get("question", "")
        score = jaccard_similarity(question, q)
        scores.append((score, e))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [e for s, e in scores[:top_n] if s > 0]

# ---- Save Q&A to JSON file ----
def save_answer_to_folder(question: str, answer: str, folder: str = "default") -> Dict[str, Any]:
    folder_path = SAVED_DIR / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    entry_id = str(uuid.uuid4())
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    filename = f"{entry_id}.json"
    path = folder_path / filename
    entry = {
        "id": entry_id,
        "question": question,
        "answer": answer,
        "folder": folder,
        "path": str(path),
        "timestamp": ts,
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2, ensure_ascii=False)
        index_add_entry(entry)
        return {"ok": True, "entry": entry}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def list_saved_folders() -> List[str]:
    return sorted([p.name for p in SAVED_DIR.iterdir() if p.is_dir()])

def list_saved_entries(folder: str = None) -> List[Dict[str, Any]]:
    if folder:
        dirp = SAVED_DIR / folder
        if not dirp.exists():
            return []
        files = sorted(dirp.glob("*.json"))
        out = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    out.append(json.load(fh))
            except Exception:
                continue
        return out
    else:
        return INDEX.get("entries", [])

# ---- LLM helpers: Gemini (online) and Llama (offline) ----
MODEL_NAME = "gemini-2.5-pro"  # used only for Gemini if available

def generate_with_gemini(prompt: str, timeout: int = 30) -> str:
    """Generate text with Google Gemini (if genai_client available)."""
    if not genai_client:
        return ""
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]
    # Accumulate text from streamed response
    reply_text = ""
    try:
        for chunk in genai_client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
        ):
            # the streaming chunk structure may differ; handle common shapes
            if getattr(chunk, "text", ""):
                reply_text += chunk.text
            elif hasattr(chunk, "candidates"):
                for cand in getattr(chunk, "candidates") or []:
                    if hasattr(cand, "content") and getattr(cand, "content", None):
                        parts = getattr(cand.content, "parts", []) or []
                        for p in parts:
                            if getattr(p, "text", ""):
                                reply_text += p.text
    except Exception as e:
        # If streaming failed, try one-shot generate_content
        try:
            res = genai_client.models.generate_content(model=MODEL_NAME, contents=contents)
            if getattr(res, "candidates", None):
                for cand in res.candidates:
                    if getattr(cand, "content", None):
                        for part in cand.content.parts:
                            if getattr(part, "text", ""):
                                reply_text += part.text
            elif getattr(res, "text", ""):
                reply_text += res.text
        except Exception:
            reply_text += f"\n\n[Gemini generation failed: {str(e)}]"
    return reply_text.strip()

def _load_llama_instance():
    global _llama_instance
    if _llama_instance is not None:
        return _llama_instance
    if not LLAMA_AVAILABLE:
        return None
    model_path = LLAMA_MODEL_PATH or os.environ.get("LLAMA_MODEL_PATH", "")
    if not model_path:
        return None
    try:
        _llama_instance = Llama(model_path=model_path, n_ctx=2048)
        return _llama_instance
    except Exception as ex:
        # If loading fails, keep _llama_instance None
        print("Failed loading Llama model:", ex)
        return None

def generate_with_llama(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Generate with local llama_cpp Llama if available."""
    llm = _load_llama_instance()
    if not llm:
        return ""
    try:
        out = llm(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
        # llama_cpp returns dict with `choices` containing `text` (older) or "choices"[0]["text"]:
        if isinstance(out, dict):
            # Support different shapes
            if "choices" in out and out["choices"]:
                # common: out["choices"][0]["text"]
                txt = out["choices"][0].get("text") or out["choices"][0].get("message", {}).get("content", "")
                return txt.strip() if txt else ""
            # some bindings return "text"
            if "text" in out:
                return out["text"].strip()
        # Fallback to string representation
        return str(out).strip()
    except Exception as e:
        print("Local Llama generation failed:", e)
        return ""

# ---- High level generate function (tries saved cache, then offline then online) ----
def generate_answer_for_question(question: str) -> Dict[str, Any]:
    """
    Returns dict with keys:
      - type: "differentiate" | "normal" | "saved"
      - answer: string (for normal)
      - left,right lists (for differentiate)
      - source: "saved" | "llama" | "gemini" | "heuristic"
    """
    # 1) Check exact saved index
    exact = index_find_exact(question)
    if exact:
        return {"type": "saved", "answer": exact.get("answer", ""), "source": "saved", "meta": exact}

    # 2) Determine if differentiate question
    is_diff = ("differentiate" in question.lower()) or ("difference" in question.lower()) or re.search(r"\bdiff(erence)?\b", question, re.IGNORECASE)

    # 3) Build prompt tailored for differentiate or normal
    if is_diff:
        prompt = f"""
You are a tutor. Use ONLY the following subject text to answer questions.
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
Point 2 for A | Point 2 for B
"""
    else:
        prompt = f"""
You are a tutor. Use ONLY the following subject text to answer questions.
If the answer is not in the text, reply exactly: "I cannot find that in the material."

--- SUBJECT MATERIAL START ---
{SUBJECT_CONTENT}
--- SUBJECT MATERIAL END ---

Question: {question}

INSTRUCTIONS:
Answer strictly as bullet points. Use short sentences. Provide at least 6-12 points if possible.
At the end add a single reference line: Reference: <textbook name>, Page <number>
"""

    # 4) Try local model first if available (llama)
    if LLAMA_AVAILABLE and (LLAMA_MODEL_PATH or os.environ.get("LLAMA_MODEL_PATH")):
        out = generate_with_llama(prompt)
        if out and len(out.strip()) > 0:
            reply_text = out.strip()
            source = "llama"
        else:
            reply_text = ""
            source = ""
    else:
        reply_text = ""
        source = ""

    # 5) If we don't have reply and Gemini is available, call Gemini
    if (not reply_text) and genai_client:
        reply_text = generate_with_gemini(prompt)
        source = "gemini" if reply_text else source

    # 6) If nothing, fallback to a heuristic extraction from SUBJECT_CONTENT
    if not reply_text:
        # Heuristic: gather sentences containing keywords from the question
        q_tokens = tokens(question)
        sents = re.split(r'(?<=[.?!])\s+', SUBJECT_CONTENT)
        scored = []
        for s in sents:
            if not s.strip():
                continue
            score = overlap_ratio(s, question)
            if score > 0:
                scored.append((score, s.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Build reply text as bullet points of top 6 sentences
        if scored:
            reply_text = "\n".join(f"• {s}" for _, s in scored[:8])
            reply_text += "\nReference: (from syllabus)"
            source = "heuristic"
        else:
            reply_text = "I cannot find that in the material."
            source = "heuristic"

    # 7) If differentiate, parse into left/right pairs; otherwise return normal
    if is_diff:
        left, right = [], []
        # Accept lines with pipes
        lines = [ln.strip() for ln in reply_text.splitlines() if "|" in ln]
        for ln in lines:
            # skip headers
            if ln.lower().startswith("term a") or ln.lower().startswith("differentiation") or ln.lower().startswith("topic:"):
                continue
            cells = [c.strip() for c in ln.split("|")]
            if len(cells) >= 2:
                left.append(cells[0])
                right.append(cells[1])
        if left and right:
            return {"type": "differentiate", "left": left, "right": right, "source": source, "raw": reply_text}
        # fallback: return normal text
        return {"type": "normal", "answer": reply_text, "source": source}
    else:
        return {"type": "normal", "answer": reply_text, "source": source}

# ---- Grader: evaluate student answer against material ----
def evaluate_answer(question: str, student_answer: str) -> Dict[str, Any]:
    """
    Prefer Gemini-based grader (JSON). If not available, use local Llama grader (attempt JSON),
    otherwise fallback to a heuristic that:
      - Extracts top sentences from SUBJECT_CONTENT that match question => ideal_answer
      - Computes a token-overlap based score (0-100)
    Returns: dict with keys {ok, score, feedback, ideal_answer(list), key_points_missed(list), found_in_material(bool)}
    """
    # If Gemini available, request JSON output using grading prompt
    if genai_client:
        grading_prompt = f"""
You are a strict examiner. Grade the student's answer ONLY using the material below.
If the material does not contain the answer, return JSON with score=0 and found_in_material=false.

MATERIAL:
--- START
{SUBJECT_CONTENT}
--- END

TASK:
1) Read the QUESTION and the STUDENT_ANSWER.
2) Grade from 0 to 100 based on factual overlap and coverage against the MATERIAL only.
3) Extract an IDEAL_ANSWER strictly from the MATERIAL (2-6 short bullet points).
4) Provide key_points_missed as an array of short bullets.
5) Return a JSON object only with keys:
   score (int), feedback (string up to 200 chars), ideal_answer (array), key_points_missed (array), found_in_material (true/false)

QUESTION: {question}
STUDENT_ANSWER: {student_answer}

Return ONLY JSON.
"""
        try:
            raw = generate_with_gemini(grading_prompt)
            # Extract JSON part
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = raw[start:end+1]
                parsed = json.loads(json_text)
                # sanitize
                score = int(parsed.get("score", 0)) if parsed.get("score") is not None else 0
                score = max(0, min(100, score))
                return {
                    "ok": True,
                    "score": score,
                    "feedback": parsed.get("feedback", ""),
                    "ideal_answer": parsed.get("ideal_answer", []),
                    "key_points_missed": parsed.get("key_points_missed", []),
                    "found_in_material": bool(parsed.get("found_in_material", False)),
                }
        except Exception as e:
            # fallback to offline heuristic
            pass

    # Offline heuristic grader
    # 1) Extract candidate ideal answer sentences from SUBJECT_CONTENT that overlap with the question
    sents = re.split(r'(?<=[.?!])\s+', SUBJECT_CONTENT)
    q_tokens = set(tokens(question))
    scored = []
    for s in sents:
        if not s.strip():
            continue
        score = overlap_ratio(s, question)
        if score > 0:
            scored.append((score, s.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    ideal = [s for _, s in scored[:5]]
    # 2) score student by overlap with ideal: compare tokens
    ideal_text = " ".join(ideal)
    if not ideal_text:
        return {"ok": True, "score": 0, "feedback": "Answer not found in material.", "ideal_answer": [], "key_points_missed": [], "found_in_material": False}
    # compute overlap
    overlap = overlap_ratio(student_answer, ideal_text)
    score = int(min(100, max(0, round(overlap * 100))))
    # identify missed keywords: compare tokens in ideal vs answered
    ideal_tokens = set(tokens(ideal_text))
    student_tokens = set(tokens(student_answer))
    missed = sorted(list(ideal_tokens - student_tokens))[:10]
    feedback = ""
    if score >= 90:
        feedback = "Excellent — you covered almost all key points."
    elif score >= 70:
        feedback = "Good — some key points are missing or incomplete."
    elif score >= 40:
        feedback = "Partial — many key points missing."
    else:
        feedback = "Poor — the answer lacks coverage of the material."
    return {"ok": True, "score": score, "feedback": feedback, "ideal_answer": ideal, "key_points_missed": missed, "found_in_material": True}

# ---- Flask routes ----

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
    Chat endpoint:
    Accepts JSON: { "message": "...", "query": "..." } (both supported)
    Also supports legacy form data "message"
    Returns:
      {"ok":True,"answers":[ {question, type, answer OR left/right, source} ]}
    """
    data = request.get_json(silent=True) or {}
    user_message = data.get("message") or data.get("query") or request.form.get("message") or ""
    user_message = (user_message or "").strip()
    if not user_message:
        return jsonify({"ok": False, "error": "No question received."})

    # allow multiple questions separated by newline or semicolon
    parts = [p.strip() for p in re.split(r"[;\n]+", user_message) if p.strip()]
    answers = []
    try:
        for p in parts:
            generated = generate_answer_for_question(p)
            # If saved type produce consistent payload
            if generated.get("type") == "saved":
                answers.append({"question": p, "type": "saved", "answer": generated.get("answer"), "source": "saved", "meta": generated.get("meta")})
            elif generated.get("type") == "differentiate":
                answers.append({"question": p, "type": "differentiate", "left": generated.get("left", []), "right": generated.get("right", []), "source": generated.get("source", "")})
            else:
                answers.append({"question": p, "type": "normal", "answer": generated.get("answer", ""), "source": generated.get("source", "")})
        return jsonify({"ok": True, "answers": answers})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Recommend saved Q&A entries given a partial query (5 words similar etc).
    Request JSON: { "text": "..." }
    Returns: {ok:True, suggestions: [entry,...]}
    """
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": True, "suggestions": []})
    # rank saved entries by jaccard
    results = []
    for e in INDEX["entries"]:
        q = e.get("question", "")
        score = jaccard_similarity(text, q)
        results.append((score, e))
    results.sort(key=lambda x: x[0], reverse=True)
    suggestions = [e for s, e in results[:10] if s > 0.10]  # threshold
    return jsonify({"ok": True, "suggestions": suggestions})

@app.route("/save_answer", methods=["POST"])
def save_answer():
    """
    Save Q&A to filesystem and index.
    Request: { question, answer, folder }
    """
    data = request.get_json(force=True)
    q = (data.get("question") or "").strip()
    a = (data.get("answer") or "").strip()
    folder = (data.get("folder") or "default").strip()
    if not q or not a:
        return jsonify({"ok": False, "error": "Missing question or answer."}), 400
    res = save_answer_to_folder(q, a, folder)
    return jsonify(res)

@app.route("/list_saved", methods=["GET"])
def list_saved():
    folder = request.args.get("folder")
    entries = list_saved_entries(folder) if folder else INDEX.get("entries", [])
    return jsonify({"ok": True, "entries": entries, "folders": list_saved_folders()})

@app.route("/evaluate", methods=["POST"])
def evaluate_route():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    student_answer = (data.get("student_answer") or "").strip()
    if not question or not student_answer:
        return jsonify({"ok": False, "error": "Missing question or answer."}), 400
    try:
        res = evaluate_answer(question, student_answer)
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Static file fallback (if needed)
@app.route("/static/<path:filename>")
def custom_static(filename):
    return send_from_directory(str(BASE_DIR / "static"), filename)

# Main
if __name__ == "__main__":
    # dev mode
    app.run(host="127.0.0.1", port=5000, debug=True)

