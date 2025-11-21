import os
import base64
import json
from typing import Optional, Any, Dict

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY", "")  # your Action API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

app = FastAPI(title="OneDrive Image Analysis via Gemini")

class AnalyzeRequest(BaseModel):
    downloadUrl: str
    userQuery: str
    itemId: Optional[str] = None
    languageCode: Optional[str] = None
    contextHint: Optional[str] = None

class AnalyzeResponse(BaseModel):
    answer: str
    imageSummary: Optional[str] = ""
    extractedText: Optional[str] = ""
    detectedLanguages: Optional[list[str]] = []
    safetyNotes: Optional[str] = ""

@app.get("/health")
def health():
    return {"ok": True}

def _check_api_key(req: Request):
    auth = req.headers.get("authorization") or ""
    expected = f"Bearer {API_KEY}"
    if not API_KEY:
        # If you forgot to set API_KEY in Render, fail loudly.
        raise HTTPException(status_code=500, detail="Server API_KEY not configured")
    if auth.strip() != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

def _guess_mime(url: str) -> str:
    u = url.lower()
    if u.endswith(".png"):
        return "image/png"
    if u.endswith(".jpg") or u.endswith(".jpeg"):
        return "image/jpeg"
    if u.endswith(".webp"):
        return "image/webp"
    # default
    return "image/png"

def _extract_text_from_gemini(raw_text: str) -> Dict[str, Any]:
    """
    We ask Gemini to return JSON.
    If it returns plain text, we wrap it.
    """
    raw_text = raw_text.strip()

    # Try to find a JSON object in the response
    try:
        # direct parse
        return json.loads(raw_text)
    except Exception:
        pass

    # try to locate JSON substring
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start:end+1])
        except Exception:
            pass

    # fallback
    return {
        "answer": raw_text,
        "imageSummary": "",
        "extractedText": "",
        "detectedLanguages": [],
        "safetyNotes": "Gemini did not return JSON; fallback used."
    }

@app.post("/analyze-onedrive-image", response_model=AnalyzeResponse)
async def analyze_onedrive_image(payload: AnalyzeRequest, req: Request):
    _check_api_key(req)

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    # 1) Fetch image bytes from OneDrive pre-authenticated URL
    mime_type = _guess_mime(payload.downloadUrl)
    async with httpx.AsyncClient(timeout=60) as client:
        img_resp = await client.get(payload.downloadUrl)
        if img_resp.status_code != 200:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to fetch image from OneDrive (status {img_resp.status_code})"
            )
        image_bytes = img_resp.content

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # 2) Build a GENERAL prompt (not OCR-only)
    lang_line = ""
    if payload.languageCode:
        lang_line = f"\nRespond in language: {payload.languageCode}"

    hint_line = ""
    if payload.contextHint:
        hint_line = f"\nContext hint: {payload.contextHint}"

    system_prompt = f"""
You are a vision assistant. You will receive an image and a user question about it.
Answer the user's question based only on what is visible in the image.
Also extract any readable text.

Return a JSON object with fields:
- answer: string (direct answer to userQuery)
- imageSummary: string (brief neutral description)
- extractedText: string (all visible text, best effort)
- detectedLanguages: array of strings (languages of extracted text)
- safetyNotes: string (uncertainty or quality notes)

User question:
{payload.userQuery}
{hint_line}
{lang_line}
""".strip()

    # 3) Call Gemini generateContent with inline image data
    gemini_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    gemini_body = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64
                        }
                    }
                ]
            }
        ],
        # Encourage JSON output
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=120) as client:
        g_resp = await client.post(gemini_url, json=gemini_body)
        if g_resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini error {g_resp.status_code}: {g_resp.text}"
            )

    data = g_resp.json()
    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raw_text = json.dumps(data)

    parsed = _extract_text_from_gemini(raw_text)

    return AnalyzeResponse(
        answer=str(parsed.get("answer", "")),
        imageSummary=str(parsed.get("imageSummary", "")),
        extractedText=str(parsed.get("extractedText", "")),
        detectedLanguages=parsed.get("detectedLanguages", []) or [],
        safetyNotes=str(parsed.get("safetyNotes", "")),
    )
