from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os, json, re, time, glob
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODEL = "openai/gpt-oss-120b"

REQUIRED_FIELDS = [
    "occupation",
    "current_state",
    "months_in_current_state",
    "monthly_income",
    "age",
    "gender",
    "has_bank_account",
    "contractor_registered_bocw",
]

FIELD_QUESTIONS = {
    "occupation": ("Aap kaun sa kaam karte hain?", "What work do you do?"),
    "current_state": ("Aap abhi kis state mein hain?", "Which state are you in?"),
    "months_in_current_state": ("Kitne mahine se yahan hain?", "How many months here?"),
    "monthly_income": ("Mahine ki kamai kitni hai?", "What is your monthly income?"),
    "age": ("Aapki umar kitni hai?", "What is your age?"),
    "gender": ("Aap purush hain ya mahila?", "Are you male or female?"),
    "has_bank_account": ("Kya aapka bank account hai?", "Do you have a bank account?"),
    "contractor_registered_bocw": ("Kya aapka thekedar aapko BOCW mein register karwa chuka hai?", "Has your contractor registered you under BOCW?"),
}

def get_missing_fields(profile: dict) -> list:
    return [f for f in REQUIRED_FIELDS if profile.get(f) is None]


def load_schemes():
    schemes = []
    possible_paths = [
        os.path.join(BASE_DIR, "..", "rules", "*.json"),
        os.path.join(BASE_DIR, "rules", "*.json"),
    ]
    for path in possible_paths:
        found = glob.glob(path)
        if found:
            for filepath in found:
                with open(filepath, "r", encoding="utf-8") as f:
                    schemes.append(json.load(f))
            print(f"Loaded {len(schemes)} schemes from {path}")
            return schemes
    print("WARNING: No schemes loaded. Check rules/ folder.")
    return schemes

SCHEMES_DATA = load_schemes()

def schemes_to_text():
    if not SCHEMES_DATA:
        return "No schemes loaded."
    lines = []
    for s in SCHEMES_DATA:
        conds = "; ".join(
            f"{c['field']} {c['operator']} {c['value']}"
            for c in s.get("eligibility_conditions", [])
        )
        docs = ", ".join(s.get("required_documents", []))
        lines.append(
            f"{s['name']}: {conds} | "
            f"Rs{s['monthly_benefit']}/mo | "
            f"{s['action']} | "
            f"DOCS:{docs} | "
            f"SOURCE:{s['official_source']} | "
            f"URL:{s.get('source_url','')}"
        )
    return "\n".join(lines)

def extract_json(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                return raw[start:i+1]
    return raw

def call_model(messages, max_tokens=1000):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens
    )
    raw = response.choices[0].message.content.strip()
    finish = response.choices[0].finish_reason
    print(f"MODEL ({finish}): '{raw[:300]}'")
    return raw, finish

class TextInput(BaseModel):
    text: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    history: list[ChatMessage]
    profile_so_far: dict = {}

class ComplaintInput(BaseModel):
    name: str = "Worker"
    months_in_current_state: str = "several"

@app.get("/")
def root():
    return FileResponse(os.path.join(BASE_DIR, "..", "frontend", "index.html"))

@app.get("/schemes")
def get_schemes():
    return {"schemes": SCHEMES_DATA}

@app.get("/test-model")
def test_model():
    try:
        raw, finish = call_model(
            [{"role": "user", "content": 'Return only: {"status":"ok"}'}],
            max_tokens=500
        )
        return {"raw": raw, "finish_reason": finish}
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
def analyze(input: TextInput):

    # Step 1: Extract profile
    profile = {}
    for attempt in range(3):
        try:
            raw, finish = call_model([
                {
                    "role": "system",
                    "content": "Extract worker facts and return ONLY a JSON object. No explanation."
                },
                {
                    "role": "user",
                    "content": (
                        "Text: " + input.text + "\n\n"
                        "Return this JSON with values filled in. Use null if unknown:\n"
                        '{"occupation":"","origin_state":"","current_state":"",'
                        '"months_in_current_state":0,"monthly_income":0,'
                        '"age":null,"has_bank_account":null,'
                        '"family_size":1,"existing_documents":[],'
                        '"contractor_registered_bocw":null,"gender":"male"}'
                    )
                }
            ], max_tokens=1000)

            if not raw:
                raise Exception("Empty response")

            try:
                profile = json.loads(extract_json(raw))
            except json.JSONDecodeError as je:
                print(f"RAW MODEL OUTPUT THAT FAILED TO PARSE:\n{raw}")
                raise je

            print(f"PROFILE OK: {profile}")
            break
        except Exception as e:
            print(f"PROFILE ERROR attempt {attempt}: {e}")
            if attempt == 2:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Profile extraction failed: {e}"}
                )
            time.sleep(0.5)

    # Step 2: Check eligibility
    schemes_text = schemes_to_text()

    for attempt in range(3):
        try:
            raw, finish = call_model([
                {
                    "role": "system",
                    "content": (
                        "You check welfare scheme eligibility. "
                        "Return ONLY a JSON object. "
                        "Keep reason and action fields under 10 words each. "
                        "Keep hindi_summary under 20 words. "
                        "Be extremely concise."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "SCHEMES:\n" + schemes_text + "\n\n"
                        "WORKER: " + json.dumps(profile) + "\n\n"
                        "Check eligibility for each scheme. "
                        "For each eligible scheme, copy its DOCS list into documents_needed.\n"
                        "contractor_violation=true if construction worker AND contractor_registered_bocw=false.\n"
                        "Return ONLY this JSON structure:\n"
                        '{"eligible_schemes":[{"scheme_name":"","reason":"","monthly_benefit":0,'
                        '"action":"","urgency":"high","confidence":"HIGH","source_citation":"",'
                        '"source_url":"","documents_needed":[]}],'
                        '"ineligible_schemes":[{"scheme_name":"","reason":""}],'
                        '"total_monthly_benefit":0,"priority_action":"","contractor_violation":false,'
                        '"hindi_summary":""}'
                    )
                }
            ], max_tokens=2000)

            if not raw:
                raise Exception("Empty response")

            if finish == "length":
                print("WARNING: eligibility response cut off - trying to parse partial")

            eligibility = json.loads(extract_json(raw))
            eligibility.setdefault("eligible_schemes", [])
            eligibility.setdefault("ineligible_schemes", [])
            eligibility.setdefault("contractor_violation", False)
            eligibility.setdefault("total_monthly_benefit", 0)
            eligibility.setdefault("priority_action", "")
            eligibility.setdefault("hindi_summary", "")
            return {"profile": profile, "eligibility": eligibility}

        except Exception as e:
            print(f"ELIGIBILITY ERROR attempt {attempt}: {e}")
            if attempt < 2:
                time.sleep(0.5)
                continue
            return JSONResponse(
                status_code=500,
                content={"error": f"Eligibility check failed: {e}"}
            )

@app.post("/chat")
def chat(input: ChatInput):
    missing = get_missing_fields(input.profile_so_far)
    example = (
        '{"question_hindi":"Aap kaun sa kaam karte hain?",'
        '"question_english":"What work do you do?",'
        '"profile_complete":false,'
        '"extracted_profile":{"occupation":null,"current_state":null,'
        '"origin_state":null,"months_in_current_state":null,'
        '"monthly_income":null,"age":null,"gender":null,'
        '"has_bank_account":null,"family_size":1,"existing_documents":[],'
        '"contractor_registered_bocw":null}}'
    )

    if missing:
        next_field = missing[0]
        q_hi, q_en = FIELD_QUESTIONS[next_field]
        field_instruction = (
            f"You MUST ask about: {next_field}. "
            f"Use this exact question (or a close natural variation): "
            f"Hindi: \"{q_hi}\" / English: \"{q_en}\"\n"
        )
    else:
        field_instruction = "All required facts are collected. Set profile_complete to true.\n"

    system = (
        "You are SahayakAI helping migrant workers find benefits.\n"
        + field_instruction +
        "Extract facts from answers. Never repeat a question already answered.\n"
        "ALWAYS return ONLY a JSON object. No other text.\n"
        "Format:\n" + example + "\n"
        "When complete: profile_complete=true, "
        'question_hindi="Shukriya! Ab eligibility check karta hoon..."'
    )

    for attempt in range(3):
        raw = ""
        try:
            messages = [{"role": "system", "content": system}]
            messages += [{"role": m.role, "content": m.content} for m in input.history]
            raw, finish = call_model(messages, max_tokens=500)

            if not raw:
                raise Exception("Empty response")

            result = json.loads(extract_json(raw))

            merged = dict(input.profile_so_far)
            extracted = result.get("extracted_profile", {})
            for key, val in extracted.items():
                if key == "family_size":
                    merged[key] = max(merged.get(key, 1), val or 1)
                elif key == "existing_documents":
                    existing = merged.get("existing_documents", [])
                    merged[key] = list(set(existing + (val or [])))
                elif val is not None:
                    merged[key] = val

            result["updated_profile"] = merged
            still_missing = get_missing_fields(merged)
            if still_missing:
                nf = still_missing[0]
                q_hi, q_en = FIELD_QUESTIONS[nf]
                result["question_hindi"] = q_hi
                result["question_english"] = q_en
                result["profile_complete"] = False
            else:
                result["profile_complete"] = True
            return result

        except Exception as e:
            print(f"CHAT ERROR attempt {attempt}: {e}")
            if attempt < 2:
                time.sleep(0.5)
                continue

            p = input.profile_so_far
            missing_fallback = get_missing_fields(p)
            next_field = missing_fallback[0] if missing_fallback else "occupation"
            q_hi, q_en = FIELD_QUESTIONS[next_field]

            return {
                "question_hindi": q_hi,
                "question_english": q_en,
                "profile_complete": len(missing_fallback) == 0,
                "updated_profile": p
            }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        content = await audio.read()
        transcription = client.audio.transcriptions.create(
            file=(audio.filename or "audio.webm", content, "audio/webm"),
            model="whisper-large-v3-turbo",
            language="hi"
        )
        return {"text": transcription.text}
    except Exception as e:
        print(f"TRANSCRIBE ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-complaint")
def generate_complaint(profile: ComplaintInput):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4

        filename = "complaint_letter.pdf"
        filepath = os.path.join(BASE_DIR, filename)
        c = canvas.Canvas(filepath, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "COMPLAINT UNDER BOCW ACT 1996")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 90, "To,")
        c.drawString(50, height - 110, "The District Labour Officer,")
        c.drawString(50, height - 130, "Bengaluru Urban District")
        c.drawString(50, height - 170, "Subject: Non-registration under BOCW Act 1996")
        c.drawString(50, height - 210, "Respected Sir/Madam,")
        c.drawString(50, height - 240, f"I, {profile.name}, a construction worker in Bengaluru, Karnataka,")
        c.drawString(50, height - 260, f"have been working here for {profile.months_in_current_state} months.")
        c.drawString(50, height - 290, "My contractor has not registered me under the BOCW Welfare Board.")
        c.drawString(50, height - 330, "Legal basis: Section 7, BOCW Act 1996.")
        c.drawString(50, height - 350, "Penalty for non-compliance: up to Rs 1,00,000.")
        c.drawString(50, height - 390, "I request:")
        c.drawString(50, height - 410, "1. Action against my contractor for this violation.")
        c.drawString(50, height - 430, "2. My registration under Karnataka BOCW Welfare Board.")
        c.drawString(50, height - 450, "3. Access to benefits I am legally entitled to.")
        c.drawString(50, height - 490, "Yours faithfully,")
        c.drawString(50, height - 520, profile.name)
        c.drawString(50, height - 540, f"Date: {time.strftime('%d/%m/%Y')}")
        c.drawString(50, height - 560, "Place: Bengaluru, Karnataka")
        c.save()

        return {"filename": filename, "url": "/complaint_letter.pdf"}
    except Exception as e:
        print(f"COMPLAINT ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/complaint_letter.pdf")
def serve_complaint():
    filepath = os.path.join(BASE_DIR, "complaint_letter.pdf")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/pdf", filename="complaint_letter.pdf")
    return JSONResponse(status_code=404, content={"error": "Complaint letter not generated yet. Please click the Generate Complaint Letter button first."})
