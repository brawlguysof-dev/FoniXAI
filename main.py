import os
import sqlite3
import smtplib
from email.message import EmailMessage
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from pydantic import BaseModel, Field
from groq import Groq, APIError, RateLimitError

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "").strip()

DB_PATH = BASE_DIR / "users.db"


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                google_id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def send_registration_email(email, name):
    if not all([SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAIL]):
        print("Email bildirishnomasi sozlanmagan.")
        return

    message = EmailMessage()
    message["Subject"] = "Fonix AI: yangi foydalanuvchi"
    message["From"] = SMTP_USER
    message["To"] = NOTIFY_EMAIL
    message.set_content(
        "Fonix AI'da yangi foydalanuvchi ro'yxatdan o'tdi.\n\n"
        f"Ismi: {name}\n"
        f"Emaili: {email}\n"
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(message)
        print("Yangi foydalanuvchi haqida email yuborildi.")
    except Exception as error:
        print(f"Email yuborilmadi: {error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Fonix AI", version="3.0", lifespan=lifespan)

# SESSION_SECRET kuchli va tasodifiy bo'lishi kerak.
if not SESSION_SECRET:
    raise RuntimeError(
        ".env faylida SESSION_SECRET sozlanmagan."
    )

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=os.getenv("COOKIE_HTTPS_ONLY", "false").lower() == "true",
    max_age=60 * 60 * 24 * 7,
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

oauth = OAuth()

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url=(
            "https://accounts.google.com/.well-known/openid-configuration"
        ),
        client_kwargs={"scope": "openid email profile"},
    )


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=4000)
    image_data: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=12)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "Fonix AI",
        "model": MODEL,
    }


@app.get("/auth/me")
def auth_me(request: Request):
    user = request.session.get("user")
    if not user:
        return {"logged_in": False}

    return {
        "logged_in": True,
        "user": user,
    }


@app.get("/auth/google/login")
async def google_login(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth sozlanmagan.",
        )

    redirect_uri = str(request.url_for("google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google/callback", name="google_callback")
async def google_callback(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=503, detail="Google OAuth sozlanmagan.")

    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")

        if not user_info:
            response = await oauth.google.parse_id_token(request, token)
            user_info = response

        if not user_info:
            raise HTTPException(
                status_code=401,
                detail="Google hisobini tekshirib bo'lmadi.",
            )

        google_id = user_info.get("sub")
        email = user_info.get("email")
        name = user_info.get("name") or "Google foydalanuvchisi"
        email_verified = user_info.get("email_verified", False)

        if not google_id or not email or email_verified is not True:
            raise HTTPException(
                status_code=401,
                detail="Tasdiqlangan Google emaili talab qilinadi.",
            )

        with sqlite3.connect(DB_PATH) as db:
            existing = db.execute(
                "SELECT google_id FROM users WHERE google_id = ?",
                (google_id,),
            ).fetchone()

            if existing is None:
                db.execute(
                    "INSERT INTO users (google_id, email, name) "
                    "VALUES (?, ?, ?)",
                    (google_id, email, name),
                )
                is_new_user = True
            else:
                db.execute(
                    "UPDATE users SET email = ?, name = ? "
                    "WHERE google_id = ?",
                    (email, name, google_id),
                )
                is_new_user = False

        request.session.clear()
        request.session["user"] = {
            "id": google_id,
            "email": email,
            "name": name,
        }

        if is_new_user:
            send_registration_email(email, name)

        return RedirectResponse(url="/", status_code=303)

    except HTTPException:
        raise
    except Exception as error:
        print(f"Google login xatosi: {error}")
        raise HTTPException(
            status_code=400,
            detail="Google orqali kirishda xatolik. Sozlamalarni tekshiring.",
        )


@app.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.post("/api/chat")
def chat(request: Request, data: ChatRequest):
    if not request.session.get("user"):
        raise HTTPException(
            status_code=401,
            detail="AI chatidan foydalanish uchun Google orqali kiring.",
        )

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API kaliti topilmadi. .env faylini tekshiring.",
        )

    for message in data.messages:
        if message.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail="Xabar turi noto'g'ri.",
            )

        if message.image_data:
            if not message.image_data.startswith(
                (
                    "data:image/jpeg;base64,",
                    "data:image/png;base64,",
                    "data:image/webp;base64,",
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Faqat JPG, PNG yoki WEBP rasmlar qabul qilinadi.",
                )

            if len(message.image_data) > 2_000_000:
                raise HTTPException(
                    status_code=413,
                    detail="Rasm juda katta. Kichikroq rasm yuboring.",
                )

    if data.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Oxirgi xabar foydalanuvchidan bo'lishi kerak.",
        )

    image_indexes = [
        i for i, message in enumerate(data.messages)
        if message.image_data
    ]
    keep_images = set(image_indexes[-5:])

    messages = [
        {
            "role": "system",
            "content": (
                "Siz Fonix AI nomli yordamchisiz. "
                "Fonix AI loyihasini foydalanuvchi yaratgan. "
                "O'zingizni OpenAI yaratgan deb noto'g'ri tanishtirmang. "
                "Odatda o'zbek tilida va hurmat bilan javob bering. "
                "Rasm yuborilsa, uni tahlil qiling. "
                "Bilmagan narsangizni to'qib chiqarmang. "
                "Kod so'ralsa, kerak bo'lganda to'liq kod yozing."
            ),
        }
    ]

    for i, message in enumerate(data.messages):
        if message.image_data and i in keep_images:
            content = [
                {"type": "text", "text": message.content},
                {
                    "type": "image_url",
                    "image_url": {"url": message.image_data},
                },
            ]
        else:
            content = message.content

        messages.append({
            "role": message.role,
            "content": content,
        })

    try:
        client = Groq(
            api_key=API_KEY,
            timeout=60.0,
            max_retries=0,
        )

        result = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=1200,
        )

        return {
            "reply": (
                result.choices[0].message.content
                or "Hozir javob tayyorlay olmadim."
            )
        }

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="API limiti tugadi. Birozdan keyin qayta urinib ko'ring.",
        )

    except APIError as error:
        print(f"Groq API xatosi: {error}")
        raise HTTPException(
            status_code=502,
            detail="Model xatosi. Terminaldagi xabarni tekshiring.",
        )

    except Exception as error:
        print(f"Fonix AI xatosi: {error}")
        raise HTTPException(
            status_code=502,
            detail="Ulanishda xatolik yuz berdi.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
