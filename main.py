import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from groq import Groq, APIError, RateLimitError

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b").strip()

app = FastAPI(title="Fonix AI", version="2.0")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
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


@app.post("/api/chat")
def chat(data: ChatRequest):
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
                ("data:image/jpeg;base64,", "data:image/png;base64,",
                 "data:image/webp;base64,")
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

    # Groq vision modelida bir so'rov uchun ko'pi bilan 5 ta rasm.
    image_messages = [
        i for i, message in enumerate(data.messages)
        if message.image_data
    ]

    if len(image_messages) > 5:
        keep_images = set(image_messages[-5:])
    else:
        keep_images = set(image_messages)

    messages = [
        {
            "role": "system",
            "content": (
                "Siz Fonix AI nomli yordamchisiz. "
                "Fonix AI loyihasini foydalanuvchi yaratgan. "
                "Siz Groq API orqali ishlaydigan sun'iy intellekt modelisiz. "
                "O'zingizni OpenAI yaratgan deb noto'g'ri tanishtirmang. "
                "Modelning texnik kelib chiqishi so'ralsa, rost javob bering. "
                "Odatda o'zbek tilida va hurmat bilan javob bering. "
                "Foydalanuvchi rasm yuborsa, rasmni tahlil qilib, "
                "savoliga javob bering. "
                "Bilmagan narsangizni to'qib chiqarmang. "
                "Kod so'ralsa, kerak bo'lganda to'liq kod yozing."
            ),
        }
    ]

    for i, message in enumerate(data.messages):
        if message.image_data and i in keep_images:
            content = [
                {
                    "type": "text",
                    "text": message.content,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": message.image_data,
                    },
                },
            ]
        else:
            content = message.content

        messages.append(
            {
                "role": message.role,
                "content": content,
            }
        )

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

        answer = result.choices[0].message.content

        return {
            "reply": answer or "Hozir javob tayyorlay olmadim."
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
