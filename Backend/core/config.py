print(">>> CONFIG.PY FOI CARREGADO <<<")

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

print(">>> BASE_DIR:", BASE_DIR)
print(">>> ENV:", BASE_DIR / ".env")
print(">>> ENV EXISTE:", (BASE_DIR / ".env").exists())

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print(">>> GROQ KEY ENCONTRADA:", bool(GROQ_API_KEY))