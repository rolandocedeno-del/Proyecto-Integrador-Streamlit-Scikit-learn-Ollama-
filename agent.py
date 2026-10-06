from __future__ import annotations
import os
import json
from groq import Groq


class LocalDataAgent:
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def status(self) -> tuple[bool, list[str], str]:
        if self.client:
            return True, [self.model], "API Groq Conectada"
        return False, [], "Falta GROQ_API_KEY"

    def ask(self, question: str, context: dict) -> str:
        if not self.client:
            raise ValueError("API Key no encontrada.")

        system_prompt = "Eres un asistente analítico de datos. Responde basándote exclusivamente en el contexto provisto."
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexto: {json.dumps(context)}\nPregunta: {question}"}
            ],
            model=self.model,
            temperature=0.2,
        )
        return response.choices[0].message.content
