from __future__ import annotations

import json
import os
import streamlit as st
from groq import Groq


class LocalDataAgent:
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None):
        self.model = model
        self.api_key = (
            api_key
            or st.secrets.get("GROQ_API_KEY")
            or os.getenv("GROQ_API_KEY")
        )
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def status(self) -> tuple[bool, list[str], str]:
        if self.client and self.api_key:
            return True, [self.model], "API Groq Conectada"
        return False, [], "Falta GROQ_API_KEY en Secrets"

    def ask(self, question: str, context: dict) -> str:
        if not self.client:
            raise ValueError(
                "No se encontró la clave de API (GROQ_API_KEY). Configúrala en los Secrets de Streamlit."
            )

        system_prompt = (
            "Eres un asistente analítico de datos profesional. "
            "Responde a las preguntas del usuario basándote de manera estricta y precisa en el contexto JSON provisto."
        )

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Contexto de datos: {json.dumps(context, ensure_ascii=False)}\n\nPregunta: {question}",
                },
            ],
            model=self.model,
            temperature=0.2,
        )
        return response.choices[0].message.content


def offline_summary(context: dict) -> str:
    return (
        "El agente generativo no está disponible. "
        "Configura GROQ_API_KEY en los Secrets de Streamlit para habilitarlo."
    )
