import time
from pathlib import Path
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA MOBILE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fame AI - L'Uomo e la Macchina",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Verifica presenza API Key e inizializzazione client
if "GEMINI_API_KEY" not in st.secrets:
    st.error("API Key non configurata nei Secrets di Streamlit.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ---------------------------------------------------------
# CONFIGURAZIONE PERSONAGGI (METADATI + FILE PROMPT ESTERNI)
# ---------------------------------------------------------
PROMPTS_DIR = Path(__file__).parent / "prompts"

PERSONAGGI = {
    "Bruno Martelli": {
        "ruolo": "Musica & Sintetizzatori",
        "avatar": "🎹",
        "prompt_file": "bruno_martelli.md",
    },
    "Carmen Diaz": {
        "ruolo": "Fama & Ambizione",
        "avatar": "💃",
        "prompt_file": "carmen_diaz.md",
    },
    "Leroy Johnson": {
        "ruolo": "Talento Grezzo & Istinto",
        "avatar": "🕺",
        "prompt_file": "leroy_johnson.md",
    },
    "Doris Finsecker": {
        "ruolo": "Identità & Vulnerabilità",
        "avatar": "🎭",
        "prompt_file": "doris_finsecker.md",
    },
    "Prof. Shorofsky": {
        "ruolo": "Maestro & Tradizione",
        "avatar": "🎻",
        "prompt_file": "prof_shorofsky.md",
    },
}


@st.cache_data
def carica_prompt(nome_file: str) -> str:
    """Legge le istruzioni del personaggio dal file Markdown esterno."""
    percorso = PROMPTS_DIR / nome_file
    if not percorso.exists():
        raise FileNotFoundError(f"File prompt non trovato: {percorso}")
    return percorso.read_text(encoding="utf-8").strip()


def genera_risposta_resiliente(
    client: genai.Client,
    contents: list[types.Content],
    system_prompt: str
) -> str:
    """Tenta la generazione con retry automatico e fallback su un altro modello."""
    modelli_da_provare = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.8-pro"]

    for modello in modelli_da_provare:
        # 2 tentativi per ciascun modello
        for tentativo in range(2):
            try:
                response = client.models.generate_content(
                    model=modello,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                    ),
                )
                return response.text
            except APIError as e:
                # Se è un errore 503 (sovraccarico) o 429 (troppi invii ravvicinati), aspetta e riprova
                if e.code in (503, 429):
                    if tentativo == 0:
                        time.sleep(2)
                        continue
                    else:
                        break
                else:
                    raise e

    raise RuntimeError("Tutti i modelli sono attualmente sovraccarichi. Riprova tra qualche secondo.")


# ---------------------------------------------------------
# INTERFACCIA GRAFICA E LOGICA DI CHAT
# ---------------------------------------------------------
st.title("🎭 Fame AI: L'Uomo e la Macchina")
st.caption("Laboratorio Interattivo di Dialogo Filosofico")

scelta_personaggio = st.selectbox(
    "Scegli il personaggio con cui dialogare:",
    options=list(PERSONAGGI.keys()),
    index=0
)

info_char = PERSONAGGI[scelta_personaggio]
st.info(f"{info_char['avatar']} **{scelta_personaggio}** — *{info_char['ruolo']}*")

# Reset automatico dello storico se si cambia personaggio
if "current_char" not in st.session_state or st.session_state.current_char != scelta_personaggio:
    st.session_state.current_char = scelta_personaggio
    st.session_state.messages = []

# Visualizzazione storico messaggi
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# Gestione dell'input dello studente e chiamata all'API
if prompt := st.chat_input("Scrivi un messaggio..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🎓"})
    with st.chat_message("user", avatar="🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=info_char["avatar"]):
        with st.spinner(f"{scelta_personaggio} sta riflettendo..."):
            try:
                system_prompt = carica_prompt(info_char["prompt_file"])

                # Ricostruzione conversazione completa per il nuovo SDK google-genai
                contents_gemini = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])],
                    )
                    for m in st.session_state.messages
                ]

                reply_text = genera_risposta_resiliente(client, contents_gemini, system_prompt)
                st.markdown(reply_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply_text,
                    "avatar": info_char["avatar"],
                })

            except Exception as e:
                st.error(f"Errore di connessione: {str(e)}")
                # Rimuove l'ultimo messaggio utente dallo storico in caso di errore
                st.session_state.messages.pop()
