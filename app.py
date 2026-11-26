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

if "GEMINI_API_KEY" not in st.secrets:
    st.error("API Key non configurata nei Secrets di Streamlit.")
    st.stop()

# Timeout di 20 secondi per evitare che l'app resti appesa all'infinito
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"],
    http_options={"timeout": 20}
)

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


@st.cache_data(ttl=3600)
def ottieni_modelli_disponibili(_client: genai.Client) -> list[str]:
    """Recupera al massimo i 3 migliori modelli Gemini attivi e stabili."""
    modelli_validi = []
    parole_da_escludere = ["image", "tts", "embedding", "vision", "aqa", "exp", "1.0", "1.5", "2.0"]

    try:
        for m in _client.models.list():
            azioni = getattr(m, "supported_actions", None) or []
            nome = m.name.replace("models/", "")
            n_low = nome.lower()

            if "generateContent" in azioni and "gemini" in n_low:
                if not any(x in n_low for x in parole_da_escludere):
                    modelli_validi.append(nome)
    except Exception:
        pass

    def priorita(nome_modello: str) -> int:
        n = nome_modello.lower()
        if "flash" in n and "lite" not in n and "preview" not in n:
            return 0
        if "flash" in n and "lite" not in n:
            return 1
        if "flash" in n:
            return 2
        if "pro" in n:
            return 3
        return 4

    modelli_validi.sort(key=priorita)
    # Restituisce al massimo i primi 3 modelli per evitare attese lunghe
    return modelli_validi[:3] if modelli_validi else ["gemini-3.8-flash"]


def genera_risposta_resiliente(
    client: genai.Client,
    contents: list[types.Content],
    system_prompt: str,
    status_placeholder
) -> str:
    """Tenta la generazione sui migliori 3 modelli disponibili con feedback visivo."""
    modelli_da_provare = ottieni_modelli_disponibili(client)

    for modello in modelli_da_provare:
        status_placeholder.caption(f"⏳ Connessione a `{modello}`...")
        try:
            response = client.models.generate_content(
                model=modello,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            status_placeholder.empty()
            return response.text

        except APIError as e:
            # Se sovraccarico (503), fa un solo rapido retry dopo 1.5s, poi cambia modello
            if e.code == 503:
                time.sleep(1.5)
                try:
                    response = client.models.generate_content(
                        model=modello,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                        ),
                    )
                    status_placeholder.empty()
                    return response.text
                except Exception:
                    continue
            # Per 404, 429 o altri errori passa subito al modello successivo
            continue
        except Exception:
            # Gestisce eventuali timeout di rete passando al modello successivo
            continue

    status_placeholder.empty()
    raise RuntimeError(
        f"Nessun modello ha risposto in tempo utile (testati: {', '.join(modelli_da_provare)})."
    )


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
        status_box = st.empty()
        try:
            system_prompt = carica_prompt(info_char["prompt_file"])

            contents_gemini = [
                types.Content(
                    role="user" if m["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=m["content"])],
                )
                for m in st.session_state.messages
            ]

            reply_text = genera_risposta_resiliente(
                client, contents_gemini, system_prompt, status_box
            )
            st.markdown(reply_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply_text,
                "avatar": info_char["avatar"],
            })

        except Exception as e:
            status_box.empty()
            st.error(f"Errore di connessione: {str(e)}")
            st.session_state.messages.pop()
