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


@st.cache_data(ttl=3600)
def ottieni_modelli_disponibili(_client: genai.Client) -> list[str]:
    """Interroga l'API e restituisce i modelli Gemini attivi ordinati per priorità."""
    modelli_validi = []
    try:
        for m in _client.models.list():
            azioni = getattr(m, "supported_actions", None) or []
            nome = m.name.replace("models/", "")
            # Filtra solo i modelli Gemini che supportano la generazione testo
            if "generateContent" in azioni and "gemini" in nome.lower():
                # Esclude modelli speciali non adatti alla chat testuale (es. image, tts, embedding)
                if not any(x in nome.lower() for x in ["image", "tts", "embedding", "vision", "aqa"]):
                    modelli_validi.append(nome)
    except Exception:
        pass

    # Ordina dando priorità ai modelli Flash più recenti, poi Pro, poi gli altri
    def priorita(nome_modello: str) -> int:
        n = nome_modello.lower()
        if "flash" in n and "lite" not in n:
            return 0
        if "flash" in n:
            return 1
        if "pro" in n:
            return 2
        return 3

    modelli_validi.sort(key=priorita)
    return modelli_validi if modelli_validi else ["gemini-3.8-flash"]


def genera_risposta_resiliente(
    client: genai.Client,
    contents: list[types.Content],
    system_prompt: str
) -> str:
    """Tenta la generazione sui modelli disponibili con retry automatico e fallback."""
    modelli_da_provare = ottieni_modelli_disponibili(client)

    for modello in modelli_da_provare:
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
                # 404: modello non supportato -> passa subito al prossimo modello
                if e.code == 404:
                    break
                # 503 (sovraccarico) o 429 (rate limit) -> attende e riprova, poi cambia modello
                if e.code in (503, 429):
                    if tentativo == 0:
                        time.sleep(2)
                        continue
                    else:
                        break
                raise e

    raise RuntimeError(
        f"Nessun modello ha risposto correttamente (modelli testati: {', '.join(modelli_da_provare)})."
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
        with st.spinner(f"{scelta_personaggio} sta riflettendo..."):
            try:
                system_prompt = carica_prompt(info_char["prompt_file"])

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
                st.session_state.messages.pop()
