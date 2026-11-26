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

# NOTA: non passiamo timeout numerici ambigui (in alcune versioni SDK sono in ms!)
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
def ottieni_modelli_reali(_client: genai.Client) -> tuple[list[str], str]:
    """Interroga l'API e restituisce i modelli Gemini reali disponibili sull'account."""
    modelli_validi = []
    escludi = ["image", "tts", "embedding", "vision", "aqa", "1.0", "1.5"]

    try:
        for m in _client.models.list():
            nome = m.name.replace("models/", "")
            n_low = nome.lower()
            # Verifica compatibilità con generateContent (supporta sia supported_actions che supported_generation_methods)
            azioni = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            
            if "gemini" in n_low and not any(x in n_low for x in escludi):
                if not azioni or "generateContent" in azioni:
                    modelli_validi.append(nome)
    except Exception as e:
        return [], f"Errore durante client.models.list(): {e}"

    def priorita(nome_modello: str) -> int:
        n = nome_modello.lower()
        if "flash" in n and "lite" not in n and "preview" not in n and "exp" not in n:
            return 0
        if "flash" in n and "lite" not in n:
            return 1
        if "flash" in n:
            return 2
        if "pro" in n:
            return 3
        return 4

    modelli_validi.sort(key=priorita)
    return modelli_validi[:4], ""


def genera_risposta_resiliente(
    client: genai.Client,
    contents: list[types.Content],
    system_prompt: str,
    status_placeholder
) -> str:
    """Tenta la generazione sui modelli reali rilevati e riporta errori dettagliati se fallisce."""
    modelli_da_provare, err_lista = ottieni_modelli_reali(client)

    if not modelli_da_provare:
        raise RuntimeError(f"Impossibile recuperare la lista modelli dall'API. Dettaglio: {err_lista}")

    errori_riscontrati = []

    for modello in modelli_da_provare:
        status_placeholder.caption(f"⏳ Risposta in corso con `{modello}`...")
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
            errori_riscontrati.append(f"`{modello}` -> Errore API {e.code}: {e.message}")
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
                except Exception as e2:
                    errori_riscontrati.append(f"`{modello}` (retry) -> {e2}")
            continue

        except Exception as e:
            errori_riscontrati.append(f"`{modello}` -> {type(e).__name__}: {e}")
            continue

    status_placeholder.empty()
    dettaglio_errori = "\n".join(f"- {err}" for err in errori_riscontrati)
    raise RuntimeError(f"Tutti i modelli hanno fallito:\n{dettaglio_errori}")


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
            st.error(f"Errore di connessione:\n{str(e)}")
            st.session_state.messages.pop()
