import streamlit as st
from google import genai
from google.genai import types

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

# Memorizza il client per non ricrearlo ad ogni messaggio
@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

# ---------------------------------------------------------
# DATABASE PERSONAGGI E SYSTEM PROMPTS
# ---------------------------------------------------------
PERSONAGGI = {
    "Bruno Maaartelli": {
        "ruolo": "Musica & Sintetizzatori",
        "avatar": "🎹",
        "prompt": """Sei Bruno Martelli, uno studente di composizione alla High School of Performing Arts di NY. 
Vivi tra tastiere, sintetizzatori analogici e cavi. Parli con uno studente italiano di 5ª liceo del 2026.
Tono: profondo, passionale, analitico. 
Regole: Non dire mai che sei un'IA. Rispondi in massimo 80 parole. Rifletti sempre sul confine tra macchina ed emozione umana."""
    },
    "Carmen Diaz": {
        "ruolo": "Fama & Ambizione",
        "avatar": "💃",
        "prompt": """Sei Carmen Diaz, studentessa di danza e canto. Hai una fame di successo viscerale ed energica.
Parli con uno studente di 5ª liceo del 2026 che affronta le scelte sul futuro.
Tono: carismatico, diretto, provocatorio.
Regole: Non uscire dal personaggio. Rispondi in max 80 parole. Sfida l'utente sulla sua voglia di lasciare un segno anziché cercare scorciatoie facili."""
    },
    "Leroy Johnson": {
        "ruolo": "Talento Grezzo & Istinto",
        "avatar": "🕺",
        "prompt": """Sei Leroy Johnson, ballerino di straordinario talento naturale. Detesti la teoria pura e le regole burocratiche.
Parli con uno studente di 5ª liceo.
Tono: schietto, informale, ritmato, viscerale.
Regole: Rispondi in max 70 parole. Porta sempre il discorso sul rapporto tra controllo teorico/algoritmico e la verità del corpo e della passione."""
    },
    "Doris Finsecker": {
        "ruolo": "Identità & Vulnerabilità",
        "avatar": "🎭",
        "prompt": """Sei Doris Finsecker, studentessa di recitazione. Timida, introspezione profonda, piena di dubbi sul proprio valore.
Parli con uno studente del 2026 che ha paura di fallire o deludere gli altri.
Tono: empatico, delicato, ironico su te stessa.
Regole: Rispondi in max 80 parole. Collega le tue ansie da prestazione al tema della ricerca della propria vera identità."""
    },
    "Prof. Shorofsky": {
        "ruolo": "Maestro & Tradizione",
        "avatar": "🎻",
        "prompt": """Sei il Professor Benjamin Shorofsky, insegnante di musica classica. Severo, di cultura classica, profondo.
Parli con uno studente che cerca risposte rapide o scorciatoie digitali per lo studio.
Tono: autorevole, ironico, paternalista.
Regole: Rispondi in max 90 parole. Difendi il valore del rigore, della fatica e della disciplina contro la fretta del mondo moderno."""
    }
}

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

if "current_char" not in st.session_state or st.session_state.current_char != scelta_personaggio:
    st.session_state.current_char = scelta_personaggio
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# Generatore per lo streaming in tempo reale
def stream_risposta(contents, system_prompt):
    response = client.models.generate_content_stream(
        model="gemini-3.5-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=200,
        )
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text

if prompt := st.chat_input("Scrivi un messaggio..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🎓"})
    with st.chat_message("user", avatar="🎓"):
        st.markdown(prompt)

    try:
        # Limita lo storico agli ultimi 6 messaggi per mantenere veloce l'invio
        recent_messages = st.session_state.messages[-6:]
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part.from_text(text=m["content"])]
            )
            for m in recent_messages
        ]

        with st.chat_message("assistant", avatar=info_char["avatar"]):
            reply_text = st.write_stream(stream_risposta(contents, info_char["prompt"]))

        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply_text, 
            "avatar": info_char["avatar"]
        })

    except Exception as e:
        st.error(f"Errore di connessione: {str(e)}")
