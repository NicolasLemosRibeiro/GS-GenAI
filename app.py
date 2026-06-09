import os
import tempfile

import streamlit as st

from rag_engine import AssistenteRAG, RAGConfig

# Configuração da página
st.set_page_config(
    page_title="Assistente Espacial RAG",
    page_icon="🛰️",
    layout="wide",
)

# Estado da sessão

if "historico" not in st.session_state:
    st.session_state.historico = []   # lista de (pergunta, resposta, fontes)
if "assistente" not in st.session_state:
    st.session_state.assistente = None


def construir_assistente() -> AssistenteRAG:
    """Cria (ou recupera) o assistente, aplicando a config da barra lateral."""
    cfg = RAGConfig(
        llm_provider=st.session_state.get("provedor", "groq"),
        llm_model=st.session_state.get("modelo", "llama-3.3-70b-versatile"),
        api_key=st.session_state.get("api_key") or os.getenv("LLM_API_KEY"),
        base_url=st.session_state.get("base_url") or None,
    )
    if st.session_state.assistente is None:
        with st.spinner("Carregando modelo de embeddings (só na 1ª vez)..."):
            st.session_state.assistente = AssistenteRAG(cfg)
    else:
        # Atualiza apenas os parâmetros do LLM sem recarregar embeddings
        st.session_state.assistente.cfg.llm_provider = cfg.llm_provider
        st.session_state.assistente.cfg.llm_model = cfg.llm_model
        st.session_state.assistente.cfg.api_key = cfg.api_key
        st.session_state.assistente.cfg.base_url = cfg.base_url
    return st.session_state.assistente



# Barra lateral – Configuração e documentos

PROVEDORES = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "modelos": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        "ajuda": "Tem plano gratuito. Pegue a chave em console.groq.com/keys",
    },
    "openai": {
        "base_url": "",
        "modelos": [
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5-mini",
            "gpt-5.4-mini",
        ],
        "ajuda": "Pegue a chave em platform.openai.com/api-keys",
    },
}

with st.sidebar:
    st.header("⚙️ Configuração")

    provedor = st.selectbox(
        "Provedor do modelo generativo",
        options=list(PROVEDORES.keys()),
        format_func=lambda x: x.capitalize(),
        key="provedor",
    )
    info = PROVEDORES[provedor]
    st.caption(info["ajuda"])

    st.session_state.base_url = info["base_url"]

    # Seleção do modelo via dropdown (evita erro de digitação).
    # A última opção permite digitar um nome manualmente, se necessário.
    OPCAO_OUTRO = "✏️ Outro (digitar manualmente)"
    escolha = st.selectbox(
        "Modelo",
        options=info["modelos"] + [OPCAO_OUTRO],
        index=0,  # primeiro da lista = modelo recomendado
        key=f"modelo_sel_{provedor}",  # chave por provedor: reseta ao trocar
    )
    if escolha == OPCAO_OUTRO:
        modelo = st.text_input(
            "Nome exato do modelo",
            value=info["modelos"][0],
            key=f"modelo_custom_{provedor}",
            help="Use o nome exato como aparece na documentação do provedor.",
        )
    else:
        modelo = escolha
    # Disponibiliza o modelo escolhido para o restante do app
    st.session_state.modelo = modelo
    st.text_input(
        "Chave de API",
        type="password",
        key="api_key",
        help="A chave fica apenas na sua sessão; não é salva em disco.",
    )

    st.divider()
    st.header("📄 Documentos")

    arquivos = st.file_uploader(
        "Envie PDF, TXT ou DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("📥 Indexar documentos", use_container_width=True):
        if not arquivos:
            st.warning("Selecione ao menos um arquivo.")
        else:
            assistente = construir_assistente()
            total_chunks = 0
            barra = st.progress(0.0)
            for i, arq in enumerate(arquivos):
                # Salva o upload num arquivo temporário para leitura
                sufixo = os.path.splitext(arq.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
                    tmp.write(arq.getbuffer())
                    caminho_tmp = tmp.name
                # Renomeia para preservar o nome original como "origem"
                try:
                    n = assistente.indexar_documento(caminho_tmp, origem=arq.name)
                    total_chunks += n
                finally:
                    os.unlink(caminho_tmp)
                barra.progress((i + 1) / len(arquivos))
            st.success(f"Indexação concluída: {total_chunks} trechos adicionados.")

    # Lista de documentos já no índice
    if st.session_state.assistente is not None:
        docs = st.session_state.assistente.store.listar_documentos()
        if docs:
            st.caption("**Indexados:**")
            for d in docs:
                st.caption(f"• {d}")

        if st.button("🗑️ Limpar índice", use_container_width=True):
            st.session_state.assistente.store.limpar()
            st.session_state.historico = []
            st.success("Índice limpo.")



# Área principal – Chat

st.title("🛰️ Assistente Inteligente — Economia Espacial")
st.caption(
    "Pergunte sobre clima, satélites, agricultura inteligente, monitoramento "
    "ambiental, desastres naturais e exploração espacial. As respostas são "
    "baseadas nos documentos que você enviar (arquitetura RAG)."
)

# Renderiza o histórico
for pergunta, resposta, fontes in st.session_state.historico:
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        st.write(resposta)
        if fontes:
            with st.expander(f"📚 Fontes consultadas ({len(fontes)})"):
                for i, f in enumerate(fontes, 1):
                    st.markdown(
                        f"**Trecho {i}** — *{f['origem']}* "
                        f"(distância: {f['distancia']:.3f})"
                    )
                    st.caption(f["texto"][:400] + ("..." if len(f["texto"]) > 400 else ""))

# Caixa de entrada
pergunta = st.chat_input("Faça uma pergunta sobre os documentos...")

if pergunta:
    assistente = construir_assistente()

    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando nos documentos e gerando resposta..."):
                resposta = assistente.responder(pergunta)
            st.write(resposta.texto)
            if resposta.fontes:
                with st.expander(f"📚 Fontes consultadas ({len(resposta.fontes)})"):
                    for i, f in enumerate(resposta.fontes, 1):
                        st.markdown(
                            f"**Trecho {i}** — *{f['origem']}* "
                            f"(distância: {f['distancia']:.3f})"
                        )
                        st.caption(
                            f["texto"][:400] + ("..." if len(f["texto"]) > 400 else "")
                        )
            st.session_state.historico.append(
                (pergunta, resposta.texto, resposta.fontes)
            )
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")
            st.info(
                "Verifique se você configurou a chave de API na barra lateral "
                "e se já indexou ao menos um documento."
            )
