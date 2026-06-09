from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Dependências de terceiros

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Leitura de documentos
from pypdf import PdfReader
import docx  # python-docx

# Cliente LLM (compatível com OpenAI: serve para OpenAI, Groq, etc.)
from openai import OpenAI

# 1. CONFIGURAÇÃO

@dataclass
class RAGConfig:
    """Parâmetros centrais da solução. Tudo num único lugar."""

    #  Embeddings (rodam localmente, sem custo) 
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    #  Chunking 
    chunk_size: int = 800          # caracteres por chunk
    chunk_overlap: int = 120       # sobreposição entre chunks

    #  Recuperação 
    top_k: int = 4                 # nº de trechos recuperados por pergunta

    #  Vector store 
    persist_dir: str = "./vector_store"
    collection_name: str = "documentos_espaciais"

    #  Modelo generativo (LLM) 
    # Compatível com a API da OpenAI. Funciona com:
    #   - OpenAI   -> base_url None,  modelo "gpt-4.1-mini"
    #   - Groq     -> base_url "https://api.groq.com/openai/v1", modelo "llama-3.3-70b-versatile"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    temperature: float = 0.2



# 2. LEITURA DE DOCUMENTOS  (PDF / TXT / DOCX)

def carregar_pdf(caminho: str) -> str:
    """Extrai o texto de um arquivo PDF."""
    reader = PdfReader(caminho)
    paginas = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(paginas)


def carregar_docx(caminho: str) -> str:
    """Extrai o texto de um arquivo DOCX."""
    documento = docx.Document(caminho)
    return "\n".join(par.text for par in documento.paragraphs)


def carregar_txt(caminho: str) -> str:
    """Lê um arquivo de texto puro."""
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def carregar_documento(caminho: str) -> str:
    """Roteia o arquivo para o leitor correto conforme a extensão."""
    ext = Path(caminho).suffix.lower()
    if ext == ".pdf":
        return carregar_pdf(caminho)
    if ext == ".docx":
        return carregar_docx(caminho)
    if ext in (".txt", ".md"):
        return carregar_txt(caminho)
    raise ValueError(f"Formato não suportado: {ext} (use PDF, TXT ou DOCX)")



# 3. CHUNKING  (divisão do texto em trechos)

def limpar_texto(texto: str) -> str:
    """Normaliza espaços e quebras de linha em excesso."""
    texto = re.sub(r"\r", "\n", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def dividir_em_chunks(texto: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Divide o texto em janelas de `chunk_size` caracteres com `overlap`
    de sobreposição. A sobreposição evita cortar uma ideia ao meio entre
    dois chunks, melhorando a recuperação.
    """
    texto = limpar_texto(texto)
    if not texto:
        return []

    chunks: List[str] = []
    inicio = 0
    passo = max(1, chunk_size - overlap)
    while inicio < len(texto):
        fim = inicio + chunk_size
        chunks.append(texto[inicio:fim].strip())
        inicio += passo
    return [c for c in chunks if c]



# 4. EMBEDDINGS  (sentence-transformers, local)

class GeradorEmbeddings:
    """Encapsula o modelo de embeddings local."""

    def __init__(self, model_name: str):
        # Carrega o modelo apenas uma vez (operação custosa).
        self.model = SentenceTransformer(model_name)

    def gerar(self, textos: List[str]) -> List[List[float]]:
        vetores = self.model.encode(
            textos,
            normalize_embeddings=True,  # facilita busca por similaridade de cosseno
            show_progress_bar=False,
        )
        return [v.tolist() for v in vetores]



# 5. VECTOR STORE  (ChromaDB)

class VectorStore:
    """Camada de persistência e busca vetorial sobre o ChromaDB."""

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.client = chromadb.PersistentClient(
            path=cfg.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def adicionar(self, chunks: List[str], embeddings: List[List[float]], origem: str):
        """Adiciona os chunks de um documento ao índice."""
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadados = [{"origem": origem, "trecho": i} for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadados,
        )

    def buscar(self, embedding_consulta: List[float], top_k: int) -> List[dict]:
        """Recupera os `top_k` trechos mais semelhantes à pergunta."""
        if self.collection.count() == 0:
            return []
        resultado = self.collection.query(
            query_embeddings=[embedding_consulta],
            n_results=min(top_k, self.collection.count()),
        )
        documentos = resultado.get("documents", [[]])[0]
        metadados = resultado.get("metadatas", [[]])[0]
        distancias = resultado.get("distances", [[]])[0]
        return [
            {"texto": d, "origem": m.get("origem", "?"), "distancia": dist}
            for d, m, dist in zip(documentos, metadados, distancias)
        ]

    def listar_documentos(self) -> List[str]:
        """Retorna a lista de documentos (origens) já indexados."""
        if self.collection.count() == 0:
            return []
        dados = self.collection.get(include=["metadatas"])
        origens = {m.get("origem", "?") for m in dados["metadatas"]}
        return sorted(origens)

    def limpar(self):
        """Apaga toda a coleção (reset do índice)."""
        self.client.delete_collection(self.cfg.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )



# 6. PIPELINE RAG  (orquestra tudo)

PROMPT_SISTEMA = (
    "Você é um assistente especializado na nova economia espacial "
    "(clima, satélites, agricultura inteligente, monitoramento ambiental, "
    "desastres naturais e exploração espacial). "
    "Responda SEMPRE em português do Brasil, de forma clara e objetiva, "
    "usando EXCLUSIVAMENTE o contexto fornecido. "
    "Se a resposta não estiver no contexto, diga honestamente que não "
    "encontrou a informação nos documentos. Nunca invente dados."
)


@dataclass
class Resposta:
    texto: str
    fontes: List[dict] = field(default_factory=list)


class AssistenteRAG:
    """Fachada principal: indexa documentos e responde perguntas via RAG."""

    def __init__(self, cfg: Optional[RAGConfig] = None):
        self.cfg = cfg or RAGConfig()
        self.embeddings = GeradorEmbeddings(self.cfg.embedding_model_name)
        self.store = VectorStore(self.cfg)

    # -- Indexação --
    def indexar_documento(self, caminho: str, origem: Optional[str] = None) -> int:
        """Lê, divide, embeda e indexa um documento. Retorna nº de chunks.

        `origem` permite preservar o nome original do arquivo quando o
        caminho aponta para um arquivo temporário (ex.: upload via web).
        """
        texto = carregar_documento(caminho)
        chunks = dividir_em_chunks(texto, self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not chunks:
            return 0
        vetores = self.embeddings.gerar(chunks)
        self.store.adicionar(chunks, vetores, origem=origem or Path(caminho).name)
        return len(chunks)

    # -- Recuperação --
    def recuperar(self, pergunta: str) -> List[dict]:
        emb = self.embeddings.gerar([pergunta])[0]
        return self.store.buscar(emb, self.cfg.top_k)

    # -- Geração --
    def _cliente_llm(self) -> OpenAI:
        if not self.cfg.api_key:
            raise RuntimeError(
                "Chave de API do modelo generativo não configurada. "
                "Defina a variável de ambiente LLM_API_KEY."
            )
        kwargs = {"api_key": self.cfg.api_key}
        if self.cfg.base_url:
            kwargs["base_url"] = self.cfg.base_url
        return OpenAI(**kwargs)

    def _montar_contexto(self, trechos: List[dict]) -> str:
        blocos = []
        for i, t in enumerate(trechos, 1):
            blocos.append(f"[Trecho {i} | fonte: {t['origem']}]\n{t['texto']}")
        return "\n\n".join(blocos)

    def responder(self, pergunta: str) -> Resposta:
        """Fluxo RAG completo: recupera -> monta contexto -> gera resposta."""
        trechos = self.recuperar(pergunta)

        if not trechos:
            return Resposta(
                texto="Nenhum documento foi indexado ainda. "
                "Envie documentos antes de fazer perguntas.",
                fontes=[],
            )

        contexto = self._montar_contexto(trechos)
        mensagem_usuario = (
            f"Contexto recuperado dos documentos:\n\n{contexto}\n\n"
            f"Pergunta do usuário: {pergunta}\n\n"
            "Responda usando apenas o contexto acima."
        )

        cliente = self._cliente_llm()
        mensagens = [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": mensagem_usuario},
        ]
        completion = self._chamar_llm(cliente, mensagens)
        texto = completion.choices[0].message.content.strip()
        return Resposta(texto=texto, fontes=trechos)

    def _chamar_llm(self, cliente: OpenAI, mensagens: list):
        """Chama o modelo generativo.

        Alguns modelos mais novos (ex.: linha GPT-5 da OpenAI) só aceitam o
        valor padrão de `temperature` (1). Se o modelo recusar a temperatura
        customizada, a chamada é refeita automaticamente sem esse parâmetro.
        """
        try:
            return cliente.chat.completions.create(
                model=self.cfg.llm_model,
                temperature=self.cfg.temperature,
                messages=mensagens,
            )
        except Exception as e:
            if "temperature" in str(e).lower():
                # Modelo não aceita temperatura customizada: usa o padrão.
                return cliente.chat.completions.create(
                    model=self.cfg.llm_model,
                    messages=mensagens,
                )
            raise