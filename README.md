# 🛰️ Assistente Inteligente com RAG — Dados e Documentos Espaciais

Projeto **GS2026.1 — Inteligência Artificial Generativa**.

## 👥 Integrantes do grupo

- Nícolas Lemos Ribeiro — 553273
- Murilo de Faria Benhossi — 562358
- Luis Fernando de Oliveira Salgado — 561401

Assistente que responde perguntas sobre a **nova economia espacial** (clima,
satélites, agricultura inteligente, monitoramento ambiental, desastres
naturais e exploração espacial) usando arquitetura **RAG
(Retrieval-Augmented Generation)**: antes de responder, o modelo consulta os
documentos enviados pelo usuário.

---

## 🧱 Arquitetura

```
┌──────────────┐   upload    ┌─────────────────┐   chunks   ┌──────────────┐
│   Usuário    │ ──────────► │  Leitor de docs │ ─────────► │   Chunking   │
│ (Streamlit)  │  PDF/TXT/   │ pypdf/python-   │            │  800 chars / │
│              │   DOCX      │     docx        │            │ overlap 120  │
└──────┬───────┘             └─────────────────┘            └──────┬───────┘
       │ pergunta                                                  │
       ▼                                                           ▼
┌──────────────┐   embedding   ┌─────────────────┐  embeddings ┌──────────────┐
│  Embedding   │ ◄──────────── │   Pergunta      │             │  Embeddings  │
│ all-MiniLM-  │               └─────────────────┘             │ (mesmo modelo)│
│   L6-v2      │                                               └──────┬───────┘
└──────┬───────┘                                                      │ indexa
       │ busca por similaridade (cosseno)                             ▼
       ▼                                                       ┌──────────────┐
┌──────────────┐   top-k trechos   ┌─────────────────┐        │  Vector Store│
│ Vector Store │ ────────────────► │  Monta contexto │◄───────│  (ChromaDB)  │
│  (ChromaDB)  │                   │   + prompt      │        └──────────────┘
└──────────────┘                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐    resposta   ┌─────────┐
                                   │  Modelo Gerativo│ ────────────► │ Usuário │
                                   │ (LLM via API)   │  + fontes     └─────────┘
                                   └─────────────────┘
```

---

## 🧰 Ferramentas e tecnologias

| Camada | Tecnologia |
|---|---|
| Interface | **Streamlit** |
| Leitura de documentos | **pypdf** (PDF), **python-docx** (DOCX), nativo (TXT) |
| Embeddings | **sentence-transformers** — `all-MiniLM-L6-v2` (local, gratuito) |
| Vector store | **ChromaDB** (persistente, busca por cosseno) |
| Modelo generativo | **LLM via API compatível com OpenAI** (Groq grátis ou OpenAI) |
| Linguagem | **Python 3.10+** |

---

## ⚙️ Instalação

```bash
# 1. Clone o repositório e entre na pasta
cd space-rag-assistant

# 2. (Recomendado) crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## 🔑 Chave de API do modelo generativo (gratuita)

O assistente precisa de um LLM para gerar a resposta final. A opção mais
simples e **gratuita** é o **Groq**:

1. Acesse <https://console.groq.com/keys>
2. Crie uma conta e gere uma API key
3. Use essa chave na barra lateral do app (ou exporte como variável de
   ambiente, ver abaixo)

> Também funciona com a OpenAI: basta selecionar "openai" como provedor e
> informar a chave de <https://platform.openai.com/api-keys>.

---

## ▶️ Como executar

### Interface web (principal)

```bash
streamlit run app.py
```

Depois, no navegador:
1. Na barra lateral, escolha o provedor (Groq), confira o modelo e cole a
   chave de API.
2. Faça upload de documentos (PDF, TXT ou DOCX) e clique em **Indexar**.
3. Digite perguntas no chat. Cada resposta mostra as **fontes** usadas.

> A pasta `documentos_exemplo/` traz dois textos sobre satélites,
> agricultura e desastres para você testar rapidamente.

### Demonstração por linha de comando (opcional)

```bash
export LLM_API_KEY="sua_chave_groq"
python demo_cli.py "O que é NDVI e para que serve?"

# Só recuperação, sem chamar o LLM:
python demo_cli.py --so-recuperar "como detectar enchentes por satélite?"
```

---

## 🔄 Fluxo RAG implementado

1. **Ingestão** — o documento é lido e o texto extraído.
2. **Chunking** — o texto é dividido em trechos de 800 caracteres com 120 de
   sobreposição, para não cortar ideias ao meio.
3. **Embeddings** — cada trecho é convertido em vetor pelo
   `all-MiniLM-L6-v2` (executado localmente).
4. **Indexação** — os vetores são armazenados no ChromaDB com metadados
   (arquivo de origem).
5. **Recuperação** — a pergunta é vetorizada e os `top_k=4` trechos mais
   similares (distância de cosseno) são recuperados.
6. **Geração** — os trechos viram contexto de um prompt enviado ao LLM, que
   responde **apenas com base no contexto**, citando as fontes.

---

## 📁 Estrutura do projeto

```
space-rag-assistant/
├── app.py                  # Interface Streamlit
├── rag_engine.py           # Núcleo RAG (leitura, embeddings, vetor, geração)
├── demo_cli.py             # Demonstração/teste por linha de comando
├── requirements.txt        # Dependências
├── README.md               # Este arquivo
└── documentos_exemplo/     # Documentos para teste
    ├── satelites_monitoramento.txt
    └── agricultura_clima_desastres.txt
```

---

## ⚠️ Limitações

- A qualidade da resposta depende dos documentos indexados (RAG não inventa
  conteúdo que não esteja nas fontes).
- O modelo de embeddings é multilíngue mas otimizado para inglês; textos em
  português funcionam bem, mas termos muito específicos podem reduzir a
  precisão da recuperação.
- Não há OCR: PDFs escaneados (imagem) não têm texto extraível sem um passo
  adicional de OCR.
- Chunking por caracteres é simples; não respeita fronteiras semânticas
  (parágrafos/seções) de forma sofisticada.
- A primeira execução baixa o modelo de embeddings (~90 MB) e exige internet.

## 🚀 Melhorias futuras

- Chunking semântico (por sentença/parágrafo) e *re-ranking* dos trechos.
- Suporte a OCR para PDFs escaneados.
- Citação com número de página e destaque do trecho exato.
- Histórico de conversa com memória (perguntas de acompanhamento).
- Avaliação automática da qualidade das respostas (RAGAS).
- Deploy em nuvem (Streamlit Community Cloud / Hugging Face Spaces).
```
