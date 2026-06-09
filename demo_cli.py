import sys
from pathlib import Path

from rag_engine import AssistenteRAG, RAGConfig

PASTA_EXEMPLOS = Path(__file__).parent / "documentos_exemplo"


def main():
    args = [a for a in sys.argv[1:]]
    so_recuperar = "--so-recuperar" in args
    args = [a for a in args if a != "--so-recuperar"]
    pergunta = args[0] if args else "O que é NDVI?"

    print("→ Inicializando assistente (carregando embeddings)...")
    assistente = AssistenteRAG(RAGConfig())

    # Indexa os documentos de exemplo (se ainda não houver nada)
    if not assistente.store.listar_documentos():
        print("→ Indexando documentos de exemplo...")
        for arquivo in PASTA_EXEMPLOS.glob("*"):
            if arquivo.suffix.lower() in (".txt", ".pdf", ".docx", ".md"):
                n = assistente.indexar_documento(str(arquivo))
                print(f"   • {arquivo.name}: {n} trechos")
    else:
        print("→ Índice já populado:", assistente.store.listar_documentos())

    print(f"\n→ Pergunta: {pergunta}\n")

    if so_recuperar:
        trechos = assistente.recuperar(pergunta)
        print("=== Trechos recuperados ===")
        for i, t in enumerate(trechos, 1):
            print(f"\n[{i}] fonte={t['origem']} distancia={t['distancia']:.3f}")
            print(t["texto"][:300])
        return

    resposta = assistente.responder(pergunta)
    print("=== Resposta ===")
    print(resposta.texto)
    print("\n=== Fontes ===")
    for i, f in enumerate(resposta.fontes, 1):
        print(f"[{i}] {f['origem']} (dist {f['distancia']:.3f})")


if __name__ == "__main__":
    main()
