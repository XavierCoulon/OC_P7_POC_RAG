# 🏗️ Diagrammes UML - Architecture du Système RAG

> Schémas UML détaillés de l'architecture système RAG pour la recommandation d'événements culturels.

---

## 📋 Table des matières

1. [Diagramme UML - Classes Principales](#diagramme-uml---classes-principales)
2. [Diagramme de Composants - Flux de Données](#diagramme-de-composants---flux-de-données)

---

## Diagramme UML - Classes Principales

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODÈLE DES CLASSES SYSTÈME                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────┐
│        RAGService                    │
├──────────────────────────────────────┤
│ - llm: ChatMistralAI                 │
│ - embedding_providers: Dict          │
│ - vector_stores: Dict[str, FAISS]    │
│ - rag_chains: Dict                   │
│ - index_managers: Dict               │
├──────────────────────────────────────┤
│ + answer_question(q, provider)       │
│ + classify_intent(q)                 │
│ + rebuild_index(provider)            │
│ + load_index(provider)               │
│ + _create_rag_chain(vs)              │
│ + _invoke_with_retry(chain, data)    │
└──────────────────────────────────────┘
            ↓ uses
    ┌──────────────────────────────────────────────────┐
    │                                                   │
┌─────────────────────────────┐  ┌────────────────────────────┐
│   EmbeddingProvider         │  │  IndexManager              │
│   (Abstract)                │  ├────────────────────────────┤
├─────────────────────────────┤  │ - index_dir: str           │
│ + get_embeddings()          │  │ - index: FAISS             │
│ + get_distance_strategy()   │  │ - metadata: Dict           │
└─────────────────────────────┘  ├────────────────────────────┤
    ↑ implements                   │ + save_index(vs, meta)     │
    │                              │ + load_index()             │
    ├────────────────────────────┬─┤ + clear_index()            │
    │                            │ │ + get_index_info()         │
┌───────────────────┐  ┌─────────────────────────┐  └────────────────────────────┘
│ MistralEmbedding  │  │ HuggingFaceEmbedding    │
│ Provider          │  │ Provider                │
├───────────────────┤  ├─────────────────────────┤
│ - api_key: str    │  │ - model_name: str       │
│ - _embeddings     │  │ - _embeddings           │
├───────────────────┤  ├─────────────────────────┤
│ - (lazy init)     │  │ - (lazy init local)     │
└───────────────────┘  └─────────────────────────┘
   mistral-embed API      paraphrase-multilingual
   (1024 dims)            (384 dims, CPU-friendly)


┌──────────────────────────────────────┐
│  Classification Module               │
├──────────────────────────────────────┤
│ + classify_query_intent()            │
│   └─ uses ChatMistralAI              │
│   └─ returns: "RAG" | "CHAT"         │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│  DocumentBuilder                     │
├──────────────────────────────────────┤
│ - chunk_size: int = 1200             │
│ - chunk_overlap: int = 200           │
│ - splitter: RecursiveCharSplitter    │
├──────────────────────────────────────┤
│ + build(event: Event)                │
│   └─ returns: List[Document]         │
│ - _build_content(event): str         │
│ - _build_metadata(event): Dict       │
└──────────────────────────────────────┘
         ↑ processes
         │
    ┌────────────────┐
    │ Event (API)    │
    │ (OpenAgenda)   │
    └────────────────┘


┌──────────────────────────────────────┐
│  FastAPI Routes                      │
├──────────────────────────────────────┤
│ + POST /ask                          │
│   └─ QueryRequest → RAGService       │
│ + POST /rebuild                      │
│   └─ RebuildRequest → RAGService     │
│ + GET /health                        │
│   └─ health check response           │
└──────────────────────────────────────┘
```

### Description des Classes

#### RAGService (Orchestrateur Principal)

-   **Responsabilité** : Orchestrer tout le pipeline RAG
-   **Attributs clés** :
    -   `llm` : Instance de ChatMistralAI (lazy initialization)
    -   `embedding_providers` : Dictionnaire des providers (Mistral/HuggingFace)
    -   `vector_stores` : Cache FAISS par provider
    -   `rag_chains` : Chaînes LangChain pré-construites
    -   `index_managers` : Gestionnaires de persistance
-   **Méthodes principales** :
    -   `answer_question()` : Point d'entrée principal pour répondre aux questions
    -   `classify_intent()` : Détermine si RAG ou CHAT
    -   `rebuild_index()` : Reconstruit complètement l'index
    -   `_invoke_with_retry()` : Gère les erreurs 429 avec backoff exponentiel

#### EmbeddingProvider (Interface Abstraite)

-   **Responsabilité** : Abstraction pour les différents providers d'embeddings
-   **Implémentations** :
    -   **MistralEmbeddingProvider** : Appelle l'API mistral-embed (1024 dims)
    -   **HuggingFaceEmbeddingProvider** : Charge localement paraphrase-multilingual (384 dims)
-   **Pattern** : Lazy initialization (charge le modèle seulement à la première utilisation)

#### IndexManager

-   **Responsabilité** : Gérer la persistance des indices FAISS
-   **Fichiers gérés** :
    -   `index.faiss` : Index binaire FAISS
    -   `index.pkl` : Docstore sérialisé
    -   `metadata.json` : Métadonnées (nombre de chunks, provider, date)
-   **Méthodes** :
    -   `save_index()` : Exporte l'index sur disque
    -   `load_index()` : Récupère l'index depuis le disque
    -   `get_index_info()` : Retourne des stats (ntotal, dimensions, etc)

#### DocumentBuilder

-   **Responsabilité** : Convertir Events en LangChain Documents chunked
-   **Flux** :
    1. Nettoie HTML (BeautifulSoup)
    2. Formate les données structurées
    3. Découpe avec RecursiveCharacterTextSplitter
    4. Crée métadonnées (UID, titre, localisation, dates)

#### Classification Module

-   **Responsabilité** : Détecter l'intention (RAG vs CHAT)
-   **Utilise** : ChatMistralAI avec prompt spécifique
-   **Résultat** : "RAG" ou "CHAT"

---

## Diagramme de Composants - Flux de Données

### Flux 1 : Construction d'Index

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       FLUX DE CONSTRUCTION D'INDEX                       │
└──────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────┐
     │  OpenAgenda API │ (699 événements)
     └────────┬────────┘
              ↓
    ┌─────────────────────────────┐
    │ fetch_all_events()          │
    │ (external/openagenda_fetch) │
    └────────┬────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ Events (liste brute)        │
    │ [Event, Event, ...]         │
    └────────┬────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ DocumentBuilder.build()     │
    │ ├─ clean HTML               │
    │ ├─ format keywords          │
    │ ├─ RecursiveCharSplitter    │
    │ └─ create metadata          │
    └────────┬────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ Documents (5,500 chunks)    │
    │ [Document, Document, ...]   │
    │ + metadata + page_content   │
    └────────┬────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ EmbeddingProvider.          │
    │ get_embeddings()            │
    │ ├─ MistralEmbedding (API)   │
    │ └─ HuggingFaceEmbedding     │
    │    (local, CPU)             │
    └────────┬────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ Vector embeddings           │
    │ (1024 or 384 dims per doc)  │
    └────────┬────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ FAISS.from_documents()      │
    │ └─ distance=Cosine          │
    │ └─ index.ntotal=5500        │
    └────────┬────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ IndexManager.save_index()   │
    ├─ index.faiss (binary)       │
    ├─ index.pkl (docstore)       │
    └─ metadata.json              │
```

**Étapes détaillées** :

| #   | Étape          | Technologie           | Entrée           | Sortie                  | Temps        |
| --- | -------------- | --------------------- | ---------------- | ----------------------- | ------------ |
| 1   | Fetch Events   | requests              | OpenAgenda API   | 699 Events              | ~2s          |
| 2   | Clean & Format | BeautifulSoup + Regex | Raw Events       | Formatted Events        | ~1s          |
| 3   | Chunking       | RecursiveCharSplitter | Events           | 5,500 Documents         | ~2s          |
| 4   | Embedding      | Mistral/HuggingFace   | Documents        | Vectors (1024/384 dims) | ~30-120s     |
| 5   | Indexing       | FAISS                 | Vectors          | FAISS Index             | ~5s          |
| 6   | Persistence    | IndexManager          | Index + Metadata | Files on Disk           | ~1s          |
|     | **TOTAL**      |                       |                  |                         | **~40-130s** |

---

### Flux 2 : Réponse à une Requête

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       FLUX DE RÉPONSE À UNE REQUÊTE                      │
└──────────────────────────────────────────────────────────────────────────┘

User Query: "Quels concerts à Bayonne ?"
         ↓
    ┌─────────────────────────────┐
    │ FastAPI /ask endpoint       │
    │ QueryRequest validation     │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ RAGService.answer_question()│
    └────────┬────────────────────┘
             ├─────────────────────────────────────────┐
             ↓                                         ↓
    ┌──────────────────────────┐  ┌──────────────────────────┐
    │ classify_intent()        │  │ _invoke_with_retry()     │
    │ - prompt engineering     │  │ - retry logic (429 err)  │
    │ - ChatMistralAI          │  │ - exponential backoff    │
    └────────┬─────────────────┘  └──────────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ intent = "RAG" or "CHAT"    │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ if CHAT → friendly response │
    │ if RAG  → continue...       │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ EmbeddingProvider.          │
    │ embed_query(query)          │
    │ → vector 1024 or 384 dims   │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ FAISS.similarity_search()   │
    │ - k=6 neighbors             │
    │ - min_score=0.4             │
    │ → 6 Document chunks         │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ LangChain RAG Chain:        │
    │ - retriever: FAISS          │
    │ - llm: ChatMistralAI        │
    │ - combine: StuffDocsChain   │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ mistral-small-latest        │
    │ generates response with:    │
    │ - prompt template           │
    │ - 6 context documents       │
    │ - temperature=0.3           │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ Response text generated     │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ Extract events from context │
    │ - parse UIDs                │
    │ - collect metadata          │
    └────────┬────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │ Return JSON response        │
    │ {                           │
    │   status, question,         │
    │   answer, events,           │
    │   context, intent           │
    │ }                           │
    └─────────────────────────────┘
```

**Étapes détaillées** :

| #   | Étape                  | Technologie          | Latence    | Détails                       |
| --- | ---------------------- | -------------------- | ---------- | ----------------------------- |
| 1   | Validation             | FastAPI/Pydantic     | ~1ms       | Vérifie API key et query      |
| 2   | Intent Classification  | ChatMistralAI        | ~150ms     | Determine RAG vs CHAT         |
| 3   | Query Embedding        | Mistral/HuggingFace  | 40-50ms    | Vecteur 1024/384 dims         |
| 4   | FAISS Search           | FAISS                | ~2ms       | K=6 neighbors + threshold 0.4 |
| 5   | RAG Chain              | LangChain            | ~0.5ms     | Setup retriever + formatter   |
| 6   | LLM Generation         | mistral-small-latest | ~300ms     | Génère réponse avec contexte  |
| 7   | Event Extraction       | Python regex         | ~5ms       | Parse UIDs et métadonnées     |
| 8   | Response Serialization | Pydantic             | ~1ms       | Convert to JSON               |
|     | **TOTAL (RAG)**        |                      | **~500ms** | Comprise tout sauf API delays |
|     | **TOTAL (CHAT)**       |                      | **~150ms** | Classification + réponse      |

---

### Cas d'Usage : CHAT vs RAG

#### Cas 1 : Question CHAT

```
User: "Bonjour, comment allez-vous ?"
          ↓
     classify_intent()
          ↓
    Result: "CHAT"
          ↓
   get_chat_response()
          ↓
Response: "Bonjour ! Je vais bien, merci de demander. Comment puis-je vous aider ?"
Latency: ~150ms
```

#### Cas 2 : Question RAG

```
User: "Quels festivals de musique y a-t-il en janvier ?"
          ↓
     classify_intent()
          ↓
    Result: "RAG"
          ↓
   FULL RAG PIPELINE
   └─ Embed query (1024-dim)
   └─ Search FAISS (K=6)
   └─ Format + contexte
   └─ Generate with mistral-small-latest
          ↓
Response: "Voici les festivals de musique disponibles en janvier..."
Events: [Festival A, Festival B, ...]
Latency: ~500ms
```

---

## 🔗 Intégration avec le Projet

### Fichiers Concernés

```
app/
├── services/rag_service.py      ← RAGService (orchestrateur)
├── core/
│   ├── embeddings.py            ← EmbeddingProvider + implémentations
│   ├── index_manager.py         ← IndexManager
│   ├── classification.py        ← Classification Module
│   └── prompts.py               ← Prompts pour LLM
├── utils/document_converter.py  ← DocumentBuilder
├── external/
│   └── openagenda_fetch.py      ← fetch_all_events()
└── routes/
    ├── query.py                 ← FastAPI /ask endpoint
    ├── rebuild.py               ← FastAPI /rebuild endpoint
    └── health.py                ← FastAPI /health endpoint
```

### Points d'Entrée

1. **Construction d'Index** : `POST /rebuild`

    - Appelle `RAGService.rebuild_index(provider)`
    - Déclenche le flux complet de construction

2. **Réponse aux Questions** : `POST /ask`

    - Appelle `RAGService.answer_question(question, provider)`
    - Gère CHAT et RAG automatiquement

3. **Santé du Système** : `GET /health`
    - Vérification rapide de disponibilité

---

## 📚 Références

Pour plus de détails sur chaque composant, voir :

-   **[RAPPORT_TECHNIQUE.md](RAPPORT_TECHNIQUE.md)** - Documentation technique complète
-   **Code source** - Docstrings dans `app/`
