# 🎭 OC_P7_POC_RAG - Système de Récupération Augmentée par Génération (RAG)

> **Système intelligent de découverte d'événements** en Pyrénées-Atlantiques utilisant la Récupération Augmentée par Génération (RAG) avec embeddings multi-providers et LLM Mistral.

[![Test Status](https://img.shields.io/badge/tests-135%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-86%25-green)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## 📖 Documentation Technique

Pour une compréhension complète du système, architecture et implémentation, consultez :

| Document                                         | Contenu                                                           |
| ------------------------------------------------ | ----------------------------------------------------------------- |
| **[RAPPORT_TECHNIQUE.md](RAPPORT_TECHNIQUE.md)** | Guide technique exhaustif (450+ lignes) couvrant tous les aspects |
| **[ARCHITECTURE_UML.md](ARCHITECTURE_UML.md)**   | Diagrammes UML détaillés des classes et flux de données           |

---

## 🎯 Vue d'ensemble

Ce projet implémente un **système RAG production-ready** capable de :

✅ **Récupérer des événements** depuis l'API OpenAgenda
✅ **Générer des embeddings** avec Mistral ou HuggingFace
✅ **Indexer vectoriellement** via FAISS
✅ **Classifier les intentions** utilisateur (RAG vs CHAT)
✅ **Générer des réponses** contextuelles avec LLM Mistral
✅ **Fournir une API** FastAPI sécurisée et documentée

---

## 🚀 Installation

### Prérequis

-   Python 3.12+
-   [uv](https://github.com/astral-sh/uv) (package manager ultrarapide)
-   API Keys : Mistral AI

### Setup

```bash
# Clone du projet
git clone <repo-url>
cd OC_P7_POC_RAG

# Créer l'environnement virtuel avec uv
uv venv

# Activer l'environnement
source .venv/bin/activate

# Installer les dépendances (lit pyproject.toml)
uv sync
```

### Configuration (.env)

```bash
# Copier le template
cp .env.example .env

# Éditer .env avec vos clés API
# Fichier .env requis:
MISTRAL_API_KEY=your_mistral_key
API_KEY=your_api_key_for_access
LOCATION_DEPARTMENT=Pyrénées-Atlantiques  # Configurable
FIRST_DATE=2025-01-01T00:00:00            # Configurable
DEFAULT_EMBEDDING_PROVIDER=mistral        # ou huggingface
```

---

## 🎮 Utilisation

### Démarrer le serveur

```bash
# Mode développement
make dev

# Ou directement avec uvicorn
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Premiers tests

```bash
# 1. Rebuilder l'index FAISS
curl -X POST "http://localhost:8000/rebuild?provider=mistral" \
  -H "X-API-Key: your_api_key"

# 2. Vérifier la santé du système
curl "http://localhost:8000/health"

# 3. Poser une question
curl -X POST "http://localhost:8000/ask?embedding_provider=mistral" \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts y a-t-il?"}'
```

### Endpoints principaux

| Endpoint   | Méthode | Description                      |
| ---------- | ------- | -------------------------------- |
| `/ask`     | POST    | Poser une question (RAG ou CHAT) |
| `/rebuild` | POST    | Reconstruire l'index FAISS       |
| `/health`  | GET     | Vérifier la santé du système     |

Voir **[RAPPORT_TECHNIQUE.md - Section 6](RAPPORT_TECHNIQUE.md#6-api-et-endpoints-exposés)** pour la documentation complète des endpoints.

---

## 📦 Stack Technologique

FastAPI 0.121.3 • LangChain 1.0.8 • Mistral AI • FAISS 1.13.0 • Pytest 9.0.1 • Ragas 0.3.9 • uv

Voir **[RAPPORT_TECHNIQUE.md - Section 2](RAPPORT_TECHNIQUE.md#technologies-utilisées)** pour détails.

---

## 🏗️ Architecture

```
User Query
    ↓
[Validation & Authentication] ← API Key
    ↓
[Classification d'Intent] ← Mistral LLM
    ├→ CHAT → Réponse générique
    └→ RAG → Recherche vectorielle
        ↓
    [Embedding Question] ← Mistral ou HuggingFace
        ↓
    [FAISS Search K=6] ← Index vectoriel
        ↓
    [LLM Generation] ← Prompt + Contexte
        ↓
    [Extract Events] ← Parse réponse
        ↓
JSON Response
```

📊 **Diagrammes UML détaillés** : Voir **[ARCHITECTURE_UML.md](ARCHITECTURE_UML.md)** pour :

-   Diagramme complet des classes (RAGService, EmbeddingProvider, IndexManager, etc.)
-   Flux de construction d'index (6 étapes)
-   Flux de réponse à une requête (9 étapes)
-   Cas d'usage CHAT vs RAG

---

## 🧪 Tests

```bash
make test              # Exécuter tous les tests
make coverage          # Rapport couverture (86%, 135 tests)
open htmlcov/index.html # Voir rapport HTML
```

---

## 📊 Données & Indexation

```bash
# Rebuilder l'index FAISS
curl -X POST "http://localhost:8000/rebuild?provider=mistral" \
  -H "X-API-Key: $API_KEY"

# Ou HuggingFace
curl -X POST "http://localhost:8000/rebuild?provider=huggingface" \
  -H "X-API-Key: $API_KEY"
```

Voir **[RAPPORT_TECHNIQUE.md - Section 5](RAPPORT_TECHNIQUE.md#5-construction-de-la-base-vectorielle)** pour détails (source, chunks, persistance).

---

## 🐳 Déploiement Docker

```bash
# Build image
docker build -t rag-system .

# Run container
docker run -p 8000:8000 \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  -e API_KEY=$API_KEY \
  rag-system

# Ou avec Docker Compose
docker compose up -d
```

---

## 🛠️ Commandes Utiles

```bash
# Gestion des dépendances
uv add package_name          # Ajouter une dépendance
uv sync                      # Installer depuis pyproject.toml
uv sync --upgrade            # Mettre à jour

# Développement
make dev                     # Démarrer dev server
make test                    # Exécuter tests
make coverage                # Rapport couverture
make lint                    # Linting (flake8, isort, black)
make format                  # Formatter le code

# Évaluation RAG
python scripts/ragas_eval.py --provider mistral --num_questions 10
```

---

## 🔐 Sécurité

✅ Authentification API Key • Validation Pydantic • Retry logic • Timeouts • Logging sûr

---

## 📁 Structure du Projet

```
app/
├── core/                    # Logique métier
│   ├── classification.py    # Détection d'intent (RAG/CHAT)
│   ├── embeddings.py        # Multi-provider embeddings
│   ├── index_manager.py     # Persistance FAISS
│   ├── prompts.py           # Prompts LLM
│   └── config.py            # Configuration
├── services/
│   └── rag_service.py       # Orchestration RAG
├── routes/
│   ├── query.py             # Endpoint /ask
│   ├── rebuild.py           # Endpoint /rebuild
│   └── health.py            # Endpoint /health
├── external/
│   └── openagenda_fetch.py  # Client OpenAgenda API
├── utils/
│   └── document_converter.py # Chunking documents
└── main.py                  # Entrée FastAPI

tests/                       # 135 tests, 86% coverage
scripts/
└── ragas_eval.py           # Évaluation RAG (Ragas)
data/
└── faiss_index_<provider>/ # Indices vectoriels
```

---

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amazing`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Pousser vers la branche (`git push origin feature/amazing`)
5. Ouvrir une Pull Request

**Standards requis** :

-   ✅ 86%+ code coverage
-   ✅ Type hints obligatoires
-   ✅ Pydantic models pour I/O
-   ✅ Docstrings complètes
-   ✅ Tests pour chaque fonction

---

## 🚨 Troubleshooting

### Problème : Index non trouvé

```bash
curl -X POST "http://localhost:8000/rebuild?provider=mistral" \
  -H "X-API-Key: $API_KEY"
```

### Problème : Réponses lentes

Utiliser HuggingFace (plus rapide) :

```bash
curl -X POST "http://localhost:8000/ask?embedding_provider=huggingface" ...
```

### Problème : Installation uv échoue

```bash
# Installer uv
pip install uv

# Ou brew (macOS)
brew install uv
```

Pour plus de détails, voir **[RAPPORT_TECHNIQUE.md](RAPPORT_TECHNIQUE.md)**.

---

## 📚 Documentation Complète

Pour une documentation exhaustive, consultez :

| Document                                         | Contenu                                     |
| ------------------------------------------------ | ------------------------------------------- |
| **[RAPPORT_TECHNIQUE.md](RAPPORT_TECHNIQUE.md)** | ⭐ Analyse technique complète (10 sections) |
| **Code + Comments**                              | Documentation inline dans `app/`            |

---

## 📊 Résultats & Évaluation

**Ragas Scores** : Faithfulness 0.87 • Answer Relevancy 0.84 • Context Recall 0.92 • **Moyenne: 0.85** ✅

Voir **[RAPPORT_TECHNIQUE.md - Section 7](RAPPORT_TECHNIQUE.md#7-évaluation-du-système)** pour analyse détaillée.

---

## ⭐ Stats

-   **Tests** : 135 passing ✅
-   **Coverage** : 86% ✅
-   **Évaluation RAG** : 0.85 score ✅
-   **Status** : Production Ready ✅

---

## 📄 Licence

MIT License

---

## 👥 Auteur

**Xavier Coulon** - OpenClassrooms Projet 7

---

## 🙏 Remerciements

-   **OpenAgenda** - API d'événements
-   **Mistral AI** - LLM & Embeddings
-   **LangChain** - Orchestration RAG
-   **FAISS** - Indexation vectorielle

---

**Dernière mise à jour** : 29 Novembre 2025
**Version** : 1.0.0
**Status** : ✅ Production Ready
