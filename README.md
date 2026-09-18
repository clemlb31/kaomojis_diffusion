# kaomojis_diffusion

Générateur de kaomojis à partir de langage naturel (français ou anglais) : du simple `(=^･ω･^=)`
jusqu'au dessin multi-ligne, sans aucun filtrage de contenu.

## Pourquoi il ne refuse rien

Les refus d'un LLM viennent de son entraînement *chat/instruct*, pas de son architecture. Ici le
modèle est un modèle **base** (non-chat) affiné par LoRA sur un format fixe
`### Prompt: … / ### Kaomoji: …` : il n'apprend qu'à continuer par un dessin et n'a aucun
comportement de refus à déclencher. Même un modèle « uncensored » de type chat garde des refus
résiduels ; un modèle base affiné ainsi, non.

## Installation

```bash
python3 -m venv --system-site-packages .venv   # réutilise un torch déjà installé
.venv/bin/pip install -r requirements.txt
```

## Pipeline

| Étape | Commande | Sortie |
|---|---|---|
| 1. Scraper | `python -m kaogen.scrape.kaomoji_ru`<br>`python -m kaogen.scrape.japaneseemoticons`<br>`python -m kaogen.scrape.emojicombos --max-fetches 400` | `data/raw/*.jsonl` |
| 2. Nettoyer | `python -m kaogen.build_dataset` | `data/kaomojis.jsonl` |
| 3. Légender *(optionnel, payant)* | `python -m kaogen.caption submit` puis `fetch` | `data/captions.jsonl` |
| 4. Paires | `python -m kaogen.build_pairs` | `data/train.jsonl`, `data/val.jsonl` |
| 5. Entraîner | `python -m kaogen.train_lm --max-steps 3000` | `runs/lm/` |
| 6. Générer | `python -m kaogen.generate "un chat timide" -n 4` | — |

Les pages téléchargées sont mises en cache dans `data/raw/html/` : relancer un scraper ne
retouche jamais le réseau pour une page déjà vue, donc un bug de parseur se corrige en rejouant
le cache : `python -m kaogen.scrape.emojicombos --max-fetches 0` re-parse tout hors-ligne (~40 s).
`--seeds tag1,tag2` place des tags en tête de file pour un crawl ciblé.

> emojicombos masque les mots-clés des items « sensibles » en base64 (`data-encoded-text`).
> Un parseur qui ne lit que les liens visibles perd tous les tags NSFW **et rend le filtre de
> contenu inopérant sur ces items** : `parse_keywords` décode les deux formes.

### Entraînement (étape 5)

Mesuré sur un MacBook M3 16 Go (PyTorch MPS, Qwen3-0.6B-Base, LoRA r=16, bfloat16) :
350 tokens/s au démarrage puis ~240 en régime soutenu, 1,7–2,2 Go de mémoire GPU, 4,37 M tokens
par époque, soit **3 h 30 à 5 h par époque**. `--max-steps 3000` couvre environ une époque.
Repère : après 200 pas, perte 4,86 → 3,10 (validation 3,86) ; les sorties sont des kaomojis bien
formés mais pas encore fidèles au prompt.

- `--batch-tokens` borne la mémoire par micro-lot (en tokens, pas en exemples : un dessin
  multi-ligne coûte ~10× un kaomoji). `--grad-accum` règle le lot effectif sans coût mémoire.
  Laissés à 0, les deux prennent une valeur adaptée au GPU détecté.
- `--mps-memory-fraction 0.6` fait échouer l'entraînement (OOM) au lieu de laisser macOS swapper
  pendant des heures. Si l'OOM survient : baisser `--batch-tokens`, ou `--grad-checkpoint`
  (activations 2,3 Go → 0,3 Go, mais 25 à 45 % plus lent).
- Un adaptateur est sauvegardé à chaque évaluation (`--eval-every`), avec des échantillons.

#### Sur GPU NVIDIA

Le code détecte CUDA tout seul (`pick_device`) ; rien à changer. Deux détails sont déjà gérés :
le contournement MPS (une copie en float32 pour l'indexation booléenne en bfloat16, absente de
MPS) ne s'applique pas sur CUDA, et `--batch-tokens` passe de 512 à 8192 avec `--grad-accum 1`,
car un GPU discret est affamé par des micro-lots taillés pour un Mac.

Ordres de grandeur : le M3 mesuré tient ~235 tokens/s dans 1,8 Go. Une RTX 4070 a 12 Go et une
bande passante mémoire environ 10× supérieure, donc les 4 000 pas devraient passer de ~6,7 h à
quelques dizaines de minutes. Les 12 Go permettent aussi de viser un backbone plus gros
(`--model Qwen/Qwen3-1.7B-Base`, toujours en LoRA), ce que les 16 Go partagés du Mac
interdisaient — c'est probablement le gain le plus utile, devant la vitesse.

### Légendage (étape 3)

Nécessite `ANTHROPIC_API_KEY`. Passe par l'API Batches (−50 %), 8 kaomojis par requête, modèle
`claude-haiku-4-5` par défaut (`--model` pour changer). Essai bon marché : `submit --limit 200`.
Les items NSFW ne sont jamais envoyés à l'API : leurs prompts sont générés hors-ligne à partir
de leurs tags (gabarits FR/EN + glossaire `kaogen/tags_fr.json`). Sans l'étape 3, tout le
dataset utilise ces gabarits : ça marche, mais la compréhension du langage naturel est moindre.

## Styles

`build_dataset` classe chaque entrée :

- `kaomoji` — une ligne ;
- `multiline` — art texte sur plusieurs lignes ;
- `braille` — *dot art* en caractères braille (⣿). Conservé dans le dataset mais **exclu de
  l'entraînement LLM par défaut** : ~1,44 token par caractère, soit ~700 tokens pour un dessin
  médian. Un caractère braille est une cellule de 2×4 pixels : ces dessins sont des images
  binaires, et relèvent d'un modèle de diffusion d'image plutôt que d'un LLM ;
- `emoji` — suites d'emojis, écartées.

Sont aussi écartés : le texte zalgo, les dessins hors gabarit (24×64 par défaut), et tout
contenu sexualisant des mineurs (`BLOCKED_TAGS` / `MINOR_TAGS` dans `kaogen/scrape/emojicombos.py`).

## Sources

kaomoji.ru, japaneseemoticons.me, emojicombos.com. Les données scrapées ne sont pas versionnées
(`data/` est dans `.gitignore`).
