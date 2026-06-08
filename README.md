# BiRefNet Rust

Serveur Rust inspire du prototype FastAPI du repertoire parent.

Fonctionnalites:

- upload d'une ou plusieurs images dans une seule requete;
- choix d'un fond `transparent`, `white`, `black` ou `image`;
- interface avec apercu avant/apres;
- API multi-images avec ZIP si plusieurs images sont envoyees;
- telechargement direct du PNG si une seule image est envoyee;
- choix du modele dans l'interface et l'API;
- traitement parallele quand plusieurs images sont envoyees;
- sortie PNG lossless aux dimensions de l'image d'origine;
- option ZIP dans l'interface;
- suppression individuelle ou globale des images selectionnees ou deja traitees;
- les metadonnees EXIF/GPS d'origine ne sont pas recopiees dans les sorties PNG;
- les dates des entrees ZIP sont neutralisees a `1980-01-01 00:00:00`;
- l'interface verifie le transport avant upload: HTTPS est accepte, localhost est accepte, HTTP distant est bloque;
- aucune image utilisateur n'est ecrite sur disque par l'application;
- les buffers d'entree et les buffers intermediaires sont remis a zero explicitement quand ils sortent du pipeline.

Le backend BiRefNet utilise `tch-rs`, donc libtorch. Le runtime Docker n'embarque pas Python: il charge uniquement des modeles BiRefNet deja exportes en TorchScript (`.ts`).

## Modeles

Les fichiers `.ts` ne doivent pas etre ajoutes a Git: ils sont lourds et changent rarement. Trois modes sont supportes:

- monter un volume Docker contenant les modeles dans `/app/models`;
- fournir des URLs via `BIREFNET_MODEL_URLS`, le conteneur les telecharge au demarrage si elles ne sont pas deja presentes;
- definir directement `BIREFNET_MODELS` avec des chemins locaux.

Format de `BIREFNET_MODEL_URLS`:

```text
id|label|url[|filename];id2|label2|url2[|filename2]
```

Exemple:

```bash
BIREFNET_MODEL_URLS='birefnet-lite|BiRefNet Lite|https://example.com/models/birefnet-lite.ts|birefnet-lite.ts'
```

Format de `BIREFNET_MODELS`:

```text
id|label|path;id2|label2|path2
```

## Lancer

Le backend `tch-rs` est active par defaut. Docker est le chemin recommande. En local, `run.sh` utilise `LIBTORCH` si la variable est definie; sinon il compile avec la feature `download-libtorch`, comme le Dockerfile.

Pour lancer en local avec les modeles presents dans `models/`:

```bash
./run.sh
```

Equivalent manuel:

```bash
LIBTORCH_BYPASS_VERSION_CHECK=1 \
BIREFNET_MODELS='birefnet-base|BiRefNet Base|models/birefnet.ts' \
cargo run --release --features download-libtorch
```

`LIBTORCH_BYPASS_VERSION_CHECK=1` evite un blocage strict de version si vous fournissez votre propre libtorch.

Pour compiler et tester:

```bash
./check.sh
```

Le traitement multi-images utilise un pool de threads. Pour limiter la concurrence, par exemple sur une machine avec peu de RAM:

```bash
RAYON_NUM_THREADS=2 ./run.sh
```

## Qualite d'image

L'inference redimensionne seulement l'image d'entree du modele en `1024x1024` pour calculer le masque. Le rendu final reutilise les pixels originaux et conserve les dimensions d'origine. La sortie est un PNG lossless, necessaire pour conserver la transparence.

Si un fond image est applique, il est redimensionne en mode `contain`: le fond complet reste visible et n'est pas coupe.

Les sorties sont encodees depuis les pixels RGBA en memoire avec un encodeur PNG neuf. Les metadonnees de l'image source, comme EXIF/GPS, profil appareil, commentaires ou date de creation, ne sont pas recopiees. Les archives ZIP generent une date neutre fixe (`1980-01-01 00:00:00`) au lieu de reprendre la date source ou la date de traitement.

## Confidentialite du transport

L'interface affiche l'etat du transport avant upload. Elle autorise `https://` et `http://localhost` / `http://127.0.0.1`, puis bloque un upload depuis une origine HTTP distante, car dans ce cas les images transiteraient sans chiffrement TLS. En production, placez le service derriere un reverse proxy HTTPS.

## Docker

Construire l'image depuis ce sous-repertoire:

```bash
docker build -t birefnet-rust .
```

Le build Docker utilise `tch/download-libtorch`, donc il telecharge libtorch C++ pendant la compilation. Python, PyTorch pip et torchvision ne sont pas installes dans l'image finale.

Alpine n'est pas utilise: les binaires libtorch officiels ciblent glibc, alors qu'Alpine utilise musl. `debian:bookworm-slim` est le compromis le plus simple et fiable.

Lancer avec un volume local de modeles:

```bash
docker run --rm \
  -p 3000:3000 \
  -v "$PWD/models:/app/models:ro" \
  birefnet-rust
```

Lancer en telechargeant les modeles au demarrage:

```bash
docker run --rm \
  -p 3000:3000 \
  -v birefnet-models:/app/models \
  -e 'BIREFNET_MODEL_URLS=birefnet-lite|BiRefNet Lite|https://example.com/models/birefnet-lite.ts|birefnet-lite.ts' \
  birefnet-rust
```

Pour un serveur distant, exposez le conteneur derriere un reverse proxy HTTPS. L'UI bloque les uploads depuis une origine HTTP distante.

```text
http://127.0.0.1:3000/ui
```

Si vous montez des fichiers dans `/app/models`, les noms standards sont detectes automatiquement:

```text
models/birefnet-base.ts
models/birefnet-lite.ts
models/birefnet-hr.ts
```

Sinon, declarez explicitement les chemins:

```bash
docker run --rm \
  -p 3000:3000 \
  -v "$PWD/private-models:/models:ro" \
  -e 'BIREFNET_MODELS=birefnet-lite|BiRefNet Lite|/models/lite.ts;birefnet-base|BiRefNet Base|/models/base.ts' \
  birefnet-rust
```

`BIREFNET_TORCHSCRIPT_PATH=models/birefnet.ts` reste supporte pour un seul modele, mais `BIREFNET_MODELS` est le format conseille. Sans modele BiRefNet configure, le serveur refuse de demarrer.

Par defaut, le serveur ecoute sur:

```text
http://127.0.0.1:3000
```

Pour choisir une autre adresse:

```bash
BIND_ADDR=127.0.0.1:8080 cargo run
```

## Interface Web

Ouvrir:

```text
http://127.0.0.1:3000/ui
```

## API

Endpoint:

```text
POST /birefnet/remove-background
```

Champs multipart:

- `images`, `files` ou `file`: une ou plusieurs images;
- `model`: identifiant expose par `/models`, par exemple `birefnet-base`, `birefnet-lite` ou `birefnet-hr`;
- `bg_mode`: `transparent`, `white`, `black` ou `image`;
- `background_image`: image de fond, requise si `bg_mode=image`.

Exemple:

```bash
curl \
  -F 'images=@photo-1.jpg' \
  -F 'images=@photo-2.jpg' \
  -F 'model=birefnet-base' \
  -F 'bg_mode=image' \
  -F 'background_image=@background.jpg' \
  http://127.0.0.1:3000/birefnet/remove-background \
  --output results.zip
```

## Confidentialite et memoire

L'application ne persiste pas les uploads, les resultats ou l'image de fond. Les donnees restent en memoire pendant la requete. Les buffers manipulables par l'application sont zeroises apres traitement; le buffer final reste necessaire jusqu'a l'envoi HTTP du telechargement, puis il est libere par le serveur.
