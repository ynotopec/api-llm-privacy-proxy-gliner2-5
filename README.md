# OpenAI Privacy Filter Proxy GLiNER2.5

Proxy OpenAI-compatible `/v1/*` qui filtre les PII avec `fastino/gliner2.5-multi-v1` avant transmission vers un backend LLM.

## Fonctionnement

Client OpenAI-compatible
→ `api-llm-privacy-proxy-gliner2.5`
→ redaction PII : `[EMAIL_1]`, `[PERSON_1]`, etc.
→ upstream OpenAI-compatible

## Installation

```bash
./install.sh
cp .env.example .env
nano .env
source run.sh 0.0.0.0 8088
```

`./install.sh` installe également NumPy, PyTorch, Transformers, Accelerate et
PEFT, requis par le moteur d'inférence GLiNER2.5, puis importe explicitement la
classe `AutoExtractor` pour vérifier toutes ses dépendances chargées paresseusement.
Pour utiliser un build CUDA précis plutôt que le paquet PyPI par défaut,
installer la version PyTorch adaptée au pilote depuis l'index officiel PyTorch
dans le même venv.

Le checkpoint `fastino/gliner2.5-multi-v1` utilise l'architecture `extractor` de
GLiNER2.5. Il est donc chargé avec `gliner2.AutoExtractor`, et non avec l'ancienne
classe de modèle span `gliner2.GLiNER2`. Le périphérique résolu est transmis via
`map_location` pendant le chargement : le modèle boundary n'est pas déplacé vers
CUDA après son initialisation.
L'attention utilise `eager` par défaut, car l'encodeur DeBERTaV2 de ce checkpoint
ne prend pas encore en charge SDPA. La variable `GLINER_ATTENTION_IMPLEMENTATION`
permet de modifier ce choix pour un autre encodeur.

## Variables importantes

```bash
INBOUND_API_KEYS='change-me'
UPSTREAM_BASE_URL='http://127.0.0.1:8000/v1'
UPSTREAM_API_KEY=''
LLM_ENABLED=true  # false => retourne uniquement le payload anonymisé, sans appeler l'upstream LLM
PRIVACY_MODEL_ID='fastino/gliner2.5-multi-v1'
PRIVACY_ENTITY_TYPES='person,full_name,first_name,last_name,date_of_birth,email,phone_number,address,street_address,city,state_or_region,postal_code,country,government_id,national_id_number,passport_number,drivers_license_number,tax_id,bank_account,account_number,iban,payment_card,card_number,username,ip_address,password,api_key,access_token,secret'
DEVICE=auto  # auto => cuda si torch.cuda.is_available(), sinon cpu
TORCH_DTYPE=auto
GLINER_ATTENTION_IMPLEMENTATION=eager  # valeur optimale pour le checkpoint DeBERTaV2 par défaut
FILTER_OUTPUT=true
MODEL_SUFFIX='-anonym'
MODEL_IDLE_UNLOAD_SECONDS=300  # <= 0 désactive le déchargement automatique
MODEL_IDLE_CHECK_SECONDS=30     # fréquence de vérification en tâche de fond
MAX_REQUEST_BYTES=10485760      # refuse les corps trop volumineux avant inférence
MAX_JSON_DEPTH=64               # protège la traversée récursive
MAX_JSON_NODES=100000           # borne le coût d'un payload JSON
MAX_CONCURRENT_INFERENCES=1     # protège RAM/VRAM; augmenter après mesure de charge
```

## Test

```bash
pytest -q
```

## Déploiement Kubernetes avec Helm (optionnel)

Le chart de production se trouve dans `deploy/helm/privacy-proxy`. Il active par
défaut un cache modèle persistant, des ressources CPU/mémoire, des probes et un
contexte de sécurité non-root avec un système de fichiers en lecture seule.

Construire et publier d'abord l'image, puis installer le chart en utilisant un
Secret Kubernetes existant (méthode recommandée afin de ne pas versionner les
clés dans un fichier de valeurs) :

```bash
kubectl create namespace privacy-proxy
kubectl -n privacy-proxy create secret generic privacy-proxy-credentials \
  --from-literal=INBOUND_API_KEYS='change-me' \
  --from-literal=UPSTREAM_API_KEY=''

helm upgrade --install privacy-proxy ./deploy/helm/privacy-proxy \
  --namespace privacy-proxy \
  --set image.repository=registry.example.com/privacy-proxy \
  --set image.tag=1.0.1 \
  --set existingSecret=privacy-proxy-credentials \
  --set config.UPSTREAM_BASE_URL=http://llm.default.svc.cluster.local:8000/v1
```

Pour un nœud NVIDIA, utiliser une image PyTorch compatible CUDA et ajouter
`resources.limits.nvidia.com/gpu: "1"` dans un fichier de valeurs. Ajuster aussi
la classe et la taille du PVC, les ressources et les contraintes de placement au
cluster. Un exemple complet des options est disponible dans
`deploy/helm/privacy-proxy/values.yaml`.

## Appel OpenAI-compatible

```bash
curl -s http://127.0.0.1:8088/v1/chat/completions \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ai-vision-anonym",
    "messages": [
      {
        "role": "user",
        "content": "ré-écrire au propre mon login antonio et mon mot de passe toto"
      }
    ]
  }' | jq .
```

## Vérifier GPU / latence

```bash
curl -s http://127.0.0.1:8088/health | jq .
```

Champs utiles :

* `llm_enabled` : indique si le proxy appelle l’upstream LLM (`true`) ou retourne seulement le payload anonymisé (`false`).
* `device` : valeur demandée par la variable d'environnement `DEVICE`.
* `resolved_device` : périphérique réellement utilisé par le modèle chargé (`cuda`, `cpu`, `unloaded` ou `unknown`).
* `cuda_available` : résultat de `torch.cuda.is_available()` lorsque `DEVICE=auto`.
* `model_loaded` : indique si le modèle GLiNER2 est déjà chargé en mémoire.
* `model_idle_unload_seconds` / `model_idle_check_seconds` : délai d'inactivité et fréquence de la tâche de fond qui décharge le modèle.

Si `resolved_device=cpu` avec `DEVICE=auto`, le conteneur/process ne voit pas CUDA. Vérifier l'image PyTorch CUDA, le runtime NVIDIA (`--gpus all`) et les drivers hôte. Pour isoler la latence GLiNER2 de la latence upstream, lire aussi les en-têtes `x-privacy-filter-latency-ms`, `x-privacy-filtered-spans` et `x-privacy-filtered-output-spans`.

Le proxy reprend le comportement mémoire du projet de référence `ynotopec/api-llm-privacy-proxy` (`MODEL_IDLE_UNLOAD_SECONDS`), mais ajoute une tâche de fond : le modèle est déchargé même si aucune nouvelle requête ne vient déclencher le contrôle d'inactivité. Le déchargement supprime la référence au modèle, lance `gc.collect()` puis vide le cache CUDA quand PyTorch voit un GPU.

Pour réduire la latence, désactiver le filtrage de sortie si non nécessaire :

```bash
FILTER_OUTPUT=false
```

## Metrics

```bash
curl -s http://127.0.0.1:8088/metrics \
  -H 'Authorization: Bearer change-me'
```

## Notes production

* Par défaut, le proxy filtre les entrées envoyées au LLM et les réponses du LLM (`FILTER_OUTPUT=true`).
* Les chaînes trop longues pour être filtrées et les payloads hors limites sont refusés avec HTTP 413 plutôt que transmis sans anonymisation.
* Le client HTTP upstream est mutualisé (pool de connexions). L'inférence synchrone est déportée hors de la boucle événementielle et sa concurrence est bornée par `MAX_CONCURRENT_INFERENCES`.
* En production, définir `INBOUND_API_KEYS`; sans cette variable, les routes `/v1` sont volontairement publiques.
* Les jetons définis dans `INBOUND_API_KEYS` ne sont jamais transmis à l'upstream; seul `UPSTREAM_API_KEY`, lorsqu'il est configuré, alimente son en-tête `Authorization`.
* `LLM_ENABLED=false` rend le LLM optionnel : les requêtes POST `/v1/chat/completions` gardent le format OpenAI-compatible (`choices[0].message.content`) avec le contenu anonymisé, sans appeler `UPSTREAM_BASE_URL`. Les autres endpoints POST retournent le payload anonymisé et les statistiques de filtrage.
* Les modèles exposés au client sont suffixés avec `-anonym` (`MODEL_SUFFIX`) et seul le champ `model` OpenAI de premier niveau est désuffixé avant envoi à l’upstream.
* Les configurations utilisateur comme `thinking` / `reasoning` sont préservées telles quelles par défaut.
* `FILTER_OUTPUT=false` permet de désactiver le filtrage des réponses si la latence est prioritaire.
* Le streaming SSE est refusé avec HTTP 400 lorsque `FILTER_OUTPUT=true` : une
  entité peut traverser plusieurs chunks, donc relayer ces chunks sans filtre
  contournerait la garantie de confidentialité. Définir explicitement
  `FILTER_OUTPUT=false` pour autoriser `stream=true`.
* Le modèle peut rater des PII, surtout hors anglais ou avec formats métier spécifiques.
* Pour contexte gouvernement / médical / RH / finance, valider sur corpus interne et ajouter éventuellement règles regex métier ou fine-tuning.

## Service systemd exemple

```bash
sudo tee /etc/systemd/system/api-llm-privacy-proxy-gliner2.service >/dev/null <<'SERVICE_EOF'
[Unit]
Description=OpenAI Privacy Filter Proxy GLiNER2
After=network-online.target
Wants=network-online.target

[Service]
User=ailab
WorkingDirectory=/home/ailab/api-llm-privacy-proxy-gliner2
Environment=VENV_DIR=/home/ailab/venv/api-llm-privacy-proxy-gliner2
ExecStart=/bin/bash -lc 'source ./run.sh 0.0.0.0 8088'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE_EOF

sudo systemctl daemon-reload
sudo systemctl enable --now api-llm-privacy-proxy-gliner2
sudo journalctl -u api-llm-privacy-proxy-gliner2 -f
```

## Dépannage

Si les logs contiennent encore `GLiNER2.extract_entities() missing 1 required positional argument: 'entity_types'`, le service lancé n'utilise pas ce code. Vérifier `/health` : le champ `revision` doit valoir `gliner2.5`, puis relancer `./install.sh` et redémarrer le service systemd.
