#!/bin/sh
cd "/home/xi/src/Strata"
exec "/home/xi/src/Strata/.venv/bin/python" "/home/xi/src/Strata/serve/server.py" "--engine" "strata" "--config" "/home/xi/src/Strata/strata-swift-iq3_xxs.json" "--host" "0.0.0.0" "--port" "8005" "--fit-max-tokens" "--open"
