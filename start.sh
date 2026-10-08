#!/usr/bin/env bash
# Guided entry point for local development.
#
# Two run modes, because LLAMA_SERVER_URL must differ between them (host.docker.internal only
# resolves inside a container):
#   ./start.sh            -> docker compose up --build (default; matches
#                             the project's actual deployment target)
#   ./start.sh --native   -> run app/main.py directly via a local venv,
#                             for fast iteration without a rebuild each time
#
# Plus one configuration command, so every variable the app needs can be
# read and changed without opening .env in an editor:
#   ./start.sh --config            -> every variable of .env.example with its .env value
#                                      (secrets masked)
#   ./start.sh --config KEY=VALUE  -> add or update one variable in .env
#   ./start.sh --config --edit     -> one question per variable, then one write; asked by
#                                      --native too when there is no .env here
#
# Plus the command line of the Admin API: one command per API route, generated from
# the routes themselves, the same handlers the admin UI uses. It talks to the running
# application, or calls the handlers in process when it is stopped:
#   ./start.sh --admin                      -> menus, one per API tag
#   ./start.sh --admin describe [--json]    -> every command, with its method, path and scope
#   ./start.sh --admin COMMAND [--flag value ...] [--json]
#   ./start.sh --chat [--agent NAME]        -> talk to an agent through the API
#   (models, --admin list-models | pull-model --spec X | import-model --path X |
#    delete-model --name X)
# The menus and the one-shot command are one command (--api was removed).
#
# And the guided restore of a database backup, with the application
# stopped: lists the backups, checks the chosen one, keeps the current database
# as a "before-restore" copy, then swaps it in:
#   ./start.sh --restore [FILE] [--list] [--yes] [--allow-unreadable]
#
# And what runs, and how to stop it:
#   ./start.sh --status          -> app (container or native), Admin API,
#                                    llama-server: up or down
#   ./start.sh --stop [--all]    -> stop the native app or the container;
#                                    with --all also the llama-server this
#                                    script started. Only processes it can
#                                    identify as its own are ever signalled.
#
# And the guided rotation of the encryption key, application stopped:
# dry run, confirmation, re-encryption with backups, read-back, what to delete:
#   ./start.sh --rekey [--dry-run] [--yes] [--allow-unreadable]
#
# With API_URL naming the API of another machine, --stop, --restore and --rekey act on that
# machine through its Admin API and its host helper instead of on this one.
#
# And the host helper, which lets the API stop, start, restart, restore and rekey,
# and manage .env and models/ for a container: started with the application when
# HOST_HELPER_ENABLED=true and stopped by --stop --all (whether it runs: --status).
#   ./start.sh --native --detach | --docker --detach   -> start in the background (the helper uses it)
#
# Both run modes need a llama-server: on this Mac (Metal-accelerated inference), or, in
# native mode, a remote one named by LLAMA_SERVER_URL. If the local one isn't reachable on
# LLAMA_PORT, this script starts it itself, using LLAMA_SERVER_BIN /
# MODELS_DIR / MODEL_FILE from .env (by default ./vendor/llama.cpp and
# ./models, inside this repository) — then leaves it running in
# the background rather than stopping it on exit: reloading the model
# costs real time at this context size, so killing it every run would
# make iteration painfully slow.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# .env holds every secret: it is created readable by its owner only,
# whatever the caller's umask. A .env that already exists is never changed,
# only reported when other users can read it.
create_env_from_example() {
  (umask 077; cp .env.example .env)
}

warn_if_env_is_shared() {
  local mode
  mode="$(python3 -c "import os; print(format(os.stat('.env').st_mode & 0o777, 'o'))")"
  if [ $(( 8#$mode & 8#077 )) -ne 0 ]; then
    echo "!! .env is readable by other users (mode ${mode}) and holds secrets. Restrict it: chmod 600 .env" >&2
  fi
}

show_config() {
  # Reading creates nothing: a .env made here would stop a later --native from asking
  # the configuration questions. Without .env, the defaults are shown.
  local env_file=.env
  if [ ! -f .env ]; then
    echo "No .env yet: the defaults of .env.example are shown. ./start.sh --config --edit creates it." >&2
    env_file=.env.example
  else
    warn_if_env_is_shared
  fi
  ENV_SHOWN="$env_file" python3 - <<'PYEOF'
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "app"))
try:
    import settings_rules
except ImportError as exc:
    print(f"The configuration could not be shown: the rules module could not be loaded ({exc}).", file=sys.stderr)
    sys.exit(1)
sys.exit(settings_rules.cli_show(os.environ["ENV_SHOWN"], ".env.example"))
PYEOF
}

set_config() {
  local kv="${1:-}"
  if [[ "$kv" != *=* ]]; then
    echo "Usage: ./start.sh --config KEY=VALUE" >&2
    exit 1
  fi
  local key="${kv%%=*}"
  local value="${kv#*=}"
  if ! [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
    echo "'${key}' is not a valid variable name (UPPER_CASE letters, digits and underscores)." >&2
    exit 1
  fi
  if [ ! -f .env ]; then
    # --config KEY=VALUE changes an existing .env; creating one is --config --edit's job.
    echo "No .env yet: ${key} was NOT set. Create it first: ./start.sh --config --edit (in a terminal)," >&2
    echo "or copy .env.example to .env." >&2
    exit 1
  fi
  warn_if_env_is_shared
  python3 - "$key" "$value" <<'PYEOF'
import os
import sys

# The checks, the rules and the write are app/settings_rules.py, the module the Admin API
# route PATCH /config uses too: the script and the API cannot disagree. Nothing is
# written when it says no, or when it cannot be loaded.
sys.path.insert(0, os.path.join(os.getcwd(), "app"))
try:
    import settings_rules
except ImportError as exc:
    print(f"{sys.argv[1]} was NOT changed: the value rules could not be loaded ({exc}).", file=sys.stderr)
    sys.exit(1)
sys.exit(settings_rules.cli_set(".env", ".env.example", sys.argv[1], sys.argv[2]))
PYEOF
  echo "==> Set ${key} in .env."
  # Settings are read once, at startup. Nothing is restarted automatically:
  # a restart would interrupt live conversations.
  echo "==> Applies at the next start: a running application keeps its old value until you restart it (./start.sh --stop, then ./start.sh)."
}

# The interactive configuration: only on request (--config --edit) or on a native start
# with no .env next to this script, never otherwise. It needs a terminal; the previous .env
# is kept first. Exit codes: 0 written or nothing to change, 1 stopped, 2 not run.
configure_env() {
  if [ ! -t 0 ]; then
    echo "The configuration asks questions: it needs a terminal. Nothing was written." >&2
    return 2
  fi
  python3 app/env_wizard.py .env .env.example
}

# --- What runs, and stopping it ---
# A pid file alone is not enough to kill anything: a stale file can name a
# process that has since been given to something else. A process counts as ours
# only when it is alive AND its command line names what this script started.
pid_alive() {  # a process that has exited but was not reaped yet (state Z) is not alive
  local state
  kill -0 "$1" 2>/dev/null || return 1
  state="$(ps -p "$1" -o stat= 2>/dev/null | tr -d ' ')"
  [ -n "$state" ] && [ "${state:0:1}" != "Z" ]
}

pid_is() {  # pid_is FILE PATTERN -> 0 when FILE holds a live pid whose command matches PATTERN
  local file="$1" pattern="$2" pid
  [ -f "$file" ] || return 1
  pid="$(cat "$file" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  pid_alive "$pid" || return 1
  ps -p "$pid" -o command= 2>/dev/null | grep -q -- "$pattern"
}

api_base_url() {  # where the clients reach the Admin API: API_URL, else this machine
  # The API, this script and a UI can each run on a different machine: API_URL names
  # where the API is. Empty: this machine, where it listens (a wildcard bind is reached
  # on loopback). Same rule as app/admin/client.py::api_base_url.
  if [ -n "${API_URL:-}" ]; then
    printf '%s' "${API_URL%/}"
    return
  fi
  local host="${API_SERVER_HOST:-127.0.0.1}"
  case "$host" in
    0.0.0.0|::|"") host=127.0.0.1 ;;
    *:*) host="[${host}]" ;;
  esac
  printf 'http://%s:%s' "$host" "${API_SERVER_PORT:-8700}"
}

engine_kind() {  # "local" or "remote" for LLAMA_SERVER_URL; exit 1, reason on stderr, when refused
  # One rule, app/settings_rules.py::native_engine_problem: local = this machine
  # (localhost, loopback, host.docker.internal), the llama-server this script reuses or
  # starts on LLAMA_PORT; remote = another machine, which a native run uses as it is and
  # never starts nor stops, over https:// only (or an SSH tunnel on localhost).
  printf %s "${LLAMA_SERVER_URL:-}" | python3 app/settings_rules.py --engine-kind
}

native_engine_url() {  # the engine URL of a process run on this machine
  if [ "$1" = "remote" ]; then
    printf '%s' "${LLAMA_SERVER_URL%/}"
  else
    printf 'http://localhost:%s' "${LLAMA_PORT:-8080}"
  fi
}

backup_status() {  # backup_status API_URL -> one line about the scheduled backup
  # The credential goes to curl on its standard input (-H @-), never on its command line,
  # where any user of this machine could read it with ps. It is the token saved by
  # `--admin sign-in` for this API, else API_SERVER_KEY (app/admin/credentials.py, the
  # script client chooses the same way).
  local json
  json="$(python3 app/admin/credentials.py --header "$1" \
    | curl -s --max-time 3 -H @- "$1/backups/schedule" 2>/dev/null || true)"
  printf '%s' "$json" | python3 -c '
import json, sys
try:
    s = json.load(sys.stdin)
    enabled = s["enabled"]
except Exception:
    print("  Backups          : unknown (GET /backups/schedule did not answer)")
    sys.exit(0)
if not enabled:
    print("  Backups          : scheduled backup disabled")
elif s["failing"]:
    print("  Backups          : FAILING (last run %s): %s" % (s["last_run_at"], s["last_error"]))
else:
    print("  Backups          : every %s min, keep %s; last ok %s, next %s" % (
        s["interval_minutes"], s["keep"], s["last_success_at"] or "none yet",
        s["next_run_at"] or "at the next check (within a minute)"))
' || echo "  Backups          : unknown (the status could not be read)"
}

http_code() {  # http_code URL -> status code, 000 when nothing answers
  curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$1" 2>/dev/null || true
}

load_env_if_present() {
  if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
}

docker_running_container() {  # prints the id of this project's channelagent container, up or restarting
  command -v docker >/dev/null 2>&1 || return 1
  # A container that crashes at start is "restarting", not "running": without it --status said
  # "not running" and --stop left a crash loop going (measured 2026-10-05).
  docker compose ps --status running --status restarting -q channelagent 2>/dev/null \
    | head -n 1 | grep .
}

show_status() {
  load_env_if_present
  local llama_port="${LLAMA_PORT:-8080}" api_url code container
  echo "==> ChannelAgent status"

  if ! command -v docker >/dev/null 2>&1; then
    echo "  app (container)  : docker not available"
  elif container="$(docker_running_container)"; then
    if [ "$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)" = "restarting" ]; then
      echo "  app (container)  : restarting in a loop (${container:0:12}); see docker compose logs channelagent"
    else
      echo "  app (container)  : running (${container:0:12})"
    fi
  else
    echo "  app (container)  : not running"
  fi

  if pid_is .app.pid "app.main"; then
    echo "  app (native)     : running (pid $(cat .app.pid))"
  else
    echo "  app (native)     : not running"
  fi

  if [ -z "${API_SERVER_KEY:-}" ]; then
    echo "  Admin API        : disabled (API_SERVER_KEY is not set)"
  else
    api_url="$(api_base_url)"
    code="$(http_code "${api_url}/users")"
    # 401 (no key sent) means the API is up and refusing, which is what it should do.
    case "$code" in
      200|401|429)
        echo "  Admin API        : up at ${api_url} (HTTP ${code})"
        backup_status "$api_url"
        ;;
      *) echo "  Admin API        : down at ${api_url}" ;;
    esac
  fi

  show_host_helper
  show_searxng
  # The machine's memory, a warning below 10 GiB available (system python3, no venv).
  echo "  memory           : $(python3 app/host_memory.py 2>/dev/null || echo unknown)"

  local kind
  kind="$(engine_kind 2>/dev/null || echo invalid)"
  if [ "$kind" = "remote" ]; then
    code="$(http_code "$(native_engine_url remote)/health")"
    if [ "$code" = "200" ]; then
      echo "  llama-server     : remote, up at $(native_engine_url remote)"
    else
      echo "  llama-server     : remote, down at $(native_engine_url remote)"
    fi
    return 0
  fi
  code="$(http_code "http://localhost:${llama_port}/health")"
  if [ "$code" = "200" ]; then
    if pid_is .llama-server.pid "llama-server"; then
      echo "  llama-server     : up on port ${llama_port} (pid $(cat .llama-server.pid), started by start.sh)"
    else
      external_pid="$(lsof -ti :"${llama_port}" 2>/dev/null | head -n 1 || true)"
      if [ -n "$external_pid" ]; then
        echo "  llama-server     : up on port ${llama_port} (pid ${external_pid}, not started by start.sh)"
      else
        echo "  llama-server     : up on port ${llama_port} (not started by start.sh)"
      fi
    fi
  else
    echo "  llama-server     : down on port ${llama_port}"
  fi
}

stop_pid_file() {  # stop_pid_file LABEL FILE PATTERN
  local label="$1" file="$2" pattern="$3" pid waited=0
  if pid_is "$file" "$pattern"; then
    pid="$(cat "$file")"
    kill "$pid"
    while pid_alive "$pid" && [ "$waited" -lt 30 ]; do
      sleep 0.5
      waited=$((waited + 1))
    done
    if pid_alive "$pid"; then
      echo "!! ${label} (pid ${pid}) did not stop within 15 s; it was sent SIGTERM only." >&2
      return 1
    fi
    rm -f "$file"
    echo "==> ${label} stopped (pid ${pid})."
  elif [ -f "$file" ]; then
    pid="$(cat "$file" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && pid_alive "$pid"; then
      echo "!! ${file} names pid ${pid}, which is not ${label} (its command line does not match): left alone." >&2
    else
      rm -f "$file"
      echo "==> ${label}: stale ${file} removed (no such process)."
    fi
  else
    echo "==> ${label}: not started by start.sh, nothing to stop."
  fi
}

# --- The host helper ---
host_helper_where() {
  if [ -n "${HOST_HELPER_SOCKET:-}" ]; then
    printf 'unix:%s' "${HOST_HELPER_SOCKET}"
  else
    printf '127.0.0.1:%s' "${HOST_HELPER_PORT:-8701}"
  fi
}

ensure_host_helper_secret() {  # HOST_HELPER_SECRET generated into .env when empty, never printed
  [ -n "${HOST_HELPER_SECRET:-}" ] && return 0
  python3 - <<'PYEOF'
import os
import secrets
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "app"))
import settings_rules

settings_rules.set_config(
    ".env", ".env.example", "HOST_HELPER_SECRET", secrets.token_hex(32), allow_initial_key=False
)
PYEOF
  echo "==> HOST_HELPER_SECRET was empty: a new one was written to .env (not shown)."
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
}

start_host_helper() {  # needs the virtualenv; does nothing unless HOST_HELPER_ENABLED=true
  [ "${HOST_HELPER_ENABLED:-false}" = "true" ] || return 0
  if pid_is .host-helper.pid "app.host.helper"; then
    echo "==> Host helper already running (pid $(cat .host-helper.pid), $(host_helper_where))."
    return 0
  fi
  ensure_host_helper_secret
  mkdir -p logs
  # A minimal environment, like the engine's: the helper reads the settings it needs from
  # .env itself, and start.sh, which it runs, does the same.
  nohup env -i PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" LANG="${LANG:-C}" \
    sh -c 'echo $$ > .host-helper.pid; exec .venv/bin/python -m app.host.helper' \
    >> logs/host-helper.log 2>&1 < /dev/null &
  local waited=0
  while [ "$waited" -lt 20 ]; do
    sleep 0.5
    waited=$((waited + 1))
    pid_is .host-helper.pid "app.host.helper" || break
    if [ -n "${HOST_HELPER_SOCKET:-}" ]; then
      [ -S "${HOST_HELPER_SOCKET}" ] && break
    elif lsof -nP -iTCP:"${HOST_HELPER_PORT:-8701}" -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
  done
  if pid_is .host-helper.pid "app.host.helper"; then
    echo "==> Host helper started (pid $(cat .host-helper.pid), $(host_helper_where), log: logs/host-helper.log)."
  else
    echo "!! The host helper did not start: see logs/host-helper.log" >&2
    return 1
  fi
}

show_host_helper() {
  if [ "${HOST_HELPER_ENABLED:-false}" != "true" ]; then
    echo "  Host helper      : off (HOST_HELPER_ENABLED is not true)"
  elif pid_is .host-helper.pid "app.host.helper"; then
    echo "  Host helper      : running (pid $(cat .host-helper.pid), $(host_helper_where))"
  else
    echo "  Host helper      : enabled, not running (it starts with the application)"
  fi
}

# SearXNG: the search engine of the built-in "search" tool, run in Docker by this script
# when SEARXNG_MANAGED=true (docker-compose.yml, service searxng, profile searxng). A failure
# leaves web search off for the run and starts everything else.
searxng_settings() {  # data/searxng/settings.yml, written once: a random secret_key, json on
  local file=data/searxng/settings.yml secret
  if [ -f "$file" ]; then
    if ! grep -q -- '- json' "$file"; then
      echo "!! ${file} does not list json in search.formats: web_search cannot read the answers." >&2
    fi
    return 0
  fi
  mkdir -p data/searxng
  secret="$(openssl rand -hex 32)"
  # 644 inside data/ (700): only the container's own user (uid 977) needs to read it.
  ( umask 022
    printf 'use_default_settings: true\nserver:\n  secret_key: "%s"\n  limiter: false\n  image_proxy: false\nsearch:\n  formats:\n    - html\n    - json\n' "$secret" > "$file" )
  echo "==> Wrote ${file} (a random secret_key, json in search.formats)."
}

start_searxng() {  # start_searxng native|docker -> exports SEARXNG_URL when SearXNG answers
  [ "${SEARXNG_MANAGED:-false}" = "true" ] || return 0
  local port="${SEARXNG_PORT:-8888}" code="000" _
  if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    echo "!! SEARXNG_MANAGED=true but Docker is not running: web search is off for this run." >&2
    return 0
  fi
  searxng_settings
  echo "==> Starting SearXNG (docker compose --profile searxng up -d searxng) on 127.0.0.1:${port}."
  if ! docker compose --profile searxng up -d searxng; then
    echo "!! SearXNG did not start: web search is off for this run." >&2
    return 0
  fi
  for _ in $(seq 1 30); do
    code="$(http_code "http://127.0.0.1:${port}/healthz")"
    [ "$code" = "200" ] && break
    sleep 1
  done
  if [ "$code" != "200" ]; then
    echo "!! SearXNG does not answer on 127.0.0.1:${port} (docker compose logs searxng): web search is off for this run." >&2
    return 0
  fi
  if [ "$1" = "docker" ]; then
    SEARXNG_URL="http://searxng:8080"  # the application's container, on the same Compose network
  else
    SEARXNG_URL="http://127.0.0.1:${port}"
  fi
  export SEARXNG_URL
  echo "==>   SearXNG is up; SEARXNG_URL=${SEARXNG_URL} for this run."
}

stop_searxng() {
  [ "${SEARXNG_MANAGED:-false}" = "true" ] || return 0
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || return 0
  echo "==> Stopping SearXNG (docker compose --profile searxng stop searxng)."
  docker compose --profile searxng stop searxng
}

show_searxng() {
  local port="${SEARXNG_PORT:-8888}"
  if [ "${SEARXNG_MANAGED:-false}" != "true" ]; then
    echo "  SearXNG          : not managed (SEARXNG_MANAGED is not true)"
  elif [ "$(http_code "http://127.0.0.1:${port}/healthz")" = "200" ]; then
    echo "  SearXNG          : up at http://127.0.0.1:${port}"
  else
    echo "  SearXNG          : down on port ${port} (it starts with the application)"
  fi
}

remote_api() {  # true when API_URL names the API and this run is not the helper's own
  [ -n "${API_URL:-}" ] && [ "${START_SH_LOCAL:-}" != "1" ]
}

api_client() {  # api_client COMMAND ARGS... -> one Admin API call with the existing virtualenv
  if [ ! -x .venv/bin/python ]; then
    echo "!! No .venv here yet: run ./start.sh --admin describe once to create it." >&2
    exit 1
  fi
  exec .venv/bin/python -m app.admin.client "$@"
}

stop_things() {
  local also_llama=0 status=0
  case "${1:-}" in
    "") ;;
    --all) also_llama=1 ;;
    *) echo "Usage: ./start.sh --stop [--all]" >&2; exit 1 ;;
  esac
  load_env_if_present
  if remote_api; then
    if [ "$also_llama" = "1" ]; then
      echo "!! --all stops this machine's engine; with API_URL set, --stop acts on $(api_base_url). Run --stop --all on that machine." >&2
      exit 1
    fi
    echo "==> API_URL is set: stopping the application at $(api_base_url) through its host helper."
    api_client host-stop
  fi

  stop_pid_file "native app" .app.pid "app.main" || status=1

  if command -v docker >/dev/null 2>&1 && docker_running_container >/dev/null; then
    echo "==> Stopping the container (docker compose stop channelagent)."
    docker compose stop channelagent || status=1
  else
    echo "==> app (container): not running."
  fi

  if [ "$also_llama" = "1" ]; then
    stop_pid_file "llama-server" .llama-server.pid "llama-server" || status=1
    stop_pid_file "host helper" .host-helper.pid "app.host.helper" || status=1
    stop_searxng || status=1
  else
    echo "==> llama-server left as it is, the host helper and SearXNG too (./start.sh --stop --all stops them)."
  fi
  return "$status"
}

# One table of commands: the help is printed from it, and a test checks that every
# command of the dispatch below has a line here, so the two cannot drift apart.
usage() {
  cat <<'USAGE'
Usage: ./start.sh [COMMAND]

  (no command) | --docker      start the application in Docker (docker compose up --build)
  --native [--detach]          start the application natively (Ctrl+C stops it and the engine);
                               --detach: in the background
  --docker [--detach] [--build|--no-build]
                               --detach: in the background; rebuild the image (the default) or
                               start it as it is
  --status                     what runs: container, native app, Admin API, host helper, engine
  --stop [--all]               stop the native app or the container; --all: the engine and the
                               host helper too (with API_URL set: the remote application)
  --config                     every variable of .env.example, secrets masked (without .env:
                               the defaults, nothing is created)
  --config KEY=VALUE           add or update one variable of an existing .env, checked by its
                               rule (the previous .env is kept as .env.bak-<time>)
  --config --edit              one question per variable, then one write (needs a terminal)
  --admin                      the Admin API as menus, one per tag (API_URL, or this machine)
  --admin COMMAND [--flag V ...] [--json] [--transport auto|http|inprocess]
                               one Admin API call; --admin describe lists every command;
                               --admin sign-in --name N keeps a named administrator's token,
                               --admin sign-out drops it
  --chat [--agent NAME] [--transport auto|http|inprocess]
                               talk to an agent through the API, as the channel terminal
  --restore [FILE] [--list] [--yes] [--allow-unreadable]
                               restore a database backup, application stopped
  --rekey [--dry-run] [--yes] [--allow-unreadable]
                               rotate ENCRYPTION_KEY, application stopped
  --help | -h | help           this help
USAGE
}

# An argument a command does not take is refused, never ignored: `--native --detatch`
# used to start in the foreground. Exit 2 and the usage, before anything is installed or started.
refuse_argument() {  # refuse_argument COMMAND ARGUMENT
  echo "Unknown option for ${1}: ${2}" >&2
  usage >&2
  exit 2
}

check_chat_arguments() {  # the options of app.admin.chat: --agent NAME, --transport MODE
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --agent|--transport) [ "$#" -ge 2 ] || refuse_argument --chat "$1"; shift 2 ;;
      *) refuse_argument --chat "$1" ;;
    esac
  done
}

case "${1:-}" in
  --help|-h|help)
    usage
    exit 0
    ;;
  --config)
    case "${2:-}" in
      "") show_config; exit 0 ;;
      --edit)
        [ "$#" -le 2 ] || refuse_argument "--config --edit" "$3"
        configure_env && rc=0 || rc=$?
        exit "$rc"
        ;;
      -*) refuse_argument --config "$2" ;;
      *)
        [ "$#" -le 2 ] || refuse_argument --config "$3"
        set_config "$2"
        exit 0
        ;;
    esac
    ;;
  --status)
    [ "$#" -le 1 ] || refuse_argument --status "$2"
    show_status
    exit 0
    ;;
  --stop)
    [ "$#" -le 2 ] || refuse_argument --stop "$3"
    stop_things "${2:-}"
    exit $?
    ;;
esac

# Docker mode only without a command or with --docker: anything else unknown (a typo) is
# refused before anything starts; it used to run docker compose up --build.
MODE="docker"
case "${1:-}" in
  ""|--docker) MODE="docker" ;;
  --native) MODE="native" ;;
  --admin) if [ "$#" -gt 1 ]; then MODE="api"; else MODE="admin"; fi ;;
  --restore) MODE="restore" ;;
  --rekey) MODE="rekey" ;;
  --chat) MODE="chat" ;;
  *)
    echo "Unknown command: ${1}" >&2
    # The forms removed: say what replaced them, without keeping them as aliases.
    case "${1}" in
      --show-config) echo "It is now ./start.sh --config" >&2 ;;
      --set) echo "It is now ./start.sh --config KEY=VALUE" >&2 ;;
      --configure) echo "It is now ./start.sh --config --edit" >&2 ;;
      --api) echo "It is now ./start.sh --admin COMMAND (./start.sh --admin describe lists them)" >&2 ;;
      --models) echo "It is now ./start.sh --admin list-models | pull-model --spec X | import-model --path X | delete-model --name X" >&2 ;;
      --host-helper) echo "The host helper starts with the application and stops with ./start.sh --stop --all" >&2 ;;
      --describe) echo "It is now ./start.sh --admin describe" >&2 ;;
      *) usage >&2 ;;
    esac
    exit 2
    ;;
esac

# --detach: start in the background and return, for the host helper's start.
DETACH=0
# The arguments of each command are checked here, before the .env step, the virtualenv and
# the engine. --admin COMMAND is generated from the routes: its client checks them.
case "$MODE" in
  native)
    for arg in "${@:2}"; do
      case "$arg" in
        --detach) DETACH=1 ;;
        *) refuse_argument --native "$arg" ;;
      esac
    done
    ;;
  api)
    # --admin COMMAND: the first word is a command name; the client's options (--json,
    # --transport, --no-wait) come after it. A leading option is a typo.
    case "$2" in
      -h|--help) ;;
      -*) refuse_argument --admin "$2" ;;
    esac
    ;;
  rekey)
    for arg in "${@:2}"; do
      case "$arg" in
        --yes|--dry-run|--allow-unreadable) ;;
        *) echo "Usage: ./start.sh --rekey [--dry-run] [--yes] [--allow-unreadable]" >&2; exit 2 ;;
      esac
    done
    ;;
  restore)
    restore_files=0
    for arg in "${@:2}"; do
      case "$arg" in
        --yes|--list|--allow-unreadable) ;;
        -*) echo "Usage: ./start.sh --restore [FILE] [--list] [--yes] [--allow-unreadable]" >&2; exit 2 ;;
        *) restore_files=$((restore_files + 1)) ;;
      esac
    done
    if [ "$restore_files" -gt 1 ]; then
      echo "Usage: ./start.sh --restore [FILE] [--list] [--yes] [--allow-unreadable]" >&2
      exit 2
    fi
    ;;
  chat) check_chat_arguments "${@:2}" ;;
esac
# Docker mode: --no-build starts the image as it is, --build rebuilds it (the default,
# detached or not: an image older than the database's migrations crashes at start, measured
# 2026-10-05; Docker's cache makes an unchanged rebuild quick). Anything else after --docker is
# refused, as an unknown command is.
BUILD=""
if [ "$MODE" = "docker" ] && [ -n "${1:-}" ]; then
  for arg in "${@:2}"; do
    case "$arg" in
      --detach) DETACH=1 ;;
      --build) BUILD=1 ;;
      --no-build) BUILD=0 ;;
      *) echo "Unknown option for --docker: ${arg}" >&2; usage >&2; exit 2 ;;
    esac
  done
fi

echo "==> ChannelAgent start.sh (mode: $MODE)"

# --- 1. .env must exist ---
# A native start in a terminal asks the questions; otherwise .env.example is copied.
if [ ! -f .env ] && [ "$MODE" = "native" ] && [ -t 0 ]; then
  echo "==> No .env next to start.sh: configuring it now."
  if ! configure_env; then
    echo "!! No .env was created: the application was not started. Run ./start.sh --config --edit." >&2
    exit 1
  fi
fi
if [ ! -f .env ]; then
  echo "!! No .env found. Copying .env.example — fill in real values before running the app" >&2
  echo "!! (./start.sh --config --edit asks for them one by one)." >&2
  create_env_from_example
fi
warn_if_env_is_shared

set -a
# shellcheck disable=SC1091
source .env
set +a

# A machine that is only a client of another machine's API (API_URL set: --admin,
# --restore NAME --yes, --rekey --yes) never needs the encryption key: the API it
# calls holds it. Every other mode reads the database here, so it needs the key.
client_only=0
case "$MODE" in
  admin|api|chat|restore|rekey) remote_api && client_only=1 ;;
esac
if [ "$client_only" = "0" ] && ! grep -q "^ENCRYPTION_KEY=.\+" .env; then
  echo "!! ENCRYPTION_KEY is missing or empty in .env — the app will refuse to start without it." >&2
  echo "!! Generate one with: python3 -c \"import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())\"" >&2
  exit 1
fi

# --- 1a. The Admin API stays on loopback unless a TLS proxy is declared ---
# One rule, app/settings_rules.py::api_exposure_problem, which the application runs too.
if [ "$MODE" = "native" ] && [ -n "${API_SERVER_KEY:-}" ]; then
  python3 app/settings_rules.py --exposure API_SERVER_HOST || exit 1
fi
if [ "$MODE" = "docker" ] && [ -n "${API_SERVER_KEY:-}" ]; then
  python3 app/settings_rules.py --exposure API_BIND_ADDRESS || exit 1
fi

# --- 1b. One instance at a time ---
# Two instances poll the same Telegram token and mailbox and want the same API port. A
# native start while the container ran (2026-09-28) polled the mailbox next to it, lost
# the port ("[Errno 48] address already in use") and overwrote .run-mode: refuse first.
# Only the two starts: --admin and --chat talk to the running app, and
# --restore and --rekey have their own "application stopped" check.
api_port="${API_SERVER_PORT:-8700}"
if { [ "$MODE" = "native" ] || [ "$MODE" = "docker" ]; } && pid_is .app.pid "app.main"; then
  echo "!! The native app is already running (pid $(cat .app.pid)). Stop it first: ./start.sh --stop" >&2
  exit 1
fi
if [ "$MODE" = "native" ]; then
  if container="$(docker_running_container)"; then
    echo "!! The channelagent container is already running (${container:0:12}): it holds port ${api_port}" >&2
    echo "!! and polls the same bot and mailbox. Stop it first: ./start.sh --stop" >&2
    exit 1
  fi
  # A connection, not lsof: lsof may be missing from PATH (/usr/sbin), which would skip the
  # check. A wildcard bind is reached on loopback.
  api_probe_host="${API_SERVER_HOST:-127.0.0.1}"
  case "$api_probe_host" in 0.0.0.0|::|"[::]") api_probe_host=127.0.0.1 ;; esac
  if (exec 3<>"/dev/tcp/${api_probe_host}/${api_port}") 2>/dev/null; then
    echo "!! Port ${api_port} (API_SERVER_PORT) is already in use on ${api_probe_host}." >&2
    lsof -nP -iTCP:"${api_port}" -sTCP:LISTEN >&2 2>/dev/null || true
    echo "!! Stop that process, or pick another port: ./start.sh --config API_SERVER_PORT=<port>" >&2
    exit 1
  fi
fi

# The project's Python: .python-version, the same version as the image, CI and the
# dependency locks. The virtualenv is made with it, and remade when it was made with another.
PYTHON_VERSION="$(tr -d '[:space:]' < "$REPO_DIR/.python-version" 2>/dev/null || true)"
PYTHON_VERSION="${PYTHON_VERSION:-3.14}"

python_minor() {  # python_minor INTERPRETER -> "3.14", empty when it does not run
  "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

find_python() {  # the first of python3.X and python3 that is the project's version
  local candidate
  for candidate in "python${PYTHON_VERSION}" python3; do
    if command -v "$candidate" >/dev/null 2>&1 \
        && [ "$(python_minor "$candidate")" = "$PYTHON_VERSION" ]; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

ensure_venv_python() {
  local found python
  if [ -x .venv/bin/python ]; then
    found="$(python_minor .venv/bin/python)"
    if [ "$found" != "$PYTHON_VERSION" ]; then
      echo "==> .venv uses Python ${found:-unknown}, the project uses ${PYTHON_VERSION}: recreating it"
      rm -rf .venv
    fi
  fi
  if [ ! -d .venv ]; then
    if ! python="$(find_python)"; then
      echo "!! Python ${PYTHON_VERSION} is required (python${PYTHON_VERSION}, or a python3 of that" \
        "version); python3 here: $(python3 --version 2>&1 || echo none)" >&2
      exit 1
    fi
    echo "==> Creating virtualenv (.venv, Python ${PYTHON_VERSION})"
    "$python" -m venv .venv
  fi
}

setup_venv() {
  ensure_venv_python
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "==> Installing dependencies"
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
  # The secret scanner (scripts/secret_scan.sh) lives in the virtualenv too: a pinned, checked
  # release, installed once. The application does not need it, so a failure only warns.
  if [ -f scripts/install_gitleaks.sh ]; then
    bash scripts/install_gitleaks.sh \
      || echo "!! gitleaks could not be installed: scripts/secret_scan.sh will not run." >&2
  fi
  # The headless browser, only when it is switched on, inside this repository
  # (vendor/playwright), never in a user cache. Installed once, kept afterwards.
  if [ "${WEB_FETCH_BROWSER:-false}" = "true" ] && ! ls -d vendor/playwright/chromium_headless_shell-* >/dev/null 2>&1; then
    echo "==> Installing the headless browser for web pages (vendor/playwright, about 200 MB)"
    PLAYWRIGHT_BROWSERS_PATH="$REPO_DIR/vendor/playwright" python -m playwright install chromium-headless-shell \
      || echo "!! The browser could not be installed: pages built by scripts will not be read." >&2
  fi
}

# --- --admin alone: the menus, a client of the Admin API, no llama-server needed ---
if [ "$MODE" = "admin" ]; then
  setup_venv
  exec python3 -m app.admin.cli
fi

# --- Rotate ENCRYPTION_KEY: the guided sequence, application stopped ---
# `.env` was just sourced, which exported the OLD ENCRYPTION_KEY: the tool must
# read the key from the file and never from the environment, so it is removed.
if [ "$MODE" = "rekey" ] && remote_api; then
  shift
  rekey_args=() rekey_yes=0
  for arg in "$@"; do
    case "$arg" in
      --yes) rekey_yes=1 ;;
      --dry-run) rekey_args+=(--dry-run true); rekey_yes=1 ;;
      --allow-unreadable) rekey_args+=(--allow-unreadable true) ;;
      *) echo "Usage: ./start.sh --rekey [--dry-run] [--yes] [--allow-unreadable]" >&2; exit 2 ;;
    esac
  done
  if [ "$rekey_yes" != "1" ]; then
    echo "!! API_URL is set: the key of $(api_base_url) is rotated without a prompt; add --yes (or --dry-run first)." >&2
    exit 2
  fi
  api_client host-rekey ${rekey_args[@]+"${rekey_args[@]}"}
fi
if [ "$MODE" = "rekey" ]; then
  setup_venv
  shift
  exec env -u ENCRYPTION_KEY -u OLD_ENCRYPTION_KEY python3 -m app.admin.rekey_guided "$@"
fi

# --- --admin COMMAND: one call, the client is generated from the API's routes ---
if [ "$MODE" = "api" ]; then
  # In process, the client runs on the host, where host.docker.internal does not resolve:
  # reach the engine the way --native does. Over HTTP the running application has its own.
  # Classified with the system python3, before the virtualenv, like --config.
  engine="$(engine_kind)" || exit 1
  setup_venv
  shift
  LLAMA_SERVER_URL="$(native_engine_url "$engine")"
  export LLAMA_SERVER_URL
  exec python3 -m app.admin.client "$@"
fi

# --- The terminal channel: a chat with an agent through the API ---
if [ "$MODE" = "chat" ]; then
  # In process the turn runs here, so the engine is reached the way --native does.
  engine="$(engine_kind)" || exit 1
  setup_venv
  shift
  LLAMA_SERVER_URL="$(native_engine_url "$engine")"
  export LLAMA_SERVER_URL
  exec python3 -m app.admin.chat "$@"
fi

# --- Restore a backup: same venv, no llama-server, the application must be stopped ---
if [ "$MODE" = "restore" ] && remote_api; then
  shift
  restore_name="" restore_yes=0 restore_list=0 restore_args=()
  for arg in "$@"; do
    case "$arg" in
      --yes) restore_yes=1 ;;
      --list) restore_list=1 ;;
      --allow-unreadable) restore_args+=(--allow-unreadable true) ;;
      -*) echo "Usage: ./start.sh --restore [FILE] [--list] [--yes] [--allow-unreadable]" >&2; exit 2 ;;
      *) restore_name="$arg" ;;
    esac
  done
  if [ "$restore_list" = "1" ] || [ -z "$restore_name" ]; then
    [ -z "$restore_name" ] && [ "$restore_list" != "1" ] \
      && echo "==> API_URL is set: name the backup to restore (./start.sh --restore NAME --yes)." >&2
    api_client list-database-backups
  fi
  if [ "$restore_yes" != "1" ]; then
    echo "!! API_URL is set: the database of $(api_base_url) is replaced without a prompt; add --yes." >&2
    exit 2
  fi
  api_client restore-backup --name "$restore_name" ${restore_args[@]+"${restore_args[@]}"}
fi
if [ "$MODE" = "restore" ]; then
  setup_venv
  shift
  exec python3 -m app.admin.restore "$@"
fi

# --- 2. Native llama-server: reuse it if running, start it if not (macOS only) ---
# LLAMA_ROUTER_MODE=true starts it in its own router mode instead of loading one
# fixed MODEL_FILE: every .gguf in MODELS_DIR becomes selectable by the request's "model"
# field, loaded on demand up to LLAMA_MODELS_MAX at once. Off by default so an existing
# single-model setup, and the live container it serves, is untouched by this change.
LLAMA_PORT="${LLAMA_PORT:-8080}"
# Native mode with a remote engine: nothing to start here, only check it answers.
# Docker mode keeps its behaviour for now (native first).
ENGINE_KIND="local"
if [ "$MODE" = "native" ]; then
  ENGINE_KIND="$(engine_kind)" || exit 1
fi
if [ "$ENGINE_KIND" = "remote" ]; then
  if curl -sf --max-time 5 "$(native_engine_url remote)/health" >/dev/null 2>&1; then
    echo "==> Inference engine: remote, up at $(native_engine_url remote) (not started nor stopped by this script)"
  else
    echo "!! The remote inference engine does not answer at $(native_engine_url remote)/health." >&2
    echo "!! Start it on its machine, or set LLAMA_SERVER_URL (./start.sh --config LLAMA_SERVER_URL=...)." >&2
    exit 1
  fi
elif [ "$(uname -s)" = "Darwin" ]; then
  router_mode="${LLAMA_ROUTER_MODE:-false}"
  if [ "$router_mode" = "true" ]; then
    model_args_ok=0
    [ -n "${LLAMA_SERVER_BIN:-}" ] && [ -x "${LLAMA_SERVER_BIN}" ] \
      && [ -d "${MODELS_DIR:-}" ] && model_args_ok=1
    model_args=(--models-dir "${MODELS_DIR:-}" --models-max "${LLAMA_MODELS_MAX:-4}")
  else
    model_args_ok=0
    [ -n "${LLAMA_SERVER_BIN:-}" ] && [ -x "${LLAMA_SERVER_BIN}" ] \
      && [ -n "${MODEL_FILE:-}" ] && [ -f "${MODELS_DIR:-}/${MODEL_FILE}" ] && model_args_ok=1
    model_args=(--model "${MODELS_DIR:-}/${MODEL_FILE:-}")
  fi
  if curl -sf --max-time 2 "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; then
    if [ "$router_mode" = "false" ] && ! pid_is .llama-server.pid "llama-server"; then
      external_pid="$(lsof -ti :"${LLAMA_PORT}" 2>/dev/null | head -n 1 || true)"
      echo "⚠️  A llama-server is already running on port ${LLAMA_PORT} (outside start.sh)."
      [ -n "$external_pid" ] && echo "⚠️  PID: ${external_pid}  —  stop it with: kill ${external_pid}"
      echo "⚠️  Ensure it is using the expected model (${MODEL_FILE:-})."
    else
      llama_pid="$(cat .llama-server.pid 2>/dev/null || true)"
      echo "==> llama-server already running on localhost:${LLAMA_PORT}"
      [ -n "$llama_pid" ] && echo "==>   PID: ${llama_pid}  |  stop: kill ${llama_pid}  (or ./start.sh --stop --all)"
      if [ "$router_mode" = "false" ]; then
        echo "==>   Model: ${MODEL_FILE:-?}  |  log: logs/llama-server.log"
      else
        echo "==>   Mode: router (models in ${MODELS_DIR:-?})  |  log: logs/llama-server.log"
      fi
    fi
  elif [ "$model_args_ok" = "1" ]; then
    echo "==> llama-server not running — starting it (loading the model can take a while)."
    if [ "$router_mode" = "false" ]; then
      echo "==>   Model: ${MODELS_DIR:-}/${MODEL_FILE:-}  |  ctx: ${LLAMA_CTX_SIZE:-65536}  |  port: ${LLAMA_PORT}"
    else
      echo "==>   Router mode  |  models-dir: ${MODELS_DIR:-}  |  max: ${LLAMA_MODELS_MAX:-4}  |  port: ${LLAMA_PORT}"
    fi
    mkdir -p logs
    # The engine saves a conversation's cache in slots/ next to the main database
    # (data/slots, the folder app/slot_cache.py reads); one model only (router mode: off).
    slot_args=()
    if [ "$router_mode" = "false" ] && [ "${SLOT_CACHE_MAX_MB:-4096}" != "0" ]; then
      slot_db="${DATABASE_URL:-sqlite+aiosqlite:///./data/channelagent.db}"
      slot_root="$(dirname "${slot_db#*:///}")"
      [ -d "$slot_root" ] || slot_root=data
      mkdir -p "$slot_root/slots" && chmod 700 "$slot_root/slots"
      slot_args=(--slot-save-path "$(cd "$slot_root/slots" && pwd)")
    fi
    # A minimal environment: the engine needs none of the secrets of .env, which
    # this script exported above, and a child process must not inherit them. The one
    # exception is its own bearer key, given as LLAMA_API_KEY in its environment,
    # never with --api-key on its command line, where any user sees it with `ps`.
    # Note: --skip-chat-parsing is intentionally omitted: with it, tool calls come
    # back as raw JSON text in content with empty tool_calls, breaking MCP and tool support.
    # umask 077 in the engine's subshell only: its saved slots are readable by this user alone.
    (umask 077; exec nohup env -i PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" LANG="${LANG:-C}" \
      ${LLAMA_SERVER_API_KEY:+"LLAMA_API_KEY=$LLAMA_SERVER_API_KEY"} \
      "${LLAMA_SERVER_BIN}" \
      --port "${LLAMA_PORT}" \
      --host 127.0.0.1 \
      "${model_args[@]}" \
      --ctx-size "${LLAMA_CTX_SIZE:-65536}" \
      -ngl 99 \
      --jinja \
      --flash-attn on \
      -ctk q8_0 \
      -ctv q8_0 \
      --predict 4096 \
      --repeat-penalty 1.1 \
      ${slot_args[@]+"${slot_args[@]}"}) \
      > logs/llama-server.log 2>&1 &
    echo $! > .llama-server.pid
    echo "==> Waiting for it to become healthy (pid $(cat .llama-server.pid), log: logs/llama-server.log)..."
    ready=0
    for _ in $(seq 1 150); do
      if curl -sf --max-time 2 "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 2
    done
    if [ "$ready" = "1" ]; then
      llama_pid="$(cat .llama-server.pid)"
      echo "==> llama-server is up"
      echo "==>   PID:  ${llama_pid}"
      echo "==>   URL:  http://127.0.0.1:${LLAMA_PORT}  (health: /health)"
      if [ "$router_mode" = "false" ]; then
        echo "==>   Model: ${MODEL_FILE:-?}"
      else
        echo "==>   Mode: router (${MODELS_DIR:-})"
      fi
      echo "==>   Log:  logs/llama-server.log"
      echo "==>   Stop: kill ${llama_pid}   or   ./start.sh --stop --all"
      echo "==>   (--native: Ctrl+C stops it too; otherwise it keeps running after this script exits)"
    else
      echo "!! llama-server did not become healthy in time — check logs/llama-server.log" >&2
      exit 1
    fi
  elif [ "$router_mode" = "true" ]; then
    echo "!! llama-server is not reachable on localhost:${LLAMA_PORT}, and LLAMA_SERVER_BIN /" >&2
    echo "!! MODELS_DIR are not both set to valid paths in .env, so router mode can't be" >&2
    echo "!! started automatically. Set those two (see .env.example), or start it yourself." >&2
    exit 1
  else
    echo "!! llama-server is not reachable on localhost:${LLAMA_PORT}, and LLAMA_SERVER_BIN /" >&2
    echo "!! MODELS_DIR / MODEL_FILE are not all set to valid paths in .env, so it can't be" >&2
    echo "!! started automatically. Set those three (see .env.example), or start it yourself." >&2
    exit 1
  fi
fi

# --- 3. Hand off to the selected mode ---
if [ "$MODE" = "docker" ]; then
  # The container runs as uid 10001, not root. If ./data does not exist, Docker
  # creates it owned by root and the application then cannot write to it: create it
  # here, as the current user, first. On Linux the directory must also belong to
  # uid 10001 (Docker Desktop on macOS and Windows maps ownership by itself).
  mkdir -p data
  if [ "$(uname -s)" = "Linux" ] && [ "$(stat -c %u data 2>/dev/null || echo 10001)" != "10001" ]; then
    echo "!! data/ is not owned by uid 10001, the user the container runs as." >&2
    echo "!! Run once: sudo chown -R 10001:10001 data" >&2
  fi
  echo docker > .run-mode
  # Docker mode uses the llama-server this script reuses or starts on LLAMA_PORT (a remote
  # engine is native mode only for now): the container reaches it on that port of the
  # host, whatever port .env's LLAMA_SERVER_URL names (with LLAMA_PORT=8090 and :8080 in the
  # URL, the application found no engine, measured 2026-09-28). docker-compose.yml passes
  # LLAMA_SERVER_URL from this environment.
  export LLAMA_SERVER_URL="http://host.docker.internal:${LLAMA_PORT:-8080}"
  echo "==> Docker mode: inference engine (${ENGINE_KIND}) at ${LLAMA_SERVER_URL:-}"
  if [ "${HOST_HELPER_ENABLED:-false}" = "true" ]; then
    setup_venv
    start_host_helper || exit 1
  fi
  start_searxng docker
  build_flag="--build"
  if [ "$BUILD" = "0" ]; then
    build_flag=""  # the image as it is
  fi
  if [ "$DETACH" = "1" ]; then
    echo "==> Starting the container in the background (docker compose up -d ${build_flag})."
    exec docker compose up -d ${build_flag:+"$build_flag"}
  fi
  echo "==> Starting the container (docker compose up ${build_flag})."
  exec docker compose up ${build_flag:+"$build_flag"}
fi

# --- native mode ---
setup_venv

# Running outside Docker: host.docker.internal does not resolve here, so a local engine is
# reached on localhost:LLAMA_PORT; a remote one is used as .env names it.
LLAMA_SERVER_URL="$(native_engine_url "$ENGINE_KIND")"
export LLAMA_SERVER_URL
echo "==> Native mode: inference engine (${ENGINE_KIND}) at ${LLAMA_SERVER_URL}"

echo native > .run-mode
start_host_helper || exit 1
start_searxng native

if [ "$DETACH" = "1" ]; then
  mkdir -p logs
  nohup sh -c 'echo $$ > .app.pid; exec python3 -m app.main' >> logs/app.log 2>&1 < /dev/null &
  echo "==> Started app/main.py in the background (pid file .app.pid, log: logs/app.log)."
  echo "==>   Stop it with ./start.sh --stop."
  exit 0
fi

echo "==> Starting app/main.py natively."
echo "==>   Admin API: listens on ${API_SERVER_HOST:-127.0.0.1}:${API_SERVER_PORT:-8700}  |  clients use: $(api_base_url)"
echo "==>   Admin UI:  $(api_base_url)/ui/  (sign in with API_SERVER_KEY)"
if [ "$ENGINE_KIND" = "remote" ]; then
  echo "==>   Ctrl+C here stops the app (the remote engine is left alone)."
else
  echo "==>   Ctrl+C here stops the app and the llama-server (as ./start.sh --stop --all)."
fi
echo "==>   From another terminal: ./start.sh --stop (app only) or ./start.sh --stop --all"

# Ctrl+C sends SIGINT to the whole foreground group: the app shuts down by itself, and
# bash runs this trap only once that foreground command has ended. llama-server, although
# started with `&`, installs its own SIGINT handler and usually stops too (measured on
# 2026-09-27: "cleaning up before exit" in its log). The trap then does what --stop --all
# does: it stops whatever is still running and removes the pid files. The container is not
# touched: it is not part of this run.
on_ctrl_c() {
  trap - INT
  echo
  echo "==> Ctrl+C: cleaning up (as ./start.sh --stop --all, the container aside)."
  echo "==>   A \"stale ... removed\" line means that process had already stopped by itself."
  stop_pid_file "native app" .app.pid "app.main" || true
  stop_pid_file "llama-server" .llama-server.pid "llama-server" || true
  stop_pid_file "host helper" .host-helper.pid "app.host.helper" || true
  stop_searxng || true
  exit 130
}
trap on_ctrl_c INT
# The app runs in the foreground, not with `&` (a background job ignores SIGINT). `sh -c`
# writes its own pid ($$ there; macOS bash 3.2 has no $BASHPID), then `exec` keeps it:
# --status and --stop find the application through it.
sh -c 'echo $$ > .app.pid; exec python3 -m app.main'
