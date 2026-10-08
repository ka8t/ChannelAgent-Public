# ChannelAgent

**A private AI assistant you reach from Telegram or e-mail, running entirely on your own machine.**

You write to it like to a person. It answers with a language model that runs locally
(`llama-server`), so no message ever leaves the machine. Several people can use it, each with
their own conversations, and each person can create agents that work on their own on a schedule
("every Thursday at 9, send me the AI news").

## Why

- **Privacy**: conversations stay on your machine, encrypted at rest. No cloud AI service, no account.
- **Nothing to install for users**: they use the apps they already have (Telegram, e-mail).
- **Several users, under control**: nobody gets in without your approval; everything is logged.
- **Agents that work alone**: a reminder, a daily summary or a weekly news digest, delivered to the chat.

## What it looks like

**1. Ask a question (Telegram)**, reply shortened

```
You:    What is the difference between RAM and VRAM, in two sentences?
Agent:  RAM is the main memory of the computer ... VRAM is the memory of the graphics card ...
```

**2. Schedule a prompt**: a real exchange, 2026-09-28

```
You:    /task add daily 09:21 Donne-moi une citation courte
Agent:  Task created. #1 [on] daily 09:21, next 2026-09-28 09:21: Donne-moi une citation courte

        (at 09:21)
Agent:  Scheduled task #1:
        « Sois toi-même, car les autres sont déjà pris. » — William Shakespeare
```

**3. Create an agent by describing it**

```
You:    /newagent every Thursday at 9, give me the AI news from my feeds
Agent:  Here is the agent I will create:
        Name: ai-news
        Purpose: Weekly digest of AI news
        Tools: mcp__feeds__read_feed
        Schedule: cron 0 9 * * 4 (Europe/Paris). Next runs: Thu 2026-10-01 09:00,
                  Thu 2026-10-08 09:00, Thu 2026-10-15 09:00
        Delivered on: telegram
        Answer yes to create it, test to run it once now first, no to cancel,
        or tell me what to change.
You:    yes
Agent:  Agent 'ai-news' created.
```

If something is missing (a feed, a time zone), the agent asks one question at a time before the
summary. `/myagents` lists your agents; `/editagent ai-news move it to 8:00` changes one.

**4. A stranger writes to the bot**

```
Stranger: hello
Agent:    You're not authorized to use this bot yet. Your request has been recorded
          and an admin will review it.
```

You, the administrator, see and approve the request:

```
./start.sh --admin list-requests
./start.sh --admin approve-request --request-id 3
```

**5. By e-mail**: send a message whose subject contains `[agent]`; the reply comes back by e-mail.
Other mail in the same mailbox is never touched.

On Telegram the bot shows "typing" as soon as it has your message, then the answer as it is
written.

## How it works

```
Telegram / e-mail / ./start.sh --chat
   │
   ▼
access check (database) ── unknown sender: access request, admins told, no model call
   │
   ├── /newagent, /editagent ──> agent builder: picks a template, asks what is missing,
   │                             shows a summary, creates the agent and its task on "yes"
   ├── /task, /agent, /model (alone), /new, /export ──> answered without the model
   │
   ▼
the turn of the user's current agent
   1. model: the one named in /model <name> <message>, else the agent's, else the tool model
      (agent with tools), else the routing rules, else the default model
   2. context: the agent's system prompt, its memory, a summary of older messages and the
      recent conversation (one conversation per user and agent, encrypted)
   3. llama-server answers, or asks for tools (MCP)
   4. each tool call is checked (granted to this user? definition approved? policy allow,
      confirm or deny?); a "confirm" tool asks the user yes or no first; the result goes back
      to the model, until it answers
   │
   ▼
the answer, shown while it is written; the message, the answer and every tool call are in the
audit trail
```

A scheduled task runs the same turn at its time and sends the answer on the channel chosen for
it. Administration (`./start.sh` and the web UI) goes through one Admin API to the same database.
Details and diagrams: [``].

## Requirements

- macOS on Apple Silicon (the model runs on the GPU through Metal) or Linux; Docker for the
  container mode.
- Python 3.14 (`python3.14`, or a `python3` of that version), `git`, `curl`, `openssl`.
- A `llama-server` build of [llama.cpp](https://github.com/ggml-org/llama.cpp) (the release
  archive for your platform) and a model in GGUF format, for example from Hugging Face
  (`./start.sh --admin pull-model --spec <repo>` downloads one once the application runs).
- A Telegram bot token, an IMAP/SMTP mailbox, or both.

## Start it

1. Put the engine and a model in the repository (both git-ignored): unpack the llama.cpp
   release into `vendor/llama.cpp/` (so that `vendor/llama.cpp/llama-server` exists) and put
   the model in `models/<model>.gguf`.
2. Run `./start.sh --native`. With no `.env` next to `start.sh`, it asks one question per
   setting, with the explanation of `.env.example` above each one (Enter keeps the default,
   Enter on `ENCRYPTION_KEY` generates one), then writes `.env` (mode 600) once you confirm.
   Answer at least `API_SERVER_KEY` (`openssl rand -hex 32`), `MODEL_FILE` (`<model>.gguf`),
   and a channel: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USERS`, or the `EMAIL_*` settings
   (see "Who can use the bot"). Ctrl+C stops without writing anything.
   Later: `./start.sh --config --edit` asks the same questions about the current `.env`, or
   `./start.sh --config KEY=VALUE` changes one setting. Without a terminal, `.env.example` is
   copied as before.
3. `./start.sh --native` starts the model and the application. Open
   `http://127.0.0.1:8700/` in a browser for the admin UI (sign in with `API_SERVER_KEY`).
4. Create your named owner account, then sign in with it (see "Administrators").

`./start.sh` alone runs the application in Docker instead (`docker-compose.yml`), with the
engine still native on the host; `docker-compose.prod.yml` puts the engine in a container too,
for a server without a GPU.

Keep `ENCRYPTION_KEY` in a password manager: without it the data cannot be read.

## Administrators

`API_SERVER_KEY` opens the Admin API only until the first named owner exists. Then each
administrator signs in with a name, a password and, for an owner, a 6-digit code from an
authenticator app (TOTP); `start.sh` and the admin UI both act as that person.

```bash
./start.sh --admin create-admin --name alice --scope owner   # asks the password
# The answer shows totp_secret and totp_uri once: add them to an authenticator app.
./start.sh --admin sign-in --name alice    # asks the password and the code
./start.sh --admin whoami                  # adm:alice, scope owner
./start.sh --admin sign-out
```

- Scopes: `read`, `operate`, `admin`, `owner`; a token never goes above its account's scope.
  `create-admin --scope admin` (or below) needs no code unless `--with-totp true`.
- `sign-in` keeps the token (30 days by default, `--hours`) in
  `~/.config/channelagent/api-tokens.json` (mode 600), one per API address. `list-tokens`,
  `revoke-token`, `disable-admin`, `set-admin-password`, `reset-admin-totp` manage them.
- Lost code or password: stop the application, then the in-process commands still accept
  `API_SERVER_KEY` (the operating-system user already holds `.env` and the database):
  `./start.sh --admin reset-admin-totp --name alice --transport inprocess`.
- `./start.sh --admin verify-audit` checks the chain of the admin events: an edited or removed
  event is reported. Keep its `last_hash` somewhere else to notice removed recent events too.
- The API stays on loopback: a non-loopback `API_SERVER_HOST` or `API_BIND_ADDRESS` is refused
  unless `API_REMOTE=tls-proxy` says a TLS proxy is in front (`docker-compose.tls.yml`).

## Who can use the bot (Telegram and email)

Access is in the database (`data/channelagent.db`, SQLite), not in `.env`, and changes without a
restart. Three tables decide it: `users` (a person), `channel_identities` (a Telegram id or an
email address belonging to a user; an address is stored hashed for lookups and encrypted) and
`permissions` (per identity: `chat` may talk to the bot, `admin` may too and is told on
Telegram of every new access request). Manage them with `./start.sh --admin ...`, the menus of
`./start.sh --admin`, or the admin UI (the *Access requests* page, and every command under
*All operations*). A sender with no identity
or no `chat` gets nothing from the model; their message is recorded as an access request.

**A Telegram bot.**
1. In Telegram, open @BotFather, send `/newbot`, choose a display name, then a username ending
   in `bot`. BotFather answers with the token: `./start.sh --config TELEGRAM_BOT_TOKEN=<token>`
   (a secret; `/revoke` in BotFather replaces it). `/setdescription` and `/setuserpic` there
   change how the bot looks; nothing else is needed there.
2. The first administrator, on an empty database: put your numeric Telegram id in
   `TELEGRAM_ALLOWED_USERS` before the first start (read once, then never again). Not known?
   Leave it empty, start, write to the bot, then:
   ```
   ./start.sh --admin list-requests          # your request: its id and your numeric external_id
   ./start.sh --admin approve-request --request-id 1               # creates your user, with chat
   ./start.sh --admin list-channel-identities --user-id 1          # the id of your identity
   ./start.sh --admin grant --user-id 1 --channel-identity-id 1 --kind admin
   ```
3. Other people: give them the bot's username. Their first message gets "You're not authorized
   to use this bot yet...", and every admin gets a Telegram message with the request number.
   Approve it (admin UI, *Access requests*, or `approve-request` as above) or refuse it
   (`deny-request --request-id N`).

**Email.** The bot reads one mailbox (the `EMAIL_*` settings; IMAP over TLS on 993, SMTP over
TLS on 465) and only the messages whose subject holds `EMAIL_TRIGGER_TAG` (default `[agent]`);
the rest of the mailbox is left alone. A stranger gets no reply (an address can be forged): their
message becomes an access request and the admins are told on Telegram. To let someone in:
```
./start.sh --admin create-user --display-name "Alice"                    # prints the user id
./start.sh --admin add-channel-identity --user-id 2 --channel email --identifier alice@example.org
./start.sh --admin grant --user-id 2 --channel-identity-id 3 --kind chat
```
`add-channel-identity` alone gives no access: the `grant` is needed. The same person can have
a Telegram and an email identity on one user, each with its own permissions and its own
conversation. Approving an email access request works as for Telegram.

**Taking access back.** `./start.sh --admin revoke --user-id 2 --channel-identity-id 3 --kind chat`
for one identity, `update-user --user-id 2 --is-active false` for the whole user.

**What a user can do.** By default: talk to their own agent, 20 messages a minute, one answer at
a time, at most 10 agents and 20 tasks, a task at most every 15 minutes
(`TASK_MIN_INTERVAL_MINUTES`). No tool unless you grant it. They never see other users'
conversations; you can read theirs (logs), so tell them. The engine answers one person at a
time: with several users, answers queue.

## Models

The engine is `llama-server`; models are GGUF files in `models/` (`MODELS_DIR`).

| Step | Command |
|---|---|
| See the models and the one loaded | `./start.sh --admin list-models` |
| Download one (Hugging Face `<repo>[:quant]`, or an https URL) | `./start.sh --admin pull-model --spec <repo>:Q4_K_M` |
| Copy a GGUF file you already have | `./start.sh --admin import-model --path /path/model.gguf` |
| Will it fit in memory with this context? | `./start.sh --admin estimate-model --name <file> --ctx 32768` |
| How fast is it here? (temporary engine, one question) | `./start.sh --admin benchmark-model --name <file>` |
| Use it | `./start.sh --config MODEL_FILE=<file>`, then restart |
| Delete one | `./start.sh --admin delete-model --name <file>` |

**Several models at once.** `./start.sh --config LLAMA_ROUTER_MODE=true`: every model of
`models/` can then answer, loaded on demand (at most `LLAMA_MODELS_MAX` at once). Which one
answers a message, first match wins:

1. the user's choice for one message: `/model <name> <message>` (`/model` alone lists them);
2. the model set on the agent: `./start.sh --admin update-agent --agent-id N --model <name>`;
3. for an agent that has tools, the tool model: `set-routing --tool-model <name>`;
4. the routing rules, on the message's length or its first word, then the default model:
   `get-routing` / `set-routing --default-model <name> --rules '[...]'`. `set-routing`
   replaces the whole table: give every field you want to keep.

A model the engine does not have falls back to the default model, and the log says so.

## Tools (MCP)

The model reaches the outside world only through tools, served by MCP servers (MCP: a standard
protocol to plug tools into a model). Six servers come with the application:

| Server | Tools | Needs |
|---|---|---|
| `time` | `get_time` | nothing |
| `calc` | `calculate`, `convert` (exact arithmetic, dates, units) | nothing |
| `web` | `fetch_page` (read one web page) | `WEB_FETCH_ALLOWED_HOSTS` (empty: every page refused; `*`: any public host) |
| `feeds` | `read_feed` (RSS and Atom) | the feed hosts in `WEB_FETCH_ALLOWED_HOSTS`; `create-feed` lists the feeds `/newagent` may propose |
| `notes` | `list_notes`, `read_note`, `search_notes`, `write_note` | `NOTES_DIR`, a folder of Markdown notes |
| `search` | `web_search` | a [SearXNG](https://docs.searxng.org/) instance: `./start.sh --config SEARXNG_MANAGED=true` makes `start.sh` run one in Docker and set `SEARXNG_URL`; or `SEARXNG_URL` to your own (with `json` in `search.formats` of its `settings.yml`) |

No tool is on by default. One command gives a built-in server to a user and adds its tools to one
of their agents, here web search and page reading for user 1 and agent 1:

```bash
./start.sh --config SEARXNG_MANAGED=true      # then restart: start.sh runs SearXNG in Docker
./start.sh --admin enable-builtin --builtin-id search --user-id 1 --agent-id 1
./start.sh --admin enable-builtin --builtin-id web --user-id 1 --agent-id 1
```

`enable-builtin` declares the server when needed, enables it, approves its new tool definitions,
grants it to every agent of the user and adds its tools to the agent. It adds and never replaces,
and a second call changes nothing. Its answer names the required setting that is still empty
(`missing_setting`) with the `./start.sh --config` command that sets it.

The same steps one by one, for a server that is not built in or for finer grants:
`create-server`, `approve-definitions`, `replace-grants`, `update-agent --tools`.

- `approve-definitions` pins what the tools say they do: if a tool's description changes later,
  the tool is off until approved again.
- `replace-grants` replaces every grant of that user, and `update-agent --tools` the agent's
  list: give the whole list. A grant without `agent_id` covers all of the user's agents, and only
  those tools are proposed by `/newagent`.
- A tool whose definition changed since it was approved is refused by `enable-builtin`: read the
  change with `list-tools`, then `approve-definitions`.
- A tool that writes or sends data out (`write_note`, `web_search`) asks the user "yes" first;
  `toggle-tool --policy allow|confirm|deny` changes that per tool.
- `list-calls` shows every tool call, refused ones included. Another MCP server runs in its own
  container: `deploy-sidecar` (owner), then `create-server --protocol http --url ...`.

## Everyday commands

| Command | Does |
|---|---|
| `./start.sh --native` | Start (Ctrl+C stops it) |
| `./start.sh --status` / `--stop` | What runs / stop it (`--stop --all` also stops the model) |
| `./start.sh --config` / `--config KEY=VALUE` | Read / change the settings |
| `./start.sh --config --edit` | Change the settings by answering one question per setting (the previous `.env` is kept) |
| `./start.sh --admin` | Menu over every administration command |
| `./start.sh --admin COMMAND` | One administration command (`--admin describe` lists them all) |
| `./start.sh --chat [--agent NAME]` | Talk to one of your agents from the terminal, through the API (needs a terminal identity: `--admin add-channel-identity --user-id N --channel terminal --identifier owner`) |
| `./start.sh --admin list-models` / `pull-model --spec <repo>` | Installed models / download one (see "Models") |
| `./start.sh --restore` / `--rekey` | Restore a backup / change the encryption key |

## Commands users type (Telegram)

| Command | Does |
|---|---|
| `/agent`, `/agent <name>` | List your agents / talk to another one |
| `/new`, `/export` | Start a fresh conversation / get yours as a file |
| `/task add daily 08:30 <prompt>` | Run a prompt on a schedule (also `every 2h`, `cron 0 8 * * 1-5`, or in words) |
| `/newagent <description>`, `/myagents`, `/editagent` | Create, list, change your agents |
| `/model`, `/prompt` | Pick a model for one message / use a saved prompt |

## Logs

- In the terminal when you start with `./start.sh --native`, and always also in
  `logs/channelagent.log` (rotated at 10 MB, 5 old files kept, readable by you only).
- Secrets (bot token, keys, passwords) are masked in both.
- Started in the background (`--native --detach`)? Follow it with `tail -f logs/channelagent.log`.
- Another file or none: `./start.sh --config LOG_FILE=...` (empty = terminal only).

## Settings

Every setting is in `.env` (list, defaults and explanations: `.env.example`).
`./start.sh --config --edit` asks for each one in turn (secrets typed without echo, never shown),
checks every answer and writes once, after a confirmation. `./start.sh --config KEY=VALUE`
checks the value against the rules of `app/settings_rules.py` before writing it, puts a value
with spaces or special characters between single quotes, and first keeps the previous file as
`.env.bak-YYYYMMDD-HHMMSS` (mode 600, every generation kept, deleting old ones is up to you; if
the copy fails nothing is written). A change applies at the next start.

## Security in short

- Who asked what and who did what: one timeline (UI *Audit trail*,
  `./start.sh --admin audit-timeline`), strangers' messages and every API request included.
- Local only: the API listens on `127.0.0.1`; remote administration through HTTPS or an SSH tunnel.
- Conversations and personal fields encrypted at rest; the key lives only in `.env`.
- Every administration command needs a key and a permission level; every change is logged.
- Every request to the internet goes through one guard (HTTPS, allowed hosts, public addresses only).
- A tool call carrying a secret, an internal address, a path escape or a shell injection is
  blocked; a tool result that gives the assistant orders is flagged; after 3 in a day the user's
  tools are suspended (`./start.sh --admin resume-tools --user-id N` lifts it). You get a message
  each time.

## Development

- Tests: `pytest`; lint: `ruff check .`.
- Dependencies are locked: edit `requirements.in`, run `scripts/update_requirements.sh --keep`,
  then `pip-audit -r requirements.txt`.
- Database changes: Alembic migrations, applied at start after a verified backup.
- Secret scan: `scripts/secret_scan.sh` runs gitleaks over the whole git history
  (`--dir PATH` scans a folder; it reads ignored files too, so never point it at a checkout
  holding `.env` or `data/`). gitleaks is in the virtualenv, `.venv/bin/gitleaks`: a pinned
  release checked against its SHA-256, installed by `./start.sh --native` or by
  `scripts/install_gitleaks.sh`.
- Every API route needs a token and declares a scope: `python scripts/api_auth_matrix.py`
  calls each one without a token, with a wrong one, with an invalid body, one scope below and
  with its own scope, and exits 1 on any gap.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
