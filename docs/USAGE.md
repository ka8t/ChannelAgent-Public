# Usage

Every command of `start.sh`, every administration command and every command a user types.
The setup is in the [README](../README.md); every setting is explained in `.env.example`.
This file is generated from the code.

## start.sh

```
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
```

## Commands users type (Telegram, terminal)

| Command | Does |
|---|---|
| `/agent, /agent <name>` | List your agents, or talk to another one |
| `/new` | Start a fresh conversation with the current agent |
| `/export` | Get the current conversation as a file |
| `/task` | List your scheduled tasks |
| `/task add daily 08:30 <prompt>` | Run a prompt every day (also `every 2h`, `cron 0 8 * * 1-5`, or in words) |
| `/task pause|resume|delete|run <id>` | Manage one task |
| `/task tz <zone>` | Set your time zone (for example `Europe/Paris`) |
| `/newagent <description>` | Create an agent by describing it; `/cancel` stops the dialogue |
| `/myagents, /editagent <name> <change>` | List your agents, change one |
| `/model, /model <name> <message>` | List the models, or answer one message with another model |
| `/prompt <name> key=value ...` | Use a saved prompt |

In the terminal, `./start.sh --chat` accepts `/agent`, `/model`, `/prompt`, `/task`, `/new`
and `/export`. By e-mail, write a message whose subject contains the trigger tag (default
`[agent]`); the reply comes back by e-mail.

## Administration commands

`./start.sh --admin COMMAND [flags]` runs one of the 130 commands below;
`./start.sh --admin` alone shows them as menus, and the admin UI has one page per command.
Each command is one call to the Admin API and needs at least the scope shown
(`read` < `operate` < `admin` < `owner`).

Flags common to every command:

- `--json`: print the response as JSON.
- `--transport auto|http|inprocess`: reach the running application over HTTP (`API_URL`, or
  this machine), or run the same handlers in process when the application is stopped.
- `--no-wait`: for a long operation (a job), print the started job instead of waiting.

`./start.sh --admin describe` prints this list; `./start.sh --admin COMMAND --help` prints
one command's flags. A flag in brackets is optional; a secret is asked without echo (`-`
reads it from standard input).

### administrators

| Command | Scope | Flags | Does |
|---|---|---|---|
| `create-admin` | owner | `--name STRING` `--scope STRING` `--password STRING` (secret, asked) [`--with-totp BOOLEAN`, default false] | Create a named administrator; an owner's TOTP secret is shown once. |
| `disable-admin` | owner | `--name STRING` | Disable an administrator and revoke their tokens (never the last owner). |
| `list-admins` | owner | none | Every named administrator, disabled ones included. |
| `list-tokens` | read | [`--every-account BOOLEAN`, default false] | The caller's tokens (never the token itself), or every account's for an owner. |
| `reset-admin-totp` | owner | `--name STRING` | Give an administrator a new TOTP secret, shown once (a lost phone). |
| `revoke-token` | read | `--token-id INTEGER` | Revoke a token: one of the caller's, or any for an owner. |
| `set-admin-password` | read | `--name STRING` `--password STRING` (secret, asked) | Change a password: one's own, or anyone's for an owner. |
| `sign-in` | read | [`--label STRING`, default cli] [`--scope STRING`] [`--hours INTEGER`, default 720] | Sign in: name and password in a Basic header, the code in X-TOTP; answers a token. |

### agent-templates

| Command | Scope | Flags | Does |
|---|---|---|---|
| `create-agent-template` | admin | `--description STRING` [`--guidance STRING`] [`--agent-instructions STRING`] [`--memory-mode STRING`] [`--tools ARRAY`] [`--needs-schedule BOOLEAN`] [`--questions OBJECT`] [`--enabled BOOLEAN`] `--name STRING` | Create an agent template (version 1). 409 when the name is taken. |
| `delete-agent-template` | admin | `--name STRING` | Delete an agent template and its versions. Agents made from it are not changed. |
| `get-agent-template` | read | `--name STRING` | One agent template. |
| `install-starter-templates` | admin | none | Create the starter templates that are missing (news digest, web page watch, reminder, ... |
| `list-agent-template-versions` | read | `--name STRING` | Every version of an agent template, oldest first, each the whole template. |
| `list-agent-templates` | read | none | Every agent template, by name. |
| `update-agent-template` | admin | `--name STRING` [`--description STRING`] [`--guidance STRING`] [`--agent-instructions STRING`] [`--memory-mode STRING`] [`--tools ARRAY`] [`--needs-schedule BOOLEAN`] [`--questions OBJECT`] [`--enabled BOOLEAN`] | Change an agent template; a real change is a new version, the old one is kept. |

### agents

| Command | Scope | Flags | Does |
|---|---|---|---|
| `agent-overview` | admin | `--user-id INTEGER` | A user's agents with their purpose and their tasks' schedule, next and last run. |
| `check-agent-spec` | admin | `--user-id INTEGER` `--name STRING` [`--purpose STRING`] [`--system-prompt STRING`] [`--model STRING`] [`--memory-mode off|ondemand|always|search`, default off] [`--tools ARRAY`] [`--skills ARRAY`] [`--task-prompt STRING`] [`--schedule-kind STRING`] [`--schedule-expr STRING`] [`--delivery-identity-id INTEGER`] [`--unattended ARRAY`] | Check an agent specification for a user, without creating anything. |
| `create-agent` | operate | `--user-id INTEGER` `--name STRING` [`--system-prompt STRING`] [`--model STRING`] [`--memory-mode off|ondemand|always|search`, default off] [`--tools ARRAY`] | Create an agent for a user, optionally with its system prompt, model, memory mode, tools. |
| `create-agent-from-spec` | admin | `--user-id INTEGER` `--name STRING` [`--purpose STRING`] [`--system-prompt STRING`] [`--model STRING`] [`--memory-mode off|ondemand|always|search`, default off] [`--tools ARRAY`] [`--skills ARRAY`] [`--task-prompt STRING`] [`--schedule-kind STRING`] [`--schedule-expr STRING`] [`--delivery-identity-id INTEGER`] [`--unattended ARRAY`] | Create an agent and its scheduled task, within the rights the user already has. |
| `get-agent` | read | `--agent-id INTEGER` | Show one agent. |
| `get-agent-spec` | admin | `--user-id INTEGER` `--agent-id INTEGER` | The specification of one of a user's agents and of its first task. |
| `list-agents` | read | `--user-id INTEGER` [`--limit INTEGER`, default 200] [`--offset INTEGER`, default 0] | List a user's agents. |
| `pause-agent` | operate | `--user-id INTEGER` `--agent-id INTEGER` | Stop the scheduled tasks of one of a user's agents (it still answers in chat). |
| `resume-agent` | operate | `--user-id INTEGER` `--agent-id INTEGER` | Start again the scheduled tasks of one of a user's agents. |
| `retire-agent` | operate | `--user-id INTEGER` `--agent-id INTEGER` | Delete the tasks of one of a user's agents and disable it (kept for the audit trail). |
| `run-agent` | admin | `--user-id INTEGER` `--agent-id INTEGER` | Run every task of one of a user's agents now, whatever its schedule. Returns the job. (job) |
| `update-agent` | operate | `--agent-id INTEGER` [`--name STRING`] [`--is-active BOOLEAN`] [`--system-prompt STRING`] [`--model STRING`] [`--memory-mode off|ondemand|always|search`] [`--tools ARRAY`] | Rename, activate or deactivate, or set the prompt, model, memory mode and tools of any ... |
| `update-agent-spec` | admin | `--user-id INTEGER` `--agent-id INTEGER` `--name STRING` [`--purpose STRING`] [`--system-prompt STRING`] [`--model STRING`] [`--memory-mode off|ondemand|always|search`, default off] [`--tools ARRAY`] [`--skills ARRAY`] [`--task-prompt STRING`] [`--schedule-kind STRING`] [`--schedule-expr STRING`] [`--delivery-identity-id INTEGER`] [`--unattended ARRAY`] | Change one of a user's agents and its task to a specification, within the user's rights. |

### audit

| Command | Scope | Flags | Does |
|---|---|---|---|
| `audit-timeline` | admin | [`--user-id INTEGER`] [`--actor STRING`] [`--channel telegram|email|matrix|terminal`] [`--kinds ARRAY`] [`--since STRING`] [`--until STRING`] [`--with-text BOOLEAN`, default false] [`--limit INTEGER`, default 100] [`--offset INTEGER`, default 0] | Who asked what and who did what, newest first: messages, messages of unknown senders, ... |
| `search-admin-events` | admin | [`--actor STRING`] [`--action STRING`] [`--target-type STRING`] [`--target-id INTEGER`] [`--since STRING`] [`--until STRING`] [`--limit INTEGER`, default 100] [`--offset INTEGER`, default 0] | Search what administrators did (the admin events). |
| `search-logs` | admin | [`--user-id INTEGER`] [`--agent-id INTEGER`] [`--channel telegram|email|matrix|terminal`] [`--direction inbound|outbound`] [`--status ok|failed|denied|limited`] [`--since STRING`] [`--until STRING`] [`--keyword STRING`] [`--limit INTEGER`, default 100] [`--offset INTEGER`, default 0] | Search the action log (conversation text) by user, agent, channel, status, date, keyword. |
| `verify-audit` | admin | none | Check the chain of the admin events: each changed or missing event is reported. |

### backups

| Command | Scope | Flags | Does |
|---|---|---|---|
| `create-database-backup` | admin | none | Copy and verify the database into `backups/`. Returns the job doing it. (job) |
| `export-backup` | owner | `--password STRING` (secret, asked) | Write the database and the conversations into one file of `backups/`, encrypted with a ... (job) |
| `get-backup-schedule` | read | none | The scheduled backup: enabled, interval, how many are kept, the last run and ... |
| `import-backup` | owner | `--name STRING` `--password STRING` (secret, asked) | Decrypt an export of `backups/` into a new directory `data/imports/import-<stamp>/`, ... (job) |
| `list-database-backups` | admin | none | The backups of the database in `backups/`, newest first. |
| `restore-backup` | owner | `--name STRING` [`--allow-unreadable BOOLEAN`, default false] | Restore a listed backup: the helper stops the application, runs the checked restore ... (job) |
| `run-backup-schedule-now` | admin | none | Run the scheduled backup now (database and checkpoints, verified, rotated). Returns ... (job) |
| `set-backup-schedule` | admin | [`--enabled BOOLEAN`] [`--interval-minutes INTEGER`] [`--keep INTEGER`] | Change the scheduled backup: enabled, `interval_minutes` (1 to 10080), `keep` (1 ... |

### channels

| Command | Scope | Flags | Does |
|---|---|---|---|
| `add-channel-identity` | operate | `--user-id INTEGER` `--channel telegram|email|matrix|terminal` `--identifier STRING` | Add a channel identity (a Telegram id, or an email address) to a user. |
| `delete-channel-identity` | operate | `--user-id INTEGER` `--channel-identity-id INTEGER` | Remove a channel identity from a user. |
| `list-channel-identities` | read | `--user-id INTEGER` [`--limit INTEGER`, default 200] [`--offset INTEGER`, default 0] | List a user's channel identities. |
| `set-identity-agent` | operate | `--user-id INTEGER` `--channel-identity-id INTEGER` `--agent-id INTEGER` | Choose which of the user's agents this channel identity talks to. |

### chat

| Command | Scope | Flags | Does |
|---|---|---|---|
| `answer-chat` | operate | `--job-id STRING` `--answer BOOLEAN` | Answer the question a tool asks in this chat job (409 when none is waiting). |
| `chat` | operate | `--text STRING` | Send a message to the agent of the terminal identity of this API account; the turn runs ... (job) |
| `get-chat` | operate | `--job-id STRING` | One chat job: running (with a `question` in its result while a tool waits for a yes), ... |

### config

| Command | Scope | Flags | Does |
|---|---|---|---|
| `get-config` | admin | none | Every variable of `.env.example` with its value in `.env`; a secret has no value. |
| `set-config` | owner | `--key STRING` `--value STRING` | Set one variable after the same checks as `./start.sh --config KEY=VALUE`. It applies at ... |

### conversations

| Command | Scope | Flags | Does |
|---|---|---|---|
| `export-conversation` | admin | `--user-id INTEGER` [`--format json|markdown`, default json] [`--agent-id INTEGER`] [`--include-text BOOLEAN`, default false] | Export one user's conversation (JSON or Markdown). The text is included only when ... |
| `reset-conversation` | operate | `--user-id INTEGER` [`--agent-id INTEGER`] | Forget the stored history of a conversation that cannot be read. The ... |

### feeds

| Command | Scope | Flags | Does |
|---|---|---|---|
| `create-feed` | admin | `--name STRING` `--url STRING` [`--topics STRING`, default ] | List a feed for the agent builder (an https address; 409 when the name is taken). |
| `delete-feed` | admin | `--feed-id INTEGER` | Remove a feed from the builder's list. |
| `list-feeds` | read | none | The feeds the agent builder may propose, by name. |

### host

| Command | Scope | Flags | Does |
|---|---|---|---|
| `host-audit` | admin | [`--limit INTEGER`, default 50] | The host helper's audit lines, newest first: before and after each call, and refusals. |
| `host-rekey` | owner | [`--allow-unreadable BOOLEAN`, default false] [`--dry-run BOOLEAN`, default false] | Rotate ENCRYPTION_KEY with the application stopped (the guided sequence of ... (job) |
| `host-restart` | owner | none | Stop and start the application again, for example after a configuration change. (job) |
| `host-start` | owner | none | Start the application in the mode start.sh last used (409 when it already runs). From ... (job) |
| `host-status` | admin | none | Whether the application runs on the host, in which mode, as the host helper sees it. |
| `host-stop` | owner | none | Stop the application (native process or container). Returns the helper's job; the API ... (job) |

### jobs

| Command | Scope | Flags | Does |
|---|---|---|---|
| `cancel-job` | admin | `--job-id STRING` | Ask a running job to stop. A finished job cannot be cancelled (409), nor a stop, ... |
| `get-job` | admin | `--job-id STRING` | One job: status, progress, result or error. A `host-...` job is the host helper's. |
| `list-jobs` | admin | none | The recent jobs, newest first. |

### mcp

| Command | Scope | Flags | Does |
|---|---|---|---|
| `approve-definitions` | admin | `--server-id INTEGER` [`--tools ARRAY`] | Pin the definitions the server offers right now (all, or the named `tools`): ... |
| `create-server` | admin | `--name STRING` `--protocol stdio|http` [`--builtin-id STRING`] [`--url STRING`] [`--env-vars OBJECT`] [`--egress local|lan|internet`, default local] [`--enabled BOOLEAN`, default true] [`--timeout-seconds INTEGER`, default 20] [`--concurrency-limit INTEGER`, default 2] [`--result-max-bytes INTEGER`, default 1000000] [`--tool-policies OBJECT`] [`--shared-credentials BOOLEAN`, default false] [`--confirm-timeout-seconds INTEGER`, default 120] | Declare a server: `stdio` names a vetted built-in, `http` an exact URL. Its ... |
| `delete-server` | admin | `--server-id INTEGER` | Remove a declared server; already-recorded McpCall rows are unaffected. |
| `deploy-sidecar` | owner | `--name STRING` `--image STRING` [`--port INTEGER`, default 8000] [`--egress local|lan|internet`, default local] [`--memory-mb INTEGER`, default 256] [`--cpus NUMBER`, default 0.5] | Run a third-party MCP server in its own container (an image pinned in ... (job) |
| `enable-builtin` | admin | `--builtin-id STRING` `--user-id INTEGER` [`--agent-id INTEGER`] | Turn a built-in server (time, calc, web, feeds, notes, search) on for one user in one ... |
| `get-agent-exposure` | read | `--agent-id INTEGER` | What the agent's MCP tools and memory can do together: private data access, ... |
| `get-server` | read | `--server-id INTEGER` | One declared server by id. |
| `list-calls` | admin | [`--limit INTEGER`, default 50] [`--offset INTEGER`, default 0] [`--user-id INTEGER`] [`--server-name STRING`] [`--decision STRING`] | Every tool call, newest first, refused ones included, with the decision taken ... |
| `list-grants` | read | [`--user-id INTEGER`] | Who may use which MCP server or tool, through which agent (null: every agent ... |
| `list-servers` | read | none | Every declared MCP server, enabled or not. |
| `list-sidecars` | admin | none | The sidecar containers of third-party MCP servers, with their state and port. |
| `list-tools` | read | `--server-id INTEGER` | A server's tools right now, connecting to it live: whether an administrator ... |
| `remove-sidecar` | owner | `--name STRING` | Stop and remove a sidecar, its forwarder and its network. |
| `replace-grants` | admin | `--user-id INTEGER` `--grants ARRAY` | Replace every grant of one user. A server flagged as holding a shared ... |
| `test-server` | admin | `--server-id INTEGER` | Connects once, lists the tools, disconnects — never joins the shared ... |
| `toggle-tool` | admin | `--server-id INTEGER` `--tool STRING` [`--enabled BOOLEAN`] [`--policy allow|confirm|deny|default`] [`--classes OBJECT`] | Enable or disable one of a server's tools, set its policy (`allow`, ... |
| `update-server` | admin | `--server-id INTEGER` [`--url STRING`] [`--env-vars OBJECT`] [`--egress local|lan|internet`] [`--enabled BOOLEAN`] [`--timeout-seconds INTEGER`] [`--concurrency-limit INTEGER`] [`--result-max-bytes INTEGER`] [`--disabled-tools ARRAY`] [`--tool-policies OBJECT`] [`--shared-credentials BOOLEAN`] [`--confirm-timeout-seconds INTEGER`] | Change one or more settings of a declared server; a field left out keeps ... |

### memory

| Command | Scope | Flags | Does |
|---|---|---|---|
| `add-memory` | admin | `--user-id INTEGER` `--agent-id INTEGER` `--title STRING` `--content STRING` | Add an entry to an agent's memory. |
| `delete-memory` | admin | `--user-id INTEGER` `--agent-id INTEGER` `--entry-id INTEGER` | Delete one memory entry. |
| `edit-memory` | admin | `--user-id INTEGER` `--agent-id INTEGER` `--entry-id INTEGER` [`--title STRING`] [`--content STRING`] | Change the title or the content of one memory entry. |
| `list-memory` | admin | `--user-id INTEGER` `--agent-id INTEGER` [`--query STRING`] [`--limit INTEGER`, default 200] [`--offset INTEGER`, default 0] | An agent's memory entries, newest first, or the ones matching `query`. |

### models

| Command | Scope | Flags | Does |
|---|---|---|---|
| `benchmark-model` | admin | `--name STRING` [`--ctx INTEGER`] [`--slots INTEGER`, default 4] | Run the model in a temporary engine, ask one question and report its real speed ... (job) |
| `delete-model` | admin | `--name STRING` | Delete a model file and its recorded hash; the loaded or configured model is refused. |
| `estimate-model` | read | `--name STRING` [`--ctx INTEGER`] [`--slots INTEGER`, default 4] [`--cache-type STRING`, default q8_0] [`--router-models INTEGER`, default 1] [`--projector STRING`] | The memory this model needs with this context, line by line with what each line is, ... |
| `import-model` | admin | `--path STRING` [`--name STRING`] [`--force BOOLEAN`, default false] | Copy a local GGUF file into the models directory. Returns the job doing it. (job) |
| `list-models` | read | none | The models in the models directory, and which one the running engine has loaded. |
| `pull-model` | admin | `--spec STRING` [`--name STRING`] [`--sha256 STRING`] [`--force BOOLEAN`, default false] | Download a model from the hub (<repo>[:quant]) or an https URL. Returns the job. (job) |

### permissions

| Command | Scope | Flags | Does |
|---|---|---|---|
| `grant` | admin | `--user-id INTEGER` `--channel-identity-id INTEGER` `--kind chat|admin` | Grant a permission to a channel identity. |
| `list-permissions` | read | `--user-id INTEGER` `--channel-identity-id INTEGER` | List the permissions of a channel identity. |
| `revoke` | admin | `--user-id INTEGER` `--channel-identity-id INTEGER` `--kind chat|admin` | Revoke a permission from a channel identity. |

### requests

| Command | Scope | Flags | Does |
|---|---|---|---|
| `approve-request` | operate | `--request-id INTEGER` | Approve an access request and create the user. |
| `deny-request` | operate | `--request-id INTEGER` | Deny an access request. |
| `list-requests` | read | [`--status STRING`, default pending] [`--limit INTEGER`, default 200] [`--offset INTEGER`, default 0] | List the access requests, by default the pending ones. |

### retention

| Command | Scope | Flags | Does |
|---|---|---|---|
| `get-retention` | admin | none | The retention periods (days; null = kept for ever, the default) and the last run. |
| `run-retention` | owner | [`--dry-run BOOLEAN`, default true] | Apply the retention: `dry_run` (the default) counts what would go per table; a real run ... (job) |
| `set-retention` | owner | [`--messages-days INTEGER`] [`--tool-calls-days INTEGER`] [`--api-calls-days INTEGER`] [`--conversations-days INTEGER`] | Set the retention periods: 1 to 3650 days, or null to keep for ever. A period left out ... |

### routing

| Command | Scope | Flags | Does |
|---|---|---|---|
| `get-routing` | read | none | The rules, in the order they are tried, and the default model a turn falls ... |
| `set-routing` | admin | [`--default-model STRING`] [`--rules ARRAY`] [`--model-ctx-sizes OBJECT`] [`--max-tools INTEGER`, default 5] [`--tool-rules ARRAY`] [`--tool-model STRING`] | Replace the whole routing table: the rules, in order, the default model, ... |

### skills

| Command | Scope | Flags | Does |
|---|---|---|---|
| `create-skill` | admin | `--name STRING` `--description STRING` `--body STRING` [`--tools ARRAY`] [`--self-service BOOLEAN`, default false] | Create a skill (version 1). 409 when the name is taken. |
| `delete-skill` | admin | `--name STRING` | Delete a skill, its versions, and its grant to every agent. |
| `get-skill` | read | `--name STRING` | One skill, with its body. |
| `import-skill` | admin | `--folder STRING` | Create or update (a new version) the skill of `SKILLS_DIR/<folder>/SKILL.md`. Its front ... |
| `list-skill-versions` | read | `--name STRING` | Every version of a skill's description and body, oldest first. |
| `list-skills` | read | none | Every skill, by name, without its body. |
| `set-agent-skills` | admin | `--agent-id INTEGER` `--skills ARRAY` | The skills this agent may load (replaces the list; [] grants none, the default). Only ... |
| `update-skill` | admin | `--name STRING` [`--description STRING`] [`--body STRING`] [`--tools ARRAY`] [`--self-service BOOLEAN`] | Change a skill; a new description or body is a new version, the old one is kept. |

### system

| Command | Scope | Flags | Does |
|---|---|---|---|
| `docs` | read | none | The interactive documentation page. |
| `openapi-json` | read | none | The OpenAPI description of this API. |
| `status` | read | none | Version, uptime, components, database revision and size, engine and its model. |
| `storage` | read | none | Show what the database and the checkpoints hold, and the rows that cannot be decrypted. |
| `whoami` | read | none | The actor and scope of the caller, and the API version. |

### tasks

| Command | Scope | Flags | Does |
|---|---|---|---|
| `create-task` | admin | `--user-id INTEGER` `--prompt STRING` `--kind STRING` `--expr STRING` [`--agent-id INTEGER`] [`--channel-identity-id INTEGER`] [`--enabled BOOLEAN`, default true] [`--standing-tools ARRAY`] | Create a scheduled task for a user. A schedule that does not parse, or never falls ... |
| `delete-task` | admin | `--task-id INTEGER` | Delete a task and its conversation. |
| `get-pause` | admin | none | Whether the scheduled tasks are paused (then none runs). |
| `get-task` | admin | `--task-id INTEGER` | One scheduled task. |
| `list-tasks` | admin | [`--user-id INTEGER`] [`--limit INTEGER`, default 200] [`--offset INTEGER`, default 0] | Every user's scheduled tasks, or one user's, oldest first. |
| `parse-schedule` | operate | `--text STRING` [`--timezone STRING`] | Read a schedule written in words ("every Thursday at 9") as cron, every or daily. |
| `run-task` | admin | `--task-id INTEGER` | Run a task now, whatever its schedule (its next time does not move), even while ... (job) |
| `set-pause` | admin | `--paused BOOLEAN` | Pause every scheduled task (`paused: true`), or resume them. While paused, a task ... |
| `update-task` | admin | `--task-id INTEGER` [`--prompt STRING`] [`--kind STRING`] [`--expr STRING`] [`--agent-id INTEGER`] [`--channel-identity-id INTEGER`] [`--enabled BOOLEAN`] [`--standing-tools ARRAY`] | Change a task: its prompt, schedule, agent, delivery identity, or stop it ... |

### telemetry

| Command | Scope | Flags | Does |
|---|---|---|---|
| `get-telemetry` | read | [`--since STRING`] [`--until STRING`] [`--group-by none|user|agent|model`, default none] | Messages, replies, failed turns, denied messages, reply latency p50 and p95 and tokens ... |

### users

| Command | Scope | Flags | Does |
|---|---|---|---|
| `create-user` | operate | [`--display-name STRING`] | Create a user. |
| `delete-user` | owner | `--user-id INTEGER` [`--purge BOOLEAN`, default false] | Delete a user; with purge, also their agents, audit trail and conversations. |
| `get-user` | read | `--user-id INTEGER` | Show one user. |
| `list-users` | read | [`--limit INTEGER`, default 200] [`--offset INTEGER`, default 0] | List the users. |
| `resume-tools` | admin | `--user-id INTEGER` | Resume the tools of a user the MCP guard suspended after repeated threats. |
| `update-user` | operate | `--user-id INTEGER` [`--display-name STRING`] [`--is-active BOOLEAN`] [`--timezone STRING`] | Rename a user, activate or deactivate them, or set their timezone (an IANA name, ... |

`sign-out` (no flag) revokes the saved token of this API and forgets it.
