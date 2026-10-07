"""The terminal chat client: `./start.sh --chat [--agent NAME]`.

Each line typed is sent with `POST /chat` and the job is followed until the agent answers; a
tool that asks for a yes is answered here ([y/N]). The commands of the other channels work as
they are (`/agent`, `/new`, `/newagent`, `/task`). `/quit` or the end of input stops.

It talks to the API like the script's other commands (app.admin.client.open_client): over
HTTP to the running application, which may be on another machine (API_URL), or in process
when nothing listens (`--transport`).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from app.admin.client import UsageError, open_client

QUIT = ("/quit", "/exit")
FINAL = ("done", "failed", "cancelled")


async def send(client: httpx.AsyncClient, text: str, read, write, poll: float) -> None:
    response = await client.post("/chat", json={"text": text})
    if response.status_code != 202:
        write(f"error {response.status_code}: {response.text[:300]}")
        return
    job = response.json()
    asked = None
    while job["status"] not in FINAL:
        question = (job.get("result") or {}).get("question")
        if question and question != asked:
            asked = question
            answer = read(f"{question} [y/N] ").strip().lower() in ("y", "yes", "o", "oui")
            await client.post(f"/chat/{job['id']}/answer", json={"answer": answer})
        await asyncio.sleep(poll)
        job = (await client.get(f"/chat/{job['id']}")).json()
    if job["status"] != "done":
        write(f"error: {job.get('error') or job['status']}")
        return
    for reply in (job.get("result") or {}).get("replies") or []:
        write(reply)


async def loop(
    client: httpx.AsyncClient, read=input, write=print, poll: float = 0.3, agent: str | None = None
) -> int:
    if agent:
        await send(client, f"/agent {agent}", read, write, poll)
    while True:
        try:
            text = read("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not text:
            continue
        if text.lower() in QUIT:
            return 0
        await send(client, text, read, write, poll)


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="./start.sh --chat", description=__doc__.split("\n")[0])
    parser.add_argument("--agent", help="talk to this agent of yours (the same as /agent NAME)")
    parser.add_argument("--transport", choices=("auto", "http", "inprocess"), default="auto")
    args = parser.parse_args(argv)
    try:
        async with open_client(args.transport) as client:
            print(f"Chat through the API ({client.transport_name}); /quit to stop.")
            return await loop(client, agent=args.agent)
    except UsageError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 2


def main() -> int:
    return asyncio.run(amain(sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
