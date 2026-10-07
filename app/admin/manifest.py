"""The manifest of the Admin API: every operation with its command name, method,
path, scope and typed parameters, built from the routes and their OpenAPI schema. The
script's command list and the UI's forms are generated from it, so a new route appears in
both without client code, and a route that cannot be described fails the contract test.
"""

from fastapi import FastAPI
from fastapi.routing import APIRoute

from app.api.scopes import _api_routes, declared_scopes
from app.api.version import API_VERSION

_SKIPPED_HEADERS = {"authorization", "x-client", "x-totp"}


def _resolve(schema: dict, spec: dict) -> dict:
    while "$ref" in schema:
        node = spec
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        schema = node
    return schema


def _describe_schema(schema: dict, spec: dict) -> dict:
    """type, enum and default of a parameter or field, nullable unions reduced to one type."""
    schema = _resolve(schema, spec)
    nullable = False
    if "anyOf" in schema:
        options = [o for o in schema["anyOf"] if o.get("type") != "null"]
        nullable = len(options) < len(schema["anyOf"])
        merged = _resolve(options[0], spec) if options else {}
        schema = {**merged, "default": schema.get("default")}
    return {
        "type": schema.get("type", "string"),
        "enum": schema.get("enum"),
        "default": schema.get("default"),
        # The field takes null (`agent_id: null` = the default agent): typed as `null`.
        "nullable": nullable,
        # A secret (`format: password`): never on a command line, masked in the UI.
        "secret": schema.get("format") == "password",
        # The type of a list's items: a list of texts is typed one per line in the UI.
        "items": _resolve(schema.get("items") or {}, spec).get("type") if schema.get("type")
        == "array" else None,
    }


def _fields(route: APIRoute, operation: dict, spec: dict) -> list[dict]:
    fields = []
    for p in operation.get("parameters", []):
        if p["in"] == "header" and p["name"].lower() in _SKIPPED_HEADERS:
            continue
        fields.append(
            {
                "name": p["name"],
                "in": p["in"],
                "required": bool(p.get("required")),
                "description": p.get("description", ""),
                **_describe_schema(p.get("schema", {}), spec),
            }
        )
    body = operation.get("requestBody", {}).get("content", {}).get("application/json")
    if body:
        schema = _resolve(body["schema"], spec)
        required = set(schema.get("required", []))
        for name, prop in schema.get("properties", {}).items():
            fields.append(
                {
                    "name": name,
                    "in": "body",
                    "required": name in required,
                    "description": _resolve(prop, spec).get("description", ""),
                    **_describe_schema(prop, spec),
                }
            )
    return fields


def operations(app: FastAPI) -> list[dict]:
    spec = app.openapi()
    result = []
    for route in _api_routes(app.routes):
        if not isinstance(route, APIRoute):
            continue
        method = sorted(route.methods - {"HEAD", "OPTIONS"})[0]
        operation = next(
            op for op in spec["paths"][route.path].values() if op["operationId"] == route.unique_id
        )
        result.append(
            {
                "command": route.name.replace("_", "-"),
                "method": method,
                "path": route.path,
                "scope": declared_scopes(route)[0].name.lower(),
                "tag": (route.tags or [""])[0],
                "description": (route.endpoint.__doc__ or "").strip().split("\n")[0],
                "fields": _fields(route, operation, spec),
                "returns_job": route.status_code == 202,
            }
        )
    return sorted(result, key=lambda o: (o["tag"], o["command"]))


def manifest(app: FastAPI) -> dict:
    ops = operations(app)
    return {"api_version": API_VERSION, "count": len(ops), "operations": ops}


def contract_problems(app: FastAPI) -> list[str]:
    """What stops a route from being a usable command and a usable form: no help text, a
    command name used twice, a path parameter with no field. Empty when the contract holds.
    """
    problems = []
    seen: dict[str, str] = {}
    for op in operations(app):
        where = f"{op['method']} {op['path']}"
        if not op["description"]:
            problems.append(f"{where}: no description (docstring of the route)")
        if op["command"] in seen:
            first = seen[op["command"]]
            problems.append(f"{where}: command {op['command']!r} already used by {first}")
        seen[op["command"]] = where
        names = {f["name"] for f in op["fields"] if f["in"] == "path"}
        for part in [p[1:-1] for p in op["path"].split("/") if p.startswith("{")]:
            if part not in names:
                problems.append(f"{where}: path parameter {part!r} has no field")
    return problems
