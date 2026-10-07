"""The version of the Admin API contract. It is the OpenAPI `info.version` and
what `/whoami` and `/status` report. Bump it when a route, a field or a status
changes in a way an existing client would notice; the script and the UI compare it.
"""

API_VERSION = "1"
