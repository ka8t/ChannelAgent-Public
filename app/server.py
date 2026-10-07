"""What the application serves over HTTP: the admin UI under `/ui`, the Admin API
everywhere else. Two separate applications: the API keeps its bearer key on every route
(the UI's pages are not API routes), the UI its session, CSRF and headers.
"""

from app.api.app import app as api_app
from app.ui.app import ui_app

ICON = "/ui/static/favicon.svg"


class Root:
    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if scope["type"] == "http" and (path == "/ui" or path.startswith("/ui/")):
            await ui_app(scope, receive, send)
        elif scope["type"] == "http" and path == "/" and scope.get("method") in ("GET", "HEAD"):
            # A browser opened on the bare address gets the UI (its sign-in page when there is
            # no session), not the API's 401. Only GET and HEAD of "/" itself: the API keeps no
            # public route, and every other request to "/" still needs the key.
            await ui_app({**scope, "path": "/ui", "raw_path": b"/ui"}, receive, send)
        elif scope["type"] == "http" and path == "/favicon.ico" and scope.get("method") in (
            "GET",
            "HEAD",
        ):
            # A browser asks for it by itself: sent to the API it was a failed sign-in,
            # and ten in a minute blocked the address, the UI's own API calls included (429).
            await ui_app({**scope, "path": ICON, "raw_path": ICON.encode()}, receive, send)
        else:
            await api_app(scope, receive, send)


root = Root()
