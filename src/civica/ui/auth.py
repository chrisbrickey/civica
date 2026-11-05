"""Browser sign-in state: auth middleware and helpers over NiceGUI's per-user storage."""

from fastapi.responses import RedirectResponse
from nicegui import app
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from civica.domain.user import UserId

HOME_PATH = "/"
LOGIN_PATH = "/login"
REGISTER_PATH = "/register"
_NICEGUI_ASSETS_PREFIX = "/_nicegui/"


class SessionUser(BaseModel):  # type: ignore[explicit-any]
    """The signed-in user, read back from per-user storage (persisted JSON)."""

    model_config = ConfigDict(frozen=True)

    user_id: UserId
    username: str


def sign_in(user_id: UserId, username: str) -> None:
    app.storage.user.update(user_id=str(user_id), username=username)


def current_user() -> SessionUser | None:
    try:
        return SessionUser.model_validate(dict(app.storage.user))
    except ValidationError:
        return None


def sign_out() -> None:
    """Clear all per-user session storage, which is data belonging to browser.

    No impact to user account or coaching memory."""
    app.storage.user.clear()


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect to the login page unless the browser is signed in or the path is public."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        is_public = path in {LOGIN_PATH, REGISTER_PATH} or path.startswith(_NICEGUI_ASSETS_PREFIX)
        if not is_public and current_user() is None:
            return RedirectResponse(LOGIN_PATH)
        return await call_next(request)


def install_auth_middleware() -> None:
    """Add AuthMiddleware as the innermost layer so it runs after NiceGUI's storage middleware.

    Appending (rather than app.add_middleware) gives the same order whether called before or after ui.run.
    """
    app.user_middleware.append(Middleware(AuthMiddleware))
