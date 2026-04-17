from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/")
@router.head("/")
def health():
    return {"status": "ok"}


@router.get("/routes")
def list_routes(request: Request):
    """Debug: list all registered routes."""
    routes = []
    for route in request.app.routes:
        if hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": sorted(getattr(route, "methods", None) or []),
            })
    return {"routes": routes}
