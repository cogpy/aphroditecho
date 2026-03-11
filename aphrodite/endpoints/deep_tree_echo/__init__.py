"""
Deep Tree Echo FastAPI endpoints for server-side rendering.

This module provides FastAPI-based endpoints for Deep Tree Echo System Network (DTESN)
processing with server-side rendering capabilities integrated with the Aphrodite Engine.

Components:
- FastAPI application factory
- Server-side route handlers for DTESN processing
- Integration with echo.kern components
- Server-side template rendering
- Performance monitoring and caching
"""

__all__ = ["create_app", "router"]


def __getattr__(name):
    if name == "create_app":
        from aphrodite.endpoints.deep_tree_echo.app_factory import create_app
        return create_app
    if name == "router":
        from aphrodite.endpoints.deep_tree_echo.routes import router
        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")