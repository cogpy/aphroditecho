"""
Deep Tree Echo FastAPI endpoints for server-side rendering.

This module provides FastAPI-based endpoints for Deep Tree Echo System Network (DTESN)
processing with server-side rendering capabilities integrated with the Aphrodite Engine.

Components:
- FastAPI application factory
- Server-side route handlers for DTESN processing
- Dynamic configuration management with hot reload
- Integration with echo.kern components
- Server-side template rendering
- Performance monitoring and caching
"""

__all__ = [
    "create_app", 
    "router",
    "config_router",
    "DynamicConfigurationManager",
    "ConfigurationUpdateRequest", 
    "ConfigurationEnvironment",
    "get_dynamic_config_manager",
    "initialize_dynamic_config_manager"
]


def __getattr__(name):
    if name == "create_app":
        from aphrodite.endpoints.deep_tree_echo.app_factory import create_app
        return create_app
    if name == "router":
        from aphrodite.endpoints.deep_tree_echo.routes import router
        return router
    if name == "config_router":
        from aphrodite.endpoints.deep_tree_echo.config_routes import config_router
        return config_router
    if name in {
        "DynamicConfigurationManager",
        "ConfigurationUpdateRequest",
        "ConfigurationEnvironment",
        "get_dynamic_config_manager",
        "initialize_dynamic_config_manager",
    }:
        from aphrodite.endpoints.deep_tree_echo.dynamic_config_manager import (
            ConfigurationEnvironment,
            ConfigurationUpdateRequest,
            DynamicConfigurationManager,
            get_dynamic_config_manager,
            initialize_dynamic_config_manager,
        )
        return {
            "DynamicConfigurationManager": DynamicConfigurationManager,
            "ConfigurationUpdateRequest": ConfigurationUpdateRequest,
            "ConfigurationEnvironment": ConfigurationEnvironment,
            "get_dynamic_config_manager": get_dynamic_config_manager,
            "initialize_dynamic_config_manager": initialize_dynamic_config_manager,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
