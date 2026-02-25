"""Role registry for discovering and instantiating roles."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wolf.roles.base import RoleBase


class RoleRegistry:
    """Central registry of all available roles."""

    _roles: dict[str, type[RoleBase]] = {}

    @classmethod
    def register(cls, role_class: type[RoleBase]) -> type[RoleBase]:
        """Decorator that registers a role class.

        Usage::

            @RoleRegistry.register
            class Villager(RoleBase):
                ...
        """
        instance = role_class()
        cls._roles[instance.name] = role_class
        return role_class

    @classmethod
    def get(cls, role_name: str) -> RoleBase:
        """Return a new instance of the named role."""
        if role_name not in cls._roles:
            raise KeyError(f"Unknown role: {role_name!r}")
        return cls._roles[role_name]()

    @classmethod
    def get_all(cls) -> dict[str, RoleBase]:
        """Return instances of every registered role keyed by name."""
        return {name: role_cls() for name, role_cls in cls._roles.items()}

    @classmethod
    def discover_plugins(cls) -> None:
        """Auto-import all submodules under wolf.roles.classic."""
        import wolf.roles.classic as classic_pkg

        for importer, modname, ispkg in pkgutil.iter_modules(classic_pkg.__path__):
            importlib.import_module(f"wolf.roles.classic.{modname}")


# Auto-discover classic roles on import.
RoleRegistry.discover_plugins()
