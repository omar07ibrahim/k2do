"""Message bus module for decoupled channel-agent communication."""

from k2do.bus.events import InboundMessage, OutboundMessage
from k2do.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
