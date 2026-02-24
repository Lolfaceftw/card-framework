from typing import Callable, Dict, List


class EventBus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance.subscribers: Dict[str, List[Callable]] = {}
        return cls._instance

    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, *args, **kwargs):
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(*args, **kwargs)


event_bus = EventBus()
