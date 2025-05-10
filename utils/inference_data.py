from dataclasses import dataclass

@dataclass
class LinearMemory:
    location_details: str
    difference_reasoning: str
    intent: str
    action_treelines: list[str]
    page_url: str
    is_question: bool

    def __repr__(self):
        return f"({self.location_details})"
        # return f"{self.action_effect} + {self.location_details} + {self.object_details}"

@dataclass
class HiddenInput:
    key: str  # llm sees this
    description: str  # llm sees this to describe what key is
    value: str  # user-defined