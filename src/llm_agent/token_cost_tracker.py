from llm_agent.config import MODEL_PRICING

class TokenCostTracker:
    def __init__(self, model_id):
        pricing = MODEL_PRICING.get(model_id, {
            "input_cost_per_1k": 0.003,  # default to Claude 3 Sonnet pricing
            "output_cost_per_1k": 0.015
        })
        self.input_cost_per_1k = pricing["input_cost_per_1k"]
        self.output_cost_per_1k = pricing["output_cost_per_1k"]
        self.input_tokens = 0
        self.output_tokens = 0

    def add_input_tokens(self, count):
        self.input_tokens += count

    def add_output_tokens(self, count):
        self.output_tokens += count

    def calculate_costs(self):
        input_cost = (self.input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (self.output_tokens / 1000) * self.output_cost_per_1k
        total_cost = input_cost + output_cost
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }

    def to_dict(self):
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens
        }

    @classmethod
    def from_dict(cls, data, model_id):
        tracker = cls(model_id)
        tracker.input_tokens = data.get('input_tokens', 0)
        tracker.output_tokens = data.get('output_tokens', 0)
        return tracker
