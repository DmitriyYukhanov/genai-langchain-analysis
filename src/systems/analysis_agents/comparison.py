from .base import BaseAnalysisAgent, AgentCapability

class ComparisonAgent(BaseAnalysisAgent):
    @property
    def agent_id(self) -> str:
        return "comparison"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            id=self.agent_id,
            keywords=["compare", "difference", "versus", "vs", "between"],
            description="Compares metrics between different periods or categories",
            priority=2,
        )

    @property
    def prompt_template(self) -> str:
        return """You are a comparison agent specializing in insurance data analysis.
        Your role is to compare insurance metrics between different time periods.
        Focus on:
        1. Direct numerical comparisons
        2. Percentage differences
        3. Relative changes
        
        Below you will find insurance cost data. Compare this data to answer the question.
        Always show your calculations and cite the specific years being compared.
        
        {context}
        
        Question: {question}
        
        Answer: Let me analyze this based on the data provided.""" 