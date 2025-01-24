from .base import BaseAnalysisAgent, AgentCapability

class TrendAnalysisAgent(BaseAnalysisAgent):
    @property
    def agent_id(self) -> str:
        return "trend"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            id=self.agent_id,
            keywords=["trend", "change", "over time", "pattern", "historical"],
            description="Analyzes trends and patterns in data over time",
            priority=1,
        )

    @property
    def prompt_template(self) -> str:
        return """You are a trend analysis agent specializing in insurance data analysis.
        Your role is to identify and explain significant trends in insurance costs and expenditures over time.
        Focus on:
        1. Year-over-year changes
        2. Long-term patterns
        3. Significant turning points
        4. Rate of change analysis
        
        Below you will find insurance cost data. Analyze this data to answer the question.
        Provide specific numbers and percentages to support your analysis.
        Always cite the years you're referring to.
        
        {context}
        
        Question: {question}
        
        Answer: Let me analyze this based on the data provided.""" 