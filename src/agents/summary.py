from .base import BaseAnalysisAgent, AgentCapability

class SummaryAgent(BaseAnalysisAgent):
    @property
    def agent_id(self) -> str:
        return "summary"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            id=self.agent_id,
            keywords=["summarize", "overview", "brief", "summary", "key points"],
            description="Provides high-level summaries of the data",
            priority=0,
        )

    @property
    def prompt_template(self) -> str:
        return """You are a summary agent specializing in insurance data analysis.
        Your role is to provide clear, concise summaries of insurance data.
        Focus on:
        1. Key statistics and figures
        2. Overall patterns
        3. Important highlights
        4. Notable outliers
        
        Below you will find insurance cost data. Summarize this data to answer the question.
        Keep the summary focused and data-driven.
        
        {context}
        
        Question: {question}
        
        Answer: Let me analyze this based on the data provided.""" 