from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from .abstract.base import BaseAnalysisAgent, AgentCapability

class TrendAnalysisAgent(BaseAnalysisAgent):
    """Agent specialized in trend analysis"""
    
    @property
    def agent_id(self) -> str:
        return "trend_analyst"

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
        return """Analyze the following data for trends and patterns:

Context:
{context}

Query: {question}

Focus on:
1. Overall trends in expenditures over time
2. Notable changes or inflection points
3. Growth rates and their implications
4. Any cyclical or seasonal patterns
5. Potential factors influencing the trends

Provide a clear and concise analysis that highlights the most important findings."""
    
    def _create_chain(self) -> RunnableSequence:
        """Create the trend analysis chain"""
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        return prompt | self.llm | StrOutputParser() 