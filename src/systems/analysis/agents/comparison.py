from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from .abstract.base import BaseAnalysisAgent, AgentCapability

class ComparisonAnalysisAgent(BaseAnalysisAgent):
    """Agent specialized in comparative analysis"""
    
    @property
    def agent_id(self) -> str:
        return "comparison_analyst"

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
        return """Compare and analyze the following data points:

Context:
{context}

Query: {question}

Focus on:
1. Key differences between time periods
2. Relative magnitudes of changes
3. Acceleration or deceleration in trends
4. Unusual or unexpected variations
5. Potential correlations between metrics

Provide insights about the most significant comparisons and their implications."""
    
    def _create_chain(self) -> RunnableSequence:
        """Create the comparison analysis chain"""
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        return prompt | self.llm | StrOutputParser() 