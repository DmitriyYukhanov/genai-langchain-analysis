from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from .abstract.base import BaseAnalysisAgent, AgentCapability

class SummaryAgent(BaseAnalysisAgent):
    """Agent specialized in providing high-level summaries"""
    
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
        return """Provide a clear and concise summary of the following data:

Context:
{context}

Query: {question}

Focus on:
1. Key statistics and figures
2. Overall patterns
3. Important highlights
4. Notable outliers

Keep the summary focused and data-driven."""
    
    def _create_chain(self) -> RunnableSequence:
        """Create the summary analysis chain"""
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        return prompt | self.llm | StrOutputParser() 