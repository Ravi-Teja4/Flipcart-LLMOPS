from langchain_groq import ChatGroq
# Correct imports - these functions are in langchain-core, not langchain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from flipkart.config import Config

class RAGChainBuilder:
    def __init__(self, vector_store):  # Fixed: double underscores
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Contextualize question prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                          Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Build the chain manually
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create contextualized retriever
        contextualize_chain = contextualize_q_prompt | self.model | StrOutputParser()
        
        def contextualized_retriever(input_dict):
            if input_dict.get("chat_history"):
                contextualized_q = contextualize_chain.invoke(input_dict)
                return retriever.invoke(contextualized_q)
            else:
                return retriever.invoke(input_dict["input"])

        # Create the RAG chain
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(contextualized_retriever(x))
            )
            | qa_prompt
            | self.model
            | StrOutputParser()
        )

        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        return conversational_rag_chain