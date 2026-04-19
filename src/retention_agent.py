import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

class RetentionAgent:
    def __init__(self):
        # Initialize Groq LLM
        self.llm = ChatGroq(
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        self.vector_store = self._setup_rag()

    def _setup_rag(self):
        """Simple RAG setup for retention strategies."""
        # You can expand this text with your specific business rules
        retention_strategies = """
# SEGMENT: INFRASTRUCTURE & TECHNICAL HEALTH
STRATEGY_TECH_01 (Fiber Reliability): Target Fiber Optic users with high technical ticket counts. 
- Action: Proactive "Technical Health Check" call.
- Offer: Complimentary router upgrade to latest Wi-Fi 6 standard or 3 months of "Proactive Monitoring" service.
- Rationale: High-value fiber users churn due to performance frustration; hardware updates reduce friction.

STRATEGY_TECH_02 (Legacy Migration): Target DSL users in areas where Fiber/Cable is newly available.
- Action: Exclusive "Early Adopter" migration path.
- Offer: Free installation + 1st month free on the upgraded tech.
- Rationale: Prevents churn to competitors who are currently marketing superior tech in the same zip code.

# SEGMENT: FINANCIAL & CONTRACTUAL OPTIMIZATION
STRATEGY_FIN_01 (The Commitment Pivot): Target Month-to-Month users with high churn probability.
- Action: Annual Contract Conversion.
- Offer: 15% discount on monthly rate in exchange for a 12-month commitment.
- Rationale: Reduces churn volatility by locking in the user during a high-risk window.

STRATEGY_FIN_02 (Bill Shock Mitigation): Target new users (Tenure < 6 months) with high total charges.
- Action: "Right-Sizing" Consultation.
- Offer: Move to a lower-tier data plan with a "Safety Buffer" (no overage fees for 3 months).
- Rationale: Builds long-term trust by prioritizing customer savings over short-term revenue (CLV > ARPU).

# SEGMENT: ENGAGEMENT & LOYALTY (CLV MAXIMIZATION)
STRATEGY_LOY_01 (High-Value At-Risk): Target users with high usage/revenue but high churn scores.
- Action: "Concierge Status" Assignment.
- Offer: Dedicated support line + 20% loyalty discount for 6 months + Zero-cost premium channel add-ons.
- Rationale: High-revenue users require high-touch human-like intervention to feel valued.

STRATEGY_LOY_02 (Low Usage/Ghosting): Target users with declining data usage over the last 30 days.
- Action: Re-engagement Campaign.
- Offer: Temporary access to a premium streaming partner or data speed boost.
- Rationale: Declining usage is the leading indicator of "silent churn."

# SEGMENT: BUNDLING & STICKINESS
STRATEGY_BND_01 (Single-Play Expansion): Target users with only one service (e.g., just Internet).
- Action: "Better Together" Bundle Offer.
- Offer: Add a secondary service (Mobile or Streaming) at a 50% discount for the first year.
- Rationale: Churn rates drop exponentially as the number of bundled services per household increases.
"""
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = [Document(page_content=x) for x in text_splitter.split_text(retention_strategies)]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)

    def generate_strategy(self, customer_data, churn_prob):
        # Retrieve relevant strategies
        context = self.vector_store.similarity_search(str(customer_data), k=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a customer retention expert. Use the provided strategies and customer data to write a concise, personalized retention plan."),
            ("human", "Customer Data: {data}\nChurn Probability: {prob}%\nRelevant Strategies: {context}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "data": customer_data,
            "prob": round(churn_prob * 100, 2),
            "context": context
        })
        
        return response.content

def run_retention_flow(customer_id, df, model):
    """Integrates with your existing inference pipeline."""
    # Assuming the first row for quick demo purposes or lookup by ID
    customer_data = df.iloc[0].to_dict()
    # Your project already uses joblib to load 'models/rf_model.pkl'
    prob = model.predict_proba(df.drop(columns=['Churn'], errors='ignore'))[0][1]
    
    agent = RetentionAgent()
    return agent.generate_strategy(customer_data, prob)
