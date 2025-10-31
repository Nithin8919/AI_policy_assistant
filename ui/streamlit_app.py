"""
AI Policy Assistant - Streamlit Interface
Interactive UI for querying Andhra Pradesh education policy documents
"""

import streamlit as st
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.enhanced_router import EnhancedRouter
from src.synthesis.qa_pipeline import QAPipeline, QAResponse

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="AI Policy Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        font-size: 1.1rem;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .answer-box {
        padding: 1.5rem;
        background-color: #ffffff;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .citation-box {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .metric-card {
        padding: 1rem;
        background-color: #e8f4f8;
        border-radius: 10px;
        text-align: center;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Initialization
# ============================================================================

if 'qa_pipeline' not in st.session_state:
    st.session_state.qa_pipeline = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# ============================================================================
# Helper Functions
# ============================================================================

def initialize_pipeline():
    """Initialize the QA pipeline with environment variables"""
    try:
        with st.spinner("🔄 Initializing AI Policy Assistant..."):
            # Get credentials from environment
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            claude_api_key = os.getenv("ANTHROPIC_API_KEY")
            
            if not all([qdrant_url, qdrant_api_key, claude_api_key]):
                st.error("⚠️ Missing required environment variables. Please set QDRANT_URL, QDRANT_API_KEY, and ANTHROPIC_API_KEY")
                return False
            
            # Initialize router
            router = EnhancedRouter(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key
            )
            
            # Initialize QA pipeline
            st.session_state.qa_pipeline = QAPipeline(
                router=router,
                claude_api_key=claude_api_key,
                enable_usage_tracking=True
            )
            
            st.session_state.initialized = True
            st.success("✅ AI Policy Assistant initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        return False


def format_confidence(score: float) -> str:
    """Format confidence score with color"""
    if score >= 0.7:
        return f'<span class="confidence-high">{score:.0%}</span>'
    elif score >= 0.4:
        return f'<span class="confidence-medium">{score:.0%}</span>'
    else:
        return f'<span class="confidence-low">{score:.0%}</span>'


def display_citation(citation: dict, index: int):
    """Display a single citation card"""
    with st.expander(f"📄 Source {index}: {citation.get('document', 'Unknown')}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Type", citation.get('doc_type', 'N/A'))
        with col2:
            st.metric("Year", citation.get('year', 'N/A'))
        with col3:
            st.metric("Relevance", f"{citation.get('score', 0):.2f}")
        
        st.markdown("**Section:** " + citation.get('section', 'N/A'))
        if citation.get('chunk_id'):
            st.caption(f"Chunk ID: {citation['chunk_id']}")


def create_metrics_chart(response: QAResponse):
    """Create visualization of retrieval and LLM metrics"""
    
    # Prepare data
    metrics_data = {
        'Metric': [
            'Retrieval Time',
            'LLM Time',
            'Total Time',
            'Chunks Retrieved',
            'Sources Cited',
            'Input Tokens',
            'Output Tokens'
        ],
        'Value': [
            response.retrieval_stats.get('retrieval_time', 0),
            response.llm_stats.get('llm_time', 0),
            response.processing_time,
            response.retrieval_stats.get('chunks_retrieved', 0),
            response.citations.get('unique_sources_cited', 0),
            response.llm_stats.get('input_tokens', 0) / 1000,  # Convert to k tokens
            response.llm_stats.get('output_tokens', 0) / 1000
        ],
        'Unit': ['sec', 'sec', 'sec', 'count', 'count', 'k tokens', 'k tokens']
    }
    
    return pd.DataFrame(metrics_data)


def create_citation_network(citations: dict):
    """Create visualization of citation network"""
    citation_details = citations.get('citation_details', {})
    
    if not citation_details:
        return None
    
    # Prepare data for visualization
    sources = []
    scores = []
    doc_types = []
    
    for cite_num, details in citation_details.items():
        sources.append(f"Source {cite_num}")
        scores.append(details.get('score', 0))
        doc_types.append(details.get('doc_type', 'unknown'))
    
    fig = go.Figure(data=[
        go.Bar(
            x=sources,
            y=scores,
            text=[f"{s:.2f}" for s in scores],
            textposition='auto',
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Relevance: %{y:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Source Relevance Scores",
        xaxis_title="Sources",
        yaxis_title="Relevance Score",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


# ============================================================================
# Main UI
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 AI Policy Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Andhra Pradesh Education Policy Knowledge Base")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize System", type="primary"):
                initialize_pipeline()
        else:
            st.success("✅ System Ready")
            
            if st.button("🔄 Reinitialize"):
                st.session_state.initialized = False
                st.session_state.qa_pipeline = None
                st.rerun()
        
        st.divider()
        
        # Query Settings
        st.header("🎛️ Query Settings")
        
        mode = st.selectbox(
            "Answer Mode",
            options=["normal_qa", "detailed", "concise", "comparative"],
            format_func=lambda x: {
                "normal_qa": "🔍 Normal Q&A",
                "detailed": "📊 Detailed Analysis",
                "concise": "⚡ Quick Answer",
                "comparative": "⚖️ Comparative"
            }[x],
            help="Select the type of answer you want"
        )
        
        top_k = st.slider(
            "Number of Sources",
            min_value=3,
            max_value=20,
            value=10,
            help="Number of document chunks to retrieve"
        )
        
        st.divider()
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            show_retrieval_details = st.checkbox("Show Retrieval Details", value=True)
            show_token_usage = st.checkbox("Show Token Usage", value=True)
            show_citations_chart = st.checkbox("Show Citation Chart", value=True)
        
        st.divider()
        
        # Usage Statistics
        if st.session_state.qa_pipeline and st.session_state.initialized:
            st.header("📊 Session Statistics")
            
            usage_stats = st.session_state.qa_pipeline.get_usage_stats()
            
            st.metric("Total Queries", usage_stats.get('total_calls', 0))
            st.metric("Total Tokens", f"{usage_stats.get('total_input_tokens', 0) + usage_stats.get('total_output_tokens', 0):,}")
            st.metric("Est. Cost", f"${usage_stats.get('total_cost_usd', 0):.4f}")
            
            if st.button("📋 Export Usage Log"):
                st.download_button(
                    label="Download Usage Data",
                    data=json.dumps(usage_stats, indent=2),
                    file_name=f"usage_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        st.divider()
        
        # Query History
        if st.session_state.query_history:
            st.header("📝 Query History")
            for i, hist_query in enumerate(reversed(st.session_state.query_history[-5:])):
                if st.button(f"📌 {hist_query[:30]}...", key=f"hist_{i}"):
                    st.session_state.current_query = hist_query
    
    # Main Content
    if not st.session_state.initialized:
        st.info("👈 Please initialize the system using the sidebar to get started.")
        
        # Show system information
        st.markdown("""
        ### Welcome to the AI Policy Assistant!
        
        This system helps you:
        - 🔍 Query Andhra Pradesh education policy documents
        - 📊 Get detailed, citation-based answers
        - 📚 Access Government Orders, Acts, Rules, and Schemes
        - ⚡ Compare different policy provisions
        
        **To get started:**
        1. Ensure you have set the required environment variables
        2. Click "Initialize System" in the sidebar
        3. Start asking questions!
        """)
        
        return
    
    # Query Input
    st.markdown("### 💬 Ask Your Question")
    
    query = st.text_area(
        "Enter your policy question:",
        height=100,
        placeholder="Example: What are the eligibility criteria for the Amma Vodi scheme?",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    with col3:
        example_button = st.button("💡 Example", use_container_width=True)
    
    # Example queries
    if example_button:
        examples = [
            "What are the responsibilities of School Management Committees under RTE Act?",
            "What is the eligibility criteria for the Amma Vodi scheme?",
            "How are teachers appointed in government schools?",
            "What are the provisions for children with disabilities in schools?",
            "Compare the powers of District Education Officer and Mandal Education Officer"
        ]
        st.info(f"**Example:** {examples[len(st.session_state.query_history) % len(examples)]}")
    
    # Clear response
    if clear_button:
        st.session_state.current_response = None
        st.rerun()
    
    # Process query
    if search_button and query.strip():
        try:
            with st.spinner("🔄 Processing your query..."):
                # Add to history
                st.session_state.query_history.append(query)
                
                # Get answer
                response = st.session_state.qa_pipeline.answer_query(
                    query=query,
                    mode=mode,
                    top_k=top_k
                )
                
                st.session_state.current_response = response
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            return
    
    # Display Results
    if st.session_state.current_response:
        response = st.session_state.current_response
        
        st.markdown("---")
        st.markdown("## 📝 Results")
        
        # Query and Confidence
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div class="query-box"><strong>Query:</strong> {response.query}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Confidence:** {format_confidence(response.confidence_score)}", unsafe_allow_html=True)
            st.caption(f"Mode: {response.mode}")
        
        # Answer
        st.markdown("### 💡 Answer")
        st.markdown(f'<div class="answer-box">{response.answer}</div>', unsafe_allow_html=True)
        
        # Metrics
        st.markdown("### 📊 Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Processing Time",
                f"{response.processing_time:.2f}s"
            )
        
        with col2:
            st.metric(
                "Sources Retrieved",
                response.retrieval_stats.get('chunks_retrieved', 0)
            )
        
        with col3:
            st.metric(
                "Sources Cited",
                response.citations.get('unique_sources_cited', 0)
            )
        
        with col4:
            st.metric(
                "Total Tokens",
                f"{response.llm_stats.get('total_tokens', 0):,}"
            )
        
        # Citation Details
        st.markdown("### 📚 Citations")
        
        citation_details = response.citations.get('citation_details', {})
        
        if citation_details:
            # Citation statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Citations", response.citations.get('total_citations', 0))
            with col2:
                st.metric("Unique Sources", response.citations.get('unique_sources_cited', 0))
            with col3:
                valid = "✅" if response.citations.get('all_citations_valid', False) else "⚠️"
                st.metric("Valid", valid)
            
            # Citation chart
            if show_citations_chart:
                fig = create_citation_network(response.citations)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Individual citations
            st.markdown("#### 📑 Source Details")
            for cite_num, citation in sorted(citation_details.items(), key=lambda x: int(x[0])):
                display_citation(citation, cite_num)
        else:
            st.info("No citations found in the answer.")
        
        # Retrieval Details
        if show_retrieval_details:
            st.markdown("### 🔍 Retrieval Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Agents Used:**")
                agents = response.retrieval_stats.get('agents_used', [])
                for agent in agents:
                    st.markdown(f"- {agent}")
            
            with col2:
                st.markdown("**Query Complexity:**")
                st.info(response.retrieval_stats.get('query_complexity', 'N/A'))
                
                st.markdown("**Retrieval Time:**")
                st.info(f"{response.retrieval_stats.get('retrieval_time', 0):.2f}s")
        
        # Token Usage
        if show_token_usage:
            st.markdown("### 🎫 Token Usage")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Input Tokens", f"{response.llm_stats.get('input_tokens', 0):,}")
            with col2:
                st.metric("Output Tokens", f"{response.llm_stats.get('output_tokens', 0):,}")
            with col3:
                cost = (response.llm_stats.get('input_tokens', 0) / 1_000_000 * 3.0 + 
                       response.llm_stats.get('output_tokens', 0) / 1_000_000 * 15.0)
                st.metric("Est. Cost", f"${cost:.4f}")
        
        # Export Options
        st.markdown("### 💾 Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Answer (JSON)",
                data=response.to_json(),
                file_name=f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Create markdown export
            markdown_export = f"""# Query: {response.query}

## Answer
{response.answer}

## Metadata
- **Confidence:** {response.confidence_score}
- **Mode:** {response.mode}
- **Processing Time:** {response.processing_time:.2f}s
- **Sources Retrieved:** {response.retrieval_stats.get('chunks_retrieved', 0)}
- **Sources Cited:** {response.citations.get('unique_sources_cited', 0)}

## Citations
{chr(10).join([f"- Source {num}: {details.get('document', 'Unknown')}" for num, details in citation_details.items()])}
"""
            st.download_button(
                label="📥 Download Answer (Markdown)",
                data=markdown_export,
                file_name=f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )


# ============================================================================
# Footer
# ============================================================================

def footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>AI Policy Assistant for Andhra Pradesh Education Policy</p>
        <p>Powered by Claude 4 Sonnet • Qdrant Vector Database</p>
        <p style='font-size: 0.8rem;'>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Run App
# ============================================================================

if __name__ == "__main__":
    main()
    footer()
