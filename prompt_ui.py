import streamlit as st
from prompt_template import summarize_paper, setup_llm
import time

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📬",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4285F4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #357AE8;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📬 Research Paper Summarizer")
st.caption("Powered by Google Gemini 2.5 Flash")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_choice = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash"],
        help="Latest Gemini model"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="0=focused, 1=creative"
    )
    
    st.success("✅ Using Google Gemini API")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Input")
    
    paper_title = st.text_area(
        "Research Paper Title/Topic",
        placeholder="Enter paper title or abstract...\n\nExample: Attention Is All You Need",
        height=150
    )
    
    style = st.selectbox(
        "Explanation Style",
        [
            "Technical but accessible",
            "Simple and beginner-friendly",
            "Advanced technical",
            "Academic formal",
            "Explain like I'm 5"
        ]
    )
    
    length = st.selectbox(
        "Summary Length",
        [
            "Very Short (100-200 words)",
            "Short (300-500 words)",
            "Medium (500-800 words)",
            "Long (800-1200 words)"
        ],
        index=2
    )

with col2:
    st.subheader("💡 Popular Papers")
    st.markdown("""
    **AI/ML:**
    - BERT
    - Attention Is All You Need
    - GPT-3
    - AlphaGo
    
    **Computer Vision:**
    - ResNet
    - YOLO
    - Vision Transformer
    """)

st.markdown("---")

if st.button("🚀 Generate Summary", use_container_width=True):
    if not paper_title.strip():
        st.error("⚠️ Please enter a paper title!")
    else:
        with st.spinner("📄 Generating summary..."):
            try:
                llm = setup_llm(model_name=model_choice, temperature=temperature)
                start_time = time.time()
                
                summary = summarize_paper(
                    paper_title=paper_title,
                    style=style,
                    length=length,
                    llm=llm
                )
                
                gen_time = round(time.time() - start_time, 2)
                
                st.success(f"✅ Generated in {gen_time}s!")
                st.markdown("---")
                
                st.subheader("📄 Summary")
                with st.expander("View Full Summary", expanded=True):
                    st.markdown(summary)
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(
                        "📥 Download TXT",
                        data=summary,
                        file_name=f"summary_{paper_title[:30].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                with col_btn2:
                    st.download_button(
                        "📥 Download MD",
                        data=f"# {paper_title}\n\n{summary}",
                        file_name=f"summary_{paper_title[:30].replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Model", model_choice.split("-")[-1])
                c2.metric("Style", style.split()[0])
                c3.metric("Words", len(summary.split()))
                c4.metric("Time", f"{gen_time}s")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
                if "API" in str(e):
                    st.info("""
                    **Setup Instructions:**
                    1. Get API key: https://aistudio.google.com/apikey
                    2. Create `.env` file
                    3. Add: `GOOGLE_API_KEY=your_key`
                    """)

st.markdown("---")
st.caption("Built with ❤️ using LangChain & Streamlit")