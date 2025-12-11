import streamlit as st
from src.analyzers.gemini_analyzer import GeminiAnalyzer

def render_gemini_ai(gemini_analyzer: GeminiAnalyzer):
    """
    Render the Gemini AI Analysis tab.
    
    Args:
        gemini_analyzer: The Gemini analyzer instance.
    """
    st.markdown(
        '<h2 class="sub-header">Gemini AI Analysis</h2>', unsafe_allow_html=True
    )
    
    if not gemini_analyzer.is_configured():
        st.warning(
            "Please configure your Gemini API key in the sidebar to use AI analysis."
        )
    else:
        st.markdown(
            '<div class="info-text">This section uses Google\'s Gemini AI to provide insights about your data.</div>',
            unsafe_allow_html=True,
        )

        # Analysis options
        analysis_type = st.selectbox(
            "Select analysis type:", gemini_analyzer.get_analysis_types()
        )

        # Custom question for custom analysis
        custom_question = None
        if analysis_type == "Custom Analysis":
            custom_question = st.text_area(
                "Enter your specific question about the data:",
                "What are the most interesting insights from this dataset and what actions would you recommend based on them?",
            )

        # Run analysis button
        if st.button("Run AI Analysis"):
            with st.spinner("Gemini AI is analyzing your data..."):
                # Generate analysis
                response = gemini_analyzer.analyze_data(
                    analysis_type, custom_question
                )

                if response:
                    # Display response
                    st.markdown(
                        '<h3 class="sub-header">AI Analysis Results</h3>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(response)

                    # Save analysis to file option
                    st.download_button(
                        label="Download Analysis",
                        data=response,
                        file_name="gemini_analysis.txt",
                        mime="text/plain",
                    )
