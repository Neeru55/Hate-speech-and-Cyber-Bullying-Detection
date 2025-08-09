import streamlit as st
import tensorflow as tf
from src.utils.data_preprocessor import DataPreprocessor
from src.model import HateSpeechDetector

# Set page config
st.set_page_config(
    page_title="Hate Speech & Cyberbullying Detector",
    page_icon="🚫",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        background-color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .safe {
        background-color: #d4edda;
        color: #155724;
    }
    .unsafe {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize model and preprocessor
@st.cache_resource
def load_model():
    preprocessor = DataPreprocessor()
    detector = HateSpeechDetector()
    detector.load_model('models/final_model')
    return detector, preprocessor

# Main app
def main():
    st.title("🚫 Hate Speech & Cyberbullying Detector")
    st.markdown("""
    This application uses BERT to detect different types of cyberbullying and hate speech in text.
    Enter any text below to analyze it for harmful content.
    """)
    
    # Load model
    detector, preprocessor = load_model()
    
    # Text input
    text = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste text here...",
        height=150
    )
    
    if st.button("Analyze Text"):
        if text:
            with st.spinner("Analyzing text..."):
                # Make prediction
                result = detector.predict(text, preprocessor)
                
                # Display results
                st.markdown("### Analysis Results")
                
                if result['predicted_class'] == 'not cyberbullying':
                    st.markdown(f"""
                    <div class="prediction-box safe">
                        <h3>✅ Safe Content</h3>
                        <p>Confidence: {result['confidence']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box unsafe">
                        <h3>⚠️ Potential {result['predicted_class'].title()} Content</h3>
                        <p>Confidence: {result['confidence']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show explanation
                st.markdown("""
                ### About the Detection
                - The model analyzes text for various types of harmful content
                - Categories include: age, ethnicity, gender, religion, and other types of cyberbullying
                - Results are based on the confidence score of the model's prediction
                """)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 