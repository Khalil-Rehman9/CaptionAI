import os
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import logging
from PIL import Image
import streamlit as st
import ollama
from transformers import AutoModelForCausalLM, AutoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Configuration settings for the application"""
    PAGE_TITLE: str = "Advanced Image Captioning"
    PAGE_ICON: str = "🖼️"
    SUPPORTED_FORMATS: List[str] = ("jpg", "jpeg", "png")
    BASE_CAPTION_DIR: Path = Path("captions")
    MAX_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)  # Maximum image dimensions
    FLORENCE_MODEL_PATH: str = "MiaoshouAI/Florence-2-base-PromptGen"
    LLAMA_MODEL_NAME: str = "llama3.2-vision"

class ImageProcessor:
    """Handles image processing operations"""
    
    @staticmethod
    def preprocess_image(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
        """Resize image if it exceeds maximum dimensions while maintaining aspect ratio"""
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size)
        return image

    @staticmethod
    def validate_image(image: Image.Image) -> bool:
        """Validate image format and dimensions"""
        try:
            image.verify()
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False

class CaptioningModel:
    """Base class for captioning models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def generate_caption(self, image: Image.Image) -> str:
        raise NotImplementedError

class Florence2Model(CaptioningModel):
    """Florence-2 captioning model implementation"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None

    def load_model(self):
        """Lazy loading of the Florence-2 model"""
        if self.model is None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    AppConfig.FLORENCE_MODEL_PATH,
                    trust_remote_code=True
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(
                    AppConfig.FLORENCE_MODEL_PATH,
                    trust_remote_code=True
                )
                logger.info("Florence-2 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Florence-2 model: {e}")
                raise

    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption using Florence-2 model"""
        self.load_model()
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.replace("<MORE_DETAILED_CAPTION>:", "").strip() + " in the Style of LOGA"

class LlamaVisionModel(CaptioningModel):
    """Llama 3.2 Vision model implementation"""
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption using Llama Vision model"""
        try:
            # Save image temporarily
            temp_path = Path("temp_image.jpg")
            image.save(temp_path)
            
            response = ollama.chat(
                model=AppConfig.LLAMA_MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': """Craft a concise, descriptive caption that vividly outlines the visual elements, 
                    composition, and context, focusing on details such as color, texture, setting, and objects. 
                    Use clear, straightforward language suitable for training a generative model.""",
                    'images': [str(temp_path)]
                }]
            )
            
            return response['message']['content']
        finally:
            if temp_path.exists():
                temp_path.unlink()

class CaptionManager:
    """Manages caption generation and storage"""
    
    def __init__(self):
        self.florence_model = Florence2Model()
        self.llama_model = LlamaVisionModel()

    def save_caption(self, image_name: str, caption: str, model_name: str) -> Path:
        """Save generated caption to file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = AppConfig.BASE_CAPTION_DIR / model_name / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
        
        caption_file = output_path / f"{Path(image_name).stem}.txt"
        caption_file.write_text(caption)
        logger.info(f"Caption saved to {caption_file}")
        return caption_file

    def process_image(self, image: Image.Image, image_name: str, model_choice: str) -> Tuple[str, Path]:
        """Process image and generate caption"""
        try:
            model = self.florence_model if model_choice == "Florence-2" else self.llama_model
            caption = model.generate_caption(image)
            save_path = self.save_caption(image_name, caption, model_choice.lower())
            return caption, save_path
        except Exception as e:
            logger.error(f"Error processing image {image_name}: {e}")
            raise

class StreamlitUI:
    """Handles Streamlit UI components and styling"""
    
    @staticmethod
    def setup_page():
        """Configure page settings and styling"""
        st.set_page_config(
            page_title=AppConfig.PAGE_TITLE,
            page_icon=AppConfig.PAGE_ICON,
            layout="wide"
        )
        
        st.markdown("""
            <style>
            .stButton>button { width: 100%; height: 3em; margin-top: 1em; }
            .caption-box { padding: 1em; border-radius: 0.5em; background-color: #f0f2f6; margin: 1em 0; }
            .success-message { padding: 1em; border-radius: 0.5em; background-color: #d1e7dd; 
                             color: #0f5132; margin: 1em 0; }
            .error-message { padding: 1em; border-radius: 0.5em; background-color: #f8d7da; 
                           color: #842029; margin: 1em 0; }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_model_selection() -> str:
        """Render model selection interface"""
        col1, col2 = st.columns(2)
        with col1:
            st.info("📷 Florence-2 Model\n\nBest for detailed visual descriptions and artistic style recognition")
        with col2:
            st.info("🦙 Llama 3.2 Vision Model\n\nExcels at natural language descriptions and context understanding")
        
        return st.radio("Select Captioning Model:", ["Florence-2", "Llama 3.2 Vision"], horizontal=True)

def main():
    """Main application entry point"""
    ui = StreamlitUI()
    ui.setup_page()
    
    st.title("🖼️ Advanced Image Captioning")
    st.markdown("### Choose your preferred captioning model and upload images")
    
    model_choice = ui.render_model_selection()
    uploaded_files = st.file_uploader(
        "Upload your images",
        type=list(AppConfig.SUPPORTED_FORMATS),
        accept_multiple_files=True
    )
    
    if uploaded_files:
        caption_manager = CaptionManager()
        image_processor = ImageProcessor()
        
        for uploaded_file in uploaded_files:
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            
            try:
                # Process image
                image = Image.open(uploaded_file).convert("RGB")
                image = image_processor.preprocess_image(image, AppConfig.MAX_IMAGE_SIZE)
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    with st.spinner("Generating caption..."):
                        caption, save_path = caption_manager.process_image(
                            image, uploaded_file.name, model_choice
                        )
                        
                        st.markdown(
                            f"<div class='caption-box'><b>Generated Caption:</b><br>{caption}</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<div class='success-message'>Caption saved to: {save_path}</div>",
                            unsafe_allow_html=True
                        )
            
            except Exception as e:
                st.markdown(
                    f"<div class='error-message'>Error processing {uploaded_file.name}: {str(e)}</div>",
                    unsafe_allow_html=True
                )
        
        st.success("✨ All images processed successfully!")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Upload multiple images and get AI-generated captions using state-of-the-art models.</p>
            <p>Captions are automatically saved with timestamps for easy reference.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()