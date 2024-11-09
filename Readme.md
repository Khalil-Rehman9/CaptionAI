
![alt text](https://github.com/Khalil-Rehman9/CaptionAI/blob/main/UI.png)

# 🖼️ Advanced Image Captioning App

A powerful and user-friendly tool that generates detailed captions for your images using state-of-the-art AI models. This app combines the strengths of Florence-2 and Llama 3.2 Vision models to provide rich, contextual descriptions of any image you upload.

## ✨ Features

- **Dual Model Support**: Choose between Florence-2 and Llama 3.2 Vision models
- **Batch Processing**: Upload and process multiple images at once
- **Organized Output**: Captions are saved with timestamps for easy reference
- **User-Friendly Interface**: Clean, intuitive Streamlit-based UI
- **Error Handling**: Comprehensive error messages and logging

## 🚀 Getting Started

### Prerequisites

Before you dive in, make sure you have Python 3.8+ installed on your machine. You'll also need some disk space for the AI models.
You will also need to install the Ollama in your Local machine you can visit here [Ollama Install](path/to/ollama-install.md)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/image-captioning-app.git
cd image-captioning-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running the App

Launch the app with this simple command:
```bash
streamlit run app.py
```

The app will open in your default web browser. If it doesn't, just head to `http://localhost:8501`.

## 🎯 How to Use

1. **Choose Your Model**:
   - **Florence-2**: Great for detailed visual descriptions and artistic style recognition
   - **Llama 3.2 Vision**: Excels at natural language descriptions and context understanding

2. **Upload Images**:
   - Click the upload button or drag and drop your images
   - Supports JPG, JPEG, and PNG formats
   - Upload multiple images at once for batch processing

3. **Get Captions**:
   - The app processes each image and shows the generated caption
   - Captions are automatically saved in the `captions` folder
   - Each session gets its own timestamped folder

## 📁 Project Structure

```
image-captioning-app/
├── app.py                 # Main application file
├── captions/             # Generated captions directory
│   ├── florence2/        # Florence-2 model outputs
│   └── llama32/         # Llama 3.2 Vision model outputs
├── requirements.txt      # Project dependencies
└── README.md            # This documentation
```

## 🛠️ Technical Details

### Components

- **ImageProcessor**: Handles image validation and preprocessing
- **CaptioningModel**: Base class for AI models
- **Florence2Model**: Implementation of the Florence-2 model
- **LlamaVisionModel**: Implementation of the Llama 3.2 Vision model
- **CaptionManager**: Manages caption generation and storage
- **StreamlitUI**: Handles the user interface

### Key Features Explained

#### Image Preprocessing
```python
def preprocess_image(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    """Resize image if it exceeds maximum dimensions while maintaining aspect ratio"""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size)
    return image
```

#### Caption Generation
```python
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
```

## 🔧 Configuration

The app uses a centralized configuration class (`AppConfig`) for easy customization:

```python
@dataclass
class AppConfig:
    PAGE_TITLE: str = "Advanced Image Captioning"
    PAGE_ICON: str = "🖼️"
    SUPPORTED_FORMATS: List[str] = ("jpg", "jpeg", "png")
    MAX_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)
    # ... other configurations
```

## 📝 Logging

The app includes comprehensive logging for better debugging and monitoring:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

## 🤝 Contributing

We'd love your contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Make sure you have enough disk space
   - Check your internet connection
   - Verify CUDA availability for GPU support

2. **Image Processing Errors**:
   - Verify image format is supported
   - Check if image file is corrupted
   - Ensure image dimensions are reasonable

3. **Memory Issues**:
   - Try processing fewer images at once
   - Close other memory-intensive applications
   - Consider using a machine with more RAM

## 📈 Future Improvements

Here are some features we're planning to add:

- [ ] Support for more AI models
- [ ] Custom prompt engineering interface
- [ ] Batch export options
- [ ] Caption editing and version history
- [ ] Integration with cloud storage services

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the Anthropic team for the Florence-2 model
- Thanks to the Llama team for the Llama 3.2 Vision model
- Thanks to the Streamlit team for their amazing framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Look through existing GitHub issues
3. Create a new issue with detailed information about your problem

Remember to include:
- Your operating system
- Python version
- Error messages
- Steps to reproduce the issue

---

Built with ❤️ by Khalil

```
Don't forget to star ⭐ this repo if you found it helpful!
```
