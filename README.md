# Background Remover API

A powerful Python FastAPI application that intelligently removes image backgrounds using the state-of-the-art **U²-Net** deep learning model. The API processes images entirely in memory and returns clean results with white backgrounds in JPEG format.

## ✨ Features

- 🖼️ **Smart Background Removal**: Uses U²-Net for accurate human and object segmentation
- 🚀 **Dual Interface**: Upload images via web form or programmatic API
- ⚡ **Memory Efficient**: Processes images entirely in memory without temporary files
- 🎯 **High Accuracy**: Leverages deep learning for precise edge detection
- 📱 **RESTful API**: Easy integration with any application
- 🔄 **Optional Preprocessing**: Uses `rembg` for faster initial processing when available
- 🖱️ **User-Friendly**: Simple web interface for quick testing

## 🛠️ Technical Stack

- **FastAPI**: Modern, fast web framework
- **U²-Net**: Deep learning model for salient object detection
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **Pillow**: Image processing
- **rembg**: Optional preprocessing library

## 📋 Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.5.0
opencv-python>=4.7.0
rembg>=2.0.50
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Nik-doid/bgremover.git
cd bgremover
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download U²-Net Model

Download the pre-trained U²-Net model and place it in the correct directory:

```bash
# Create model directory
mkdir -p saved_models/u2net

# Download the model (replace with actual download command or manual download)
# Download u2net-human-seg.pth from the official U²-Net repository
# Save it to: saved_models/u2net/u2net-human-seg.pth
```

> **Note**: Download `u2net-human-seg.pth` from the [official U²-Net repository](https://github.com/xuebinqin/U-2-Net) and save it in `saved_models/u2net/` directory.

### 5. Run the Application

```bash
python3 main.py
```

The API will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📚 API Usage

### Web Interface

1. Open your browser and go to `http://localhost:8000`
2. Upload an image using the web form
3. Click "Remove Background"
4. Download the processed image

### Programmatic API

#### Upload Image

```bash
curl -X POST "http://localhost:8000/remove-bg" \
     -H "accept: image/jpeg" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your-image.jpg"
```

#### Python Example

```python
import requests

url = "http://localhost:8000/remove-bg"
files = {"file": open("your-image.jpg", "rb")}

response = requests.post(url, files=files)

if response.status_code == 200:
    with open("output-image.jpg", "wb") as f:
        f.write(response.content)
    print("Background removed successfully!")
else:
    print(f"Error: {response.status_code}")
```

#### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/remove-bg', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const url = URL.createObjectURL(blob);
    // Use the URL to display or download the image
});
```

## 📁 Project Structure

```
background-remover-api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── saved_models/
│   └── u2net/
│       └── u2net-human-seg.pth  # U²-Net model file            
├── templates/             # HTML templates (if any)
├── u2Net                   # u2net repo 
```

## ⚙️ Configuration

The application can be configured by modifying the following parameters in `main.py`:

- **Host**: Default `0.0.0.0`
- **Port**: Default `8000`
- **Model Path**: `saved_models/u2net/u2net-human-seg.pth`
- **Maximum File Size**: Configure in FastAPI settings
- **Allowed File Types**: JPG, PNG, JPEG, etc.

## 🐳 Docker Support (Optional)

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

Build and run:

```bash
docker build -t background-remover .
docker run -p 8000:8000 background-remover
```

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Found Error**
   - Ensure `u2net-human-seg.pth` is in `saved_models/u2net/` directory
   - Check file permissions

2. **Memory Issues**
   - Reduce image size before processing
   - Increase system RAM or use GPU acceleration

3. **Slow Processing**
   - Install `rembg` for faster preprocessing
   - Consider using GPU-enabled PyTorch

### Performance Tips

- Use GPU acceleration if available
- Resize large images before processing
- Consider batch processing for multiple images
- Implement caching for frequently processed images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this project in your research, please cite the original U²-Net paper:

```bibtex
@InProceedings{Qin_2020_PR,
    title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
    author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
    journal = {Pattern Recognition},
    volume = {106},
    pages = {107404},
    year = {2020}
}
```

## 🔗 Related Links

- [U²-Net Official Repository](https://github.com/xuebinqin/U-2-Net)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [rembg Library](https://github.com/danielgatis/rembg)

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](link-to-issues) section
2. Create a new issue with detailed description
3. Include error logs and system information

---

**Made with ❤️ using U²-Net and FastAPI**
