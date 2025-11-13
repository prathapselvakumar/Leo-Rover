# Leo Rover 3D Object Detection and Analysis System

## 📝 Project Overview
This project implements a real-time 3D object detection and analysis system designed for the Leo Rover platform. The system utilizes an Intel RealSense depth camera to detect objects, identify their colors, and classify their 3D shapes, providing spatial awareness for autonomous navigation and interaction.

## 🎯 Key Features

### Core Functionality
- **3D Object Detection**: Real-time detection and tracking of objects in 3D space
- **Color Recognition**: Advanced color identification using K-means clustering in HSV color space
- **Shape Classification**: Intelligent 3D shape classification using contour and depth analysis
- **Depth Sensing**: Utilizes Intel RealSense D400 series for accurate depth measurement
- **Edge Detection**: Combines color and depth data for robust edge detection

### Technical Highlights
- **YOLOv8 Integration**: For fast and accurate object detection
- **Real-time Processing**: Optimized for performance on edge devices
- **Hand-Eye Calibration**: For precise spatial mapping between camera and robotic arm
- **Modular Architecture**: Easy to extend and customize for different applications

## 🏗️ Project Structure
```
Leo-Rover/
├── Coding/
│   ├── Hand-Eye/               # Hand-eye calibration implementation
│   └── Object Detection/       # Main object detection implementation
│       ├── object_shape_color_depth.py  # Main detection script
│       ├── hand_eye_calibration.py      # Camera-robot calibration
│       ├── requirements.txt             # Python dependencies
│       └── README.md                    # Detailed documentation
├── Documentation/              # Project documentation and reports
└── Output/                     # Sample outputs and results
    ├── 3D Real-Sense/          # 3D scanning results
    ├── Co-Bot/                 # Collaborative robot integration
    └── Trail Run/              # Field test results
```

## 🚀 Getting Started

### Prerequisites
- Intel RealSense D400 series depth camera
- NVIDIA GPU (recommended for better performance)
- Python 3.8+
- Leo Rover platform (or compatible robotic platform)

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/prathapselvakumar/Leo-Rover.git
   cd Leo-Rover/Coding/Object\ Detection
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Hardware Setup**
   - Connect the Intel RealSense camera
   - Ensure proper power supply to all components
   - Verify camera recognition with RealSense Viewer

## 🛠️ Usage

### Object Detection
```bash
python object_shape_color_depth.py
```

### Hand-Eye Calibration
```bash
python hand_eye_calibration.py
```

## 📊 Outputs
The system provides real-time visual feedback including:
- 3D object detection with bounding boxes
- Color-coded object classification
- Depth map visualization
- Edge detection results

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact
For questions or feedback, please open an issue in the repository or contact the project maintainers.

## 📚 Resources
- [Intel RealSense Documentation](https://dev.intelrealsense.com/docs)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Leo Rover Documentation](https://docs.leorover.tech/)
