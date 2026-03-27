# Camera PDF Generator

A JavaFX application that processes camera images using AI detection and generates PDF reports.

## Project Structure

```
demo/
├── src/main/java/com/example/demo/     # JavaFX Frontend
│   ├── HelloApplication.java           # Main application class
│   ├── HelloController.java            # FXML controller
│   └── module-info.java               # Module configuration
├── src/main/backend/                   # Python Flask Backend
│   ├── app.py                         # Main Flask application
│   ├── model.py                       # AI model wrapper
│   ├── requirements.txt               # Python dependencies
│   ├── run_backend.py                 # Backend runner script
│   └── models/                        # AI model files
│       └── best.pt                    # YOLO model file
└── pom.xml                            # Maven configuration
```

## Prerequisites

- Java 17 or higher
- Maven 3.6 or higher
- Python 3.8 or higher
- pip (Python package manager)

## Setup Instructions

### 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd src/main/backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your YOLO model file (`best.pt`) in the `models/` directory

4. Start the backend server:
   ```bash
   python run_backend.py
   ```

   The backend will be available at `http://127.0.0.1:5000`

### 2. Frontend Setup

1. In the project root directory, compile and run the JavaFX application:
   ```bash
   mvn clean compile
   mvn javafx:run
   ```

   Or using the Maven wrapper:
   ```bash
   ./mvnw clean compile
   ./mvnw javafx:run
   ```

## Usage

1. **Upload Image**: Click "Upload Image" to select a camera image file
2. **Fill Details**: Enter camera model, location, and additional notes
3. **Generate PDF**: Click "Generate PDF" to create a report with:
   - AI-detected annotations on the image
   - Camera details and notes
   - Professional PDF formatting

## Features

- **AI Image Processing**: Uses YOLO model for object detection and annotation
- **PDF Generation**: Creates professional PDF reports with Apache PDFBox
- **Modern UI**: Clean JavaFX interface with FXML
- **Cross-platform**: Works on Windows, macOS, and Linux

## Troubleshooting

### Backend Issues
- Ensure Python dependencies are installed: `pip install -r requirements.txt`
- Check that the model file exists in `models/best.pt`
- Verify the backend is running on port 5000

### Frontend Issues
- Ensure Java 17+ is installed and configured
- Check Maven dependencies: `mvn dependency:resolve`
- Verify JavaFX modules are properly configured

### Connection Issues
- Ensure backend is running before starting the frontend
- Check firewall settings for port 5000
- Verify the backend URL in `HelloController.java` (default: `http://127.0.0.1:5000`)

## Dependencies

### Java Dependencies
- JavaFX 17.0.6
- Apache PDFBox 2.0.31
- OkHttp 4.12.0
- Gson 2.10.1

### Python Dependencies
- Flask 2.0+
- Flask-CORS
- Ultralytics (YOLO)
- Pillow (PIL)
- NumPy
- ReportLab

## License

This project is for educational and demonstration purposes.

