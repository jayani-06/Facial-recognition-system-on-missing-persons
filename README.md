# Facial Recognition System for Missing Persons v1.0

A robust AI-powered facial recognition system designed to trace missing persons and identify unidentified bodies using advanced computer vision techniques.

## Overview

This system uses a combination of face detection, image enhancement, and multi-modal feature extraction to match faces against a database of missing persons and unidentified bodies. It provides confidence scores for human verification and supports various lighting and image quality conditions.

## Features

### Core Capabilities
- **Face Detection & Alignment**: Automatically detects faces and aligns them based on eye positions for consistent feature extraction
- **Image Enhancement**: CLAHE-based lighting correction, detail enhancement, and sharpening to handle poor quality images
- **Multi-Modal Feature Extraction**:
  - Color Histograms (HSV) - Captures skin tone and color patterns
  - HOG (Histogram of Oriented Gradients) - Detects structural facial features
  - LBP (Local Binary Patterns) - Captures texture and thermal patterns
- **Similarity Matching**: Cosine similarity-based comparison with configurable thresholds
- **Database Management**: Local persistence of facial features and metadata

### Use Cases
- Register missing persons with metadata (name, age, last seen location)
- Register unidentified bodies with case information
- Search and match faces with confidence scoring
- View database statistics and entries

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the system:
```bash
python facial_recognition_system.py
```

## Usage

### Main Menu Options

#### 1. Register Missing Person
Register a missing person's photo into the database.

**Required:**
- Image path to the person's photo

**Optional:**
- Name
- Age
- Last seen location

**Example:**
```
Enter image path: /path/to/missing_person.jpg
Enter person's name: John Doe
Enter age (optional): 28
Enter last seen location (optional): Downtown Mall
```

#### 2. Register Unidentified Body
Register an unidentified body with case information.

**Required:**
- Image path to the photo
- Case ID

**Optional:**
- Location found
- Date found

**Example:**
```
Enter image path: /path/to/unidentified.jpg
Enter case ID: UB-2024-001
Enter location found (optional): Forest Area, Sector 12
Enter date found (optional): 2024-10-15
```

#### 3. Search for Match
Search for matches against the database.

**Parameters:**
- Query image path
- Number of top matches to show (default: 5)
- Minimum confidence threshold 0-1 (default: 0.65)

**Example:**
```
Enter query image path: /path/to/query.jpg
Number of top matches to show (default 5): 5
Minimum confidence threshold 0-1 (default 0.65): 0.65
```

**Output:**
- Visual display of query image and top matches
- Confidence scores as percentages
- Metadata for each match (name, type, registration date)
- Additional information (age, location, case ID, etc.)

#### 4. View Database Stats
Displays:
- Total entries in database
- Number of missing persons
- Number of unidentified bodies

#### 5. Exit
Saves and closes the system.

## Technical Details

### Image Processing Pipeline

1. **Face Detection**
   - Uses Haar Cascade classifiers for face and eye detection
   - Aligns faces based on eye positions to normalize orientation
   - Extracts face region for feature extraction

2. **Image Enhancement**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction
   - Detail enhancement to bring out facial features
   - Sharpening filter to improve edge definition

3. **Feature Extraction**
   - **Color Histogram**: 170-dimensional HSV histogram (50 + 60 + 60 bins)
   - **HOG Features**: Structural gradient-based features in 8x8 pixel cells
   - **LBP Features**: Texture patterns using uniform local binary patterns
   - All features are normalized and concatenated into a single feature vector

4. **Matching Algorithm**
   - Cosine similarity comparison between query and database features
   - Results ranked by similarity score
   - Configurable threshold for minimum match confidence
   - Returns top-K matches for human verification

### Database Structure

The system stores data in `face_database/` directory:
- `database.pkl`: Serialized database containing features and metadata

**Metadata fields:**
- name: Person's name or case ID
- type: 'missing' or 'unidentified'
- registered_date: ISO format timestamp
- image_path: Original image location
- additional_info: Dictionary of custom fields (age, location, case details, etc.)

## Configuration

### Adjustable Parameters

**In FaceEnhancer class:**
- CLAHE clip limit: Controls contrast enhancement (default: 2.0)
- CLAHE tile grid size: Granularity of enhancement (default: 8x8)

**In FaceDetector class:**
- Scale factor: Face detection sensitivity (default: 1.1)
- Min neighbors: Detection confidence (default: 5)
- Min size: Minimum face size in pixels (default: 30x30)

**In FeatureExtractor class:**
- Target size: Normalized face size for feature extraction (default: 128x128)
- HOG orientations: Gradient histogram bins (default: 9)
- LBP radius: Local pattern radius (default: 3)

**In search_face method:**
- top_k: Number of results to return (default: 5)
- threshold: Minimum similarity score 0-1 (default: 0.65)

## Performance Considerations

### Accuracy
- Works best with frontal face images
- Requires at least one eye to be visible for alignment
- Handles various lighting conditions through CLAHE enhancement
- Confidence scores above 75% indicate strong matches
- Scores between 65-75% require careful human verification
- Scores below 65% are typically false matches

### Speed
- Registration: ~0.5-1 second per image
- Search: ~0.1-0.5 seconds for databases up to 1000 entries
- Scales linearly with database size

### Storage
- Each entry requires ~50KB (features + metadata)
- 1000 entries ≈ 50MB storage

## Limitations

1. **Face Detection**: May fail with:
   - Extreme angles (side profiles)
   - Heavy occlusions (masks, sunglasses)
   - Very low resolution images (<50x50 pixels)

2. **Matching Accuracy**: Reduced by:
   - Significant time gap between photos (aging)
   - Major changes in appearance (facial hair, hairstyle)
   - Different facial expressions
   - Poor image quality

3. **Database Size**: Performance degrades with databases >10,000 entries. Consider implementing FAISS for larger datasets.

## Future Enhancements (v2.0 Roadmap)

- Integration with deep learning models (FaceNet, ArcFace)
- FAISS indexing for large-scale databases
- Web interface with REST API
- Batch processing capabilities
- Age progression/regression models
- Multi-face image support
- Export reports in PDF format
- Integration with law enforcement databases

## Troubleshooting

**No face detected:**
- Ensure face is clearly visible and frontal
- Check image quality and resolution
- Try adjusting face detector parameters

**Low confidence scores:**
- Verify image quality of both query and database images
- Consider lowering threshold for preliminary matches
- Review feature extraction parameters

**Slow performance:**
- Database may be too large - consider optimization
- Check system resources (CPU, RAM)
- Reduce image resolution if too high

## License

This software is intended for law enforcement and humanitarian purposes only. Use responsibly and in compliance with local privacy and data protection laws.

## Version History

### v1.0 (Current)
- Initial release
- Face detection and alignment
- CLAHE-based image enhancement
- Multi-modal feature extraction (Color, HOG, LBP)
- Cosine similarity matching
- Interactive CLI interface
- Local database persistence

## Support

For issues or questions, ensure you have:
- Python version
- Installed package versions
- Sample images (if applicable)
- Error messages or unexpected behavior description
