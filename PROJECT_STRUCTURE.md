# Exoplanet AI System - Project Structure

## Overview
This project has been successfully converted from HTML/JavaScript to a modern React application while preserving all original functionality and design.

## Project Structure

```
Hunting_For_Exoplanets_With_AI_Nasa2025/
├── Frontend/                          # React Application
│   ├── public/
│   │   └── index.html                 # Main HTML template
│   ├── src/
│   │   ├── components/                # Reusable React components
│   │   │   ├── Navigation.js          # Main navigation component
│   │   │   ├── LoadingSpinner.js      # Loading indicator
│   │   │   ├── Toast.js               # Notification component
│   │   │   ├── StatusBadge.js         # Status display component
│   │   │   └── ProgressBar.js         # Progress visualization
│   │   ├── pages/                     # Page components
│   │   │   ├── Home.js                # Landing page
│   │   │   ├── MergeTrain.js          # Merge & Train (from index.html)
│   │   │   ├── StartTraining.js       # Quick training (from train.html)
│   │   │   ├── TrainingProgress.js    # Training status (from training.html)
│   │   │   ├── Prediction.js          # Predictions (from t.html)
│   │   │   ├── MergeTrainAdvanced.js  # Advanced tools (from m.html)
│   │   │   └── Result.js              # Results display (from result.html)
│   │   ├── services/
│   │   │   └── api.js                 # API service layer
│   │   ├── App.js                     # Main app component with routing
│   │   ├── index.js                   # React entry point
│   │   └── index.css                  # Global styles
│   ├── package.json                   # Dependencies and scripts
│   ├── README.md                      # Frontend documentation
│   └── .gitignore                     # Git ignore rules
├── Data/                              # Original data files
│   └── k2.txt
├── DataSet/                           # Original HTML files (preserved)
│   ├── index.html                     # → MergeTrain.js
│   ├── train.html                     # → StartTraining.js
│   ├── training.html                  # → TrainingProgress.js
│   ├── result.html                    # → Result.js
│   ├── t.html                         # → Prediction.js
│   ├── m.html                         # → MergeTrainAdvanced.js
│   ├── KOI/                           # KOI dataset files
│   ├── K2/                            # K2 dataset files
│   └── TOI/                           # TOI dataset files
├── main/                              # Python backend
│   ├── model.py                       # ML models
│   ├── train.py                       # Training scripts
│   ├── preprocess.py                  # Data preprocessing
│   └── ...
├── server/                            # Django backend
│   ├── myapp/                         # Django app
│   ├── server/                        # Django settings
│   └── ...
└── README.md                          # Main project documentation
```

## Page Mappings

| Original HTML | React Component | Route | Description |
|---------------|-----------------|-------|-------------|
| `index.html` | `MergeTrain.js` | `/merge-train` | Merge CSV files and train models |
| `train.html` | `StartTraining.js` | `/start-training` | Quick training interface |
| `training.html` | `TrainingProgress.js` | `/training-progress` | Training progress with live updates |
| `result.html` | `Result.js` | `/result` | Training results display |
| `t.html` | `Prediction.js` | `/prediction` | Make predictions on new data |
| `m.html` | `MergeTrainAdvanced.js` | `/merge-train-advanced` | Advanced merge and train tools |
| - | `Home.js` | `/` | New attractive home page |

## Key Features Preserved

### 1. **Merge & Train Functionality**
- ✅ View existing merged CSV files
- ✅ Merge two new CSV files with options
- ✅ Preview merged data
- ✅ Select satellite (KOI/K2/TOI) and model type
- ✅ Start training with progress tracking

### 2. **Training Progress**
- ✅ Real-time status updates
- ✅ Progress bar with animations
- ✅ Performance metrics display
- ✅ Confusion matrix visualization
- ✅ Model download links
- ✅ Live logs display

### 3. **Prediction System**
- ✅ CSV file upload with drag & drop
- ✅ Configurable prediction parameters
- ✅ Results table with sorting and pagination
- ✅ Search and filter capabilities
- ✅ Class distribution visualization
- ✅ Download prediction results

### 4. **Advanced Tools**
- ✅ File selection interface
- ✅ Live job status monitoring
- ✅ Real-time logs with tail functionality
- ✅ Comprehensive training options
- ✅ JSON status display

## Technical Improvements

### 1. **Modern React Architecture**
- Component-based structure
- React Hooks for state management
- React Router for navigation
- Service layer for API calls

### 2. **Enhanced User Experience**
- Responsive design for all screen sizes
- Loading states and error handling
- Toast notifications for user feedback
- Smooth animations and transitions

### 3. **Code Organization**
- Separation of concerns
- Reusable components
- Centralized API service
- Consistent styling with CSS variables

### 4. **Development Experience**
- Hot reloading
- ESLint for code quality
- Modern JavaScript (ES6+)
- Comprehensive documentation

## API Integration

The React frontend communicates with the existing Django backend through:

- **Merge API**: `/api/merge` - File merging operations
- **Training API**: `/api/train` - Model training
- **Status API**: `/api/train/{job_id}/status` - Training progress
- **Logs API**: `/api/train/{job_id}/logs` - Training logs
- **Prediction API**: `/api/predict` - Make predictions
- **Uploads API**: `/api/uploads` - File management

## Getting Started

### Frontend Development
```bash
cd Frontend
npm install
npm start
```

### Backend (Existing)
```bash
cd server
python manage.py runserver
```

## Design Preservation

The React application maintains the original dark theme design with:
- Same color scheme and CSS variables
- Identical layout and component structure
- Preserved animations and transitions
- Consistent typography and spacing
- Original responsive breakpoints

## Navigation Flow

1. **Home** (`/`) → Overview and feature access
2. **Merge & Train** (`/merge-train`) → Merge files and train models
3. **Start Training** (`/start-training`) → Quick training interface
4. **Training Progress** (`/training-progress`) → Live training status
5. **Prediction** (`/prediction`) → Make predictions on new data
6. **Advanced Tools** (`/merge-train-advanced`) → Comprehensive tools

The conversion successfully maintains all original functionality while providing a modern, maintainable React codebase.
