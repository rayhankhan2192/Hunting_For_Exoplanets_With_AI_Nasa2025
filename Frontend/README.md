# Exoplanet AI System - React Frontend

This is the React frontend for the Exoplanet AI System, designed for NASA Space Apps Challenge 2025. The system helps discover exoplanets using advanced machine learning models trained on Kepler, K2, and TESS data.

## Features

- **Merge & Train**: Combine multiple CSV datasets and train ML models
- **Quick Training**: Upload single CSV files for rapid model training
- **Predictions**: Use trained models to predict exoplanet classifications
- **Advanced Tools**: Comprehensive merging tools and model management
- **Real-time Progress**: Live training progress tracking with visualizations
- **Responsive Design**: Modern dark theme with mobile-friendly interface

## Project Structure

```
Frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Navigation.js
│   │   ├── LoadingSpinner.js
│   │   ├── Toast.js
│   │   ├── StatusBadge.js
│   │   └── ProgressBar.js
│   ├── pages/
│   │   ├── Home.js
│   │   ├── MergeTrain.js
│   │   ├── StartTraining.js
│   │   ├── TrainingProgress.js
│   │   ├── Prediction.js
│   │   ├── MergeTrainAdvanced.js
│   │   └── Result.js
│   ├── services/
│   │   └── api.js
│   ├── App.js
│   ├── index.js
│   └── index.css
├── package.json
└── README.md
```

## Getting Started

### Prerequisites

- Node.js (version 14 or higher)
- npm or yarn

### Installation

1. Navigate to the Frontend directory:
   ```bash
   cd Frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Building for Production

```bash
npm run build
```

This builds the app for production to the `build` folder.

## API Configuration

The frontend is configured to connect to the backend API at `http://203.190.12.138:8080` by default. You can modify this in the `api.js` service file or set the `REACT_APP_API_BASE` environment variable.

## Pages

### Home (`/`)
- Welcome page with feature overview
- Navigation to all system features
- Information about supported models and datasets

### Merge & Train (`/merge-train`)
- View existing merged CSV files
- Merge two new CSV files
- Preview merged data
- Start training with selected parameters

### Start Training (`/start-training`)
- Quick training interface
- Upload single CSV file
- Select satellite and model type
- Redirects to progress page

### Training Progress (`/training-progress`)
- Real-time training status
- Progress visualization
- Performance metrics
- Confusion matrix display
- Model download links

### Prediction (`/prediction`)
- Upload CSV for predictions
- Configure prediction parameters
- View results in sortable table
- Download prediction results
- Search and filter capabilities

### Advanced Tools (`/merge-train-advanced`)
- Advanced file selection interface
- Live job status monitoring
- Real-time logs display
- Comprehensive training options

## Supported Models

- **XGBoost (XGB)**: Gradient boosting framework
- **Random Forest (RF)**: Ensemble learning method
- **Decision Tree (DT)**: Simple tree-based model
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Logistic Regression**: Linear classification model

## Supported Datasets

- **KOI**: Kepler Object of Interest
- **K2**: Kepler's second mission
- **TOI**: TESS Object of Interest

## Technologies Used

- **React 18**: Modern React with hooks
- **React Router**: Client-side routing
- **CSS3**: Custom styling with CSS variables
- **Fetch API**: HTTP requests
- **ES6+**: Modern JavaScript features

## Development

The project uses Create React App for development and building. Key features:

- Hot reloading during development
- ESLint for code quality
- Production optimizations
- Responsive design
- Dark theme optimized for data visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the NASA Space Apps Challenge 2025.
