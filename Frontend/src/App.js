import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import MergeTrain from './pages/MergeTrain';
import StartTraining from './pages/StartTraining';
import TrainingProgress from './pages/TrainingProgress';
import Prediction from './pages/Prediction';
import MergeTrainAdvanced from './pages/MergeTrainAdvanced';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/merge-train" element={<MergeTrain />} />
          <Route path="/start-training" element={<StartTraining />} />
          <Route path="/training-progress" element={<TrainingProgress />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/merge-train-advanced" element={<MergeTrainAdvanced />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
