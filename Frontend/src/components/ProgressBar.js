import React from 'react';

const ProgressBar = ({ progress = 0, indeterminate = false, className = '' }) => {
  return (
    <div className={`progress ${className}`}>
      <div 
        className={`bar ${indeterminate ? 'indeterminate' : ''}`}
        style={{ width: `${progress}%` }}
      ></div>
    </div>
  );
};

export default ProgressBar;
