import React from 'react';

const StatusBadge = ({ status, className = '' }) => {
  const getStatusClass = (status) => {
    switch (status) {
      case 'RUNNING': return 'RUNNING';
      case 'PENDING': 
      case 'STARTING': return 'PENDING';
      case 'SUCCEEDED': return 'SUCCEEDED';
      case 'FAILED': return 'FAILED';
      default: return 'PENDING';
    }
  };

  return (
    <div className={`status-badge ${getStatusClass(status)} ${className}`}>
      <span className="dot"></span>
      <span>{status}</span>
    </div>
  );
};

export default StatusBadge;
