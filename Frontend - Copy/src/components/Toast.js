import React, { useState, useEffect } from 'react';

const Toast = ({ message, type = 'success', duration = 3000, onClose }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (message) {
      setVisible(true);
      const timer = setTimeout(() => {
        setVisible(false);
        if (onClose) onClose();
      }, duration);
      return () => clearTimeout(timer);
    }
  }, [message, duration, onClose]);

  if (!message) return null;

  return (
    <div 
      className={`toast ${visible ? 'show' : ''}`}
      style={{
        borderColor: type === 'error' ? 'rgba(239,68,68,.35)' : 'rgba(16,185,129,.35)'
      }}
    >
      {message}
    </div>
  );
};

export default Toast;
