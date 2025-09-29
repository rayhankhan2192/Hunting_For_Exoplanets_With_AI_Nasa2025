const API_BASE = process.env.REACT_APP_API_BASE || "http://203.190.12.138:8080";

class ApiService {
  constructor() {
    this.baseURL = API_BASE;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      
      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Merge API
  async getMergedFiles() {
    return this.request('/api/merge');
  }

  async mergeFiles(fileA, fileB, options = {}) {
    const formData = new FormData();
    formData.append('file_a', fileA);
    formData.append('file_b', fileB);
    
    if (options.dedupe !== undefined) {
      formData.append('dedupe', options.dedupe);
    }
    if (options.output_name) {
      formData.append('output_name', options.output_name);
    }

    return this.request('/api/merge', {
      method: 'POST',
      headers: {}, // Let browser set Content-Type for FormData
      body: formData,
    });
  }

  // Training API
  async startTraining(file, satellite, model) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('satellite', satellite);
    formData.append('model', model);

    return this.request('/api/train', {
      method: 'POST',
      headers: {}, // Let browser set Content-Type for FormData
      body: formData,
    });
  }

  async getTrainingStatus(jobId) {
    return this.request(`/api/train/${jobId}/status`);
  }

  async getTrainingLogs(jobId, tail = 400) {
    return this.request(`/api/train/${jobId}/logs?tail=${tail}`);
  }

  // Prediction API
  async predict(file, satellite, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('satellite', satellite);
    
    if (options.from_csv_range !== undefined) {
      formData.append('from_csv_range', options.from_csv_range);
    }
    if (options.to_csv_range !== undefined) {
      formData.append('to_csv_range', options.to_csv_range);
    }

    return this.request('/api/predict', {
      method: 'POST',
      headers: {}, // Let browser set Content-Type for FormData
      body: formData,
    });
  }

  // Uploads API
  async getUploads() {
    return this.request('/api/uploads');
  }

  // Merge and Train API (advanced)
  async mergeAndTrain(payload) {
    return this.request('/api/merge-train', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }
}

export default new ApiService();
