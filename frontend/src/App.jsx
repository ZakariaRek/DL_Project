// ========================================
// 🍊 REACT FRONTEND - ORANGE DISEASE DETECTION
// App.jsx - Main Application Component (Updated)
// ========================================

import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState('image');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
      
      // Create preview URL
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
      
      // Detect file type
      if (selectedFile.type.startsWith('image/')) {
        setFileType('image');
      } else if (selectedFile.type.startsWith('video/')) {
        setFileType('video');
      }
    }
  };

  // Handle file upload and analysis
  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const endpoint = fileType === 'image' 
        ? `${API_URL}/api/classify-image`
        : `${API_URL}/api/analyze-video`;

      const response = await axios.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  // Clear all states
  const handleReset = () => {
    setFile(null);
    setFileType('image');
    setResult(null);
    setError(null);
    setPreviewUrl(null);
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="logo">
              <span className="orange-icon">🍊</span>
              <h1>Orange Disease Detection</h1>
            </div>
            <nav className="nav">
              <a href="#about">About</a>
              <a href="#features">Features</a>
              <a href="#upload">Upload</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <div className="hero-content">
            <h2 className="hero-title">
              AI-Powered Orange Disease Detection
            </h2>
            <p className="hero-subtitle">
              Upload an image or video to detect and classify orange diseases using 
              state-of-the-art deep learning technology
            </p>
            <div className="hero-stats">
              <div className="stat">
                <div className="stat-number">4</div>
                <div className="stat-label">Disease Classes</div>
              </div>
              <div className="stat">
                <div className="stat-number">95%</div>
                <div className="stat-label">Accuracy</div>
              </div>
              <div className="stat">
                <div className="stat-number">Real-time</div>
                <div className="stat-label">Processing</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="about">
        <div className="container">
          <h2 className="section-title">About the Project</h2>
          <div className="about-content">
            <div className="about-card">
              <h3>🎯 Purpose</h3>
              <p>
                This system helps farmers and agricultural professionals quickly identify 
                orange diseases, enabling early intervention and reducing crop losses.
              </p>
            </div>
            <div className="about-card">
              <h3>🔬 Technology</h3>
              <p>
                Built with PyTorch and YOLOv8, combining ShuffleNet V2 for efficient 
                disease classification and YOLO for real-time orange detection in videos.
              </p>
            </div>
            <div className="about-card">
              <h3>🏥 Disease Classes</h3>
              <p>
                Detects 4 conditions: Blackspot, Canker, Fresh (healthy), and Grenning. 
                Each classification includes confidence scores and detailed analysis.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <div className="container">
          <h2 className="section-title">Key Features</h2>
          <div className="features-grid">
            <div className="feature">
              <div className="feature-icon">📸</div>
              <h3>Image Analysis</h3>
              <p>Upload a single orange image for instant disease classification</p>
            </div>
            <div className="feature">
              <div className="feature-icon">🎥</div>
              <h3>Video Processing</h3>
              <p>Analyze entire video footage to detect multiple oranges</p>
            </div>
            <div className="feature">
              <div className="feature-icon">📊</div>
              <h3>Detailed Reports</h3>
              <p>Get comprehensive statistics and health scores</p>
            </div>
            <div className="feature">
              <div className="feature-icon">⚡</div>
              <h3>Fast & Accurate</h3>
              <p>Real-time processing with 95%+ accuracy</p>
            </div>
            <div className="feature">
              <div className="feature-icon">💡</div>
              <h3>Easy to Use</h3>
              <p>Simple drag-and-drop interface</p>
            </div>
            <div className="feature">
              <div className="feature-icon">🔒</div>
              <h3>Secure</h3>
              <p>Your data is processed securely and not stored</p>
            </div>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section id="upload" className="upload-section">
        <div className="container">
          <h2 className="section-title">Analyze Your Oranges</h2>
          
          <div className="upload-container">
            {/* File Upload */}
            <div className="upload-card">
              <div className="upload-area">
                <input
                  type="file"
                  id="file-input"
                  accept="image/*,video/*"
                  onChange={handleFileChange}
                  className="file-input"
                />
                <label htmlFor="file-input" className="file-label">
                  {previewUrl ? (
                    <div className="preview">
                      {fileType === 'image' ? (
                        <img src={previewUrl} alt="Preview" />
                      ) : (
                        <video src={previewUrl} controls />
                      )}
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <div className="upload-icon">📤</div>
                      <p>Click to upload or drag and drop</p>
                      <span>Images (JPG, PNG) or Videos (MP4, AVI, MOV)</span>
                    </div>
                  )}
                </label>
              </div>

              {file && (
                <div className="file-info">
                  <p><strong>File:</strong> {file.name}</p>
                  <p><strong>Size:</strong> {(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  <p><strong>Type:</strong> {fileType.toUpperCase()}</p>
                </div>
              )}

              <div className="button-group">
                <button
                  onClick={handleAnalyze}
                  disabled={!file || loading}
                  className="btn btn-primary"
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing...
                    </>
                  ) : (
                    'Analyze'
                  )}
                </button>
                <button
                  onClick={handleReset}
                  disabled={loading}
                  className="btn btn-secondary"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Results */}
            {error && (
              <div className="result-card error-card">
                <h3>❌ Error</h3>
                <p>{error}</p>
              </div>
            )}

            {result && fileType === 'image' && (
              <ImageResult result={result} />
            )}

            {result && fileType === 'video' && (
              <VideoResult result={result} />
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>© 2024 Orange Disease Detection System</p>
          <p>Built with ❤️ using PyTorch, FastAPI, and React</p>
        </div>
      </footer>
    </div>
  );
}

// ========================================
// IMAGE RESULT COMPONENT
// ========================================
function ImageResult({ result }) {
  const getHealthStatus = (disease, confidence) => {
    if (disease === 'fresh') return { emoji: '✅', label: 'Healthy', color: '#10b981' };
    if (confidence > 90) return { emoji: '🔴', label: 'Severe', color: '#ef4444' };
    if (confidence > 75) return { emoji: '⚠️', label: 'Moderate', color: '#f59e0b' };
    return { emoji: '⚠️', label: 'Mild', color: '#3b82f6' };
  };

  const status = getHealthStatus(result.disease, result.confidence);

  return (
    <div className="result-card">
      <h3>🍊 Analysis Result</h3>
      
      <div className="result-main">
        <div className="disease-badge" style={{ borderColor: status.color }}>
          <span className="disease-emoji">{status.emoji}</span>
          <div>
            <div className="disease-name">{result.disease.toUpperCase()}</div>
            <div className="disease-status" style={{ color: status.color }}>
              {status.label}
            </div>
          </div>
        </div>
        
        <div className="confidence-display">
          <div className="confidence-label">Confidence</div>
          <div className="confidence-value">{result.confidence.toFixed(1)}%</div>
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ 
                width: `${result.confidence}%`,
                backgroundColor: status.color 
              }}
            />
          </div>
        </div>
      </div>

      <div className="probabilities">
        <h4>All Probabilities</h4>
        <div className="prob-list">
          {Object.entries(result.probabilities).map(([disease, prob]) => (
            <div key={disease} className="prob-item">
              <span className="prob-name">{disease}</span>
              <div className="prob-bar-container">
                <div 
                  className="prob-bar" 
                  style={{ width: `${prob}%` }}
                />
              </div>
              <span className="prob-value">{prob.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="result-meta">
        <p><strong>Analysis Time:</strong> {new Date(result.timestamp).toLocaleString()}</p>
      </div>
    </div>
  );
}

// ========================================
// VIDEO RESULT COMPONENT (UPDATED)
// ========================================
function VideoResult({ result }) {
  const getHealthLevel = (score) => {
    if (score >= 80) return { label: 'EXCELLENT', color: '#10b981', emoji: '✅' };
    if (score >= 60) return { label: 'GOOD', color: '#3b82f6', emoji: '👍' };
    if (score >= 40) return { label: 'FAIR', color: '#f59e0b', emoji: '⚠️' };
    return { label: 'POOR', color: '#ef4444', emoji: '🔴' };
  };

  const health = getHealthLevel(result.health_score);

  return (
    <div className="result-card">
      <h3>🎥 Video Analysis Report</h3>

      {/* Summary */}
      <div className="video-summary">
        <div className="summary-item">
          <div className="summary-label">Total Oranges</div>
          <div className="summary-value">{result.total_oranges}</div>
        </div>
        <div className="summary-item">
          <div className="summary-label">Frames Analyzed</div>
          <div className="summary-value">{result.processed_frames}</div>
        </div>
        <div className="summary-item">
          <div className="summary-label">Duration</div>
          <div className="summary-value">
            {result.video_info.duration.toFixed(1)}s
          </div>
        </div>
      </div>

      {/* Health Score */}
      <div className="health-score">
        <h4>Orchard Health Score</h4>
        <div className="health-display">
          <div className="health-circle" style={{ borderColor: health.color }}>
            <span className="health-emoji">{health.emoji}</span>
            <span className="health-percentage">{result.health_score.toFixed(1)}%</span>
          </div>
          <div className="health-status" style={{ color: health.color }}>
            {health.label}
          </div>
        </div>
      </div>

      {/* Disease Distribution */}
      <div className="disease-stats">
        <h4>Disease Distribution</h4>
        {Object.entries(result.statistics).map(([disease, stats]) => (
          <div key={disease} className="disease-stat">
            <div className="stat-header">
              <span className="stat-disease">{disease.toUpperCase()}</span>
              <span className="stat-count">
                {stats.count} ({stats.percentage.toFixed(1)}%)
              </span>
            </div>
            <div className="stat-bar-container">
              <div 
                className="stat-bar" 
                style={{ 
                  width: `${stats.percentage}%`,
                  backgroundColor: getDiseaseColor(disease)
                }}
              />
            </div>
            <div className="stat-details">
              <span>Avg Confidence: {stats.avg_confidence.toFixed(1)}%</span>
              <span>Range: {stats.min_confidence.toFixed(1)}% - {stats.max_confidence.toFixed(1)}%</span>
            </div>
          </div>
        ))}
      </div>

      {/* Detection Timeline */}
      {result.detections.length > 0 && (
        <div className="detections-table">
          <h4>Detection Timeline (First 10)</h4>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Frame</th>
                <th>Disease</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {result.detections.slice(0, 10).map((detection) => (
                <tr key={detection.id}>
                  <td>{detection.time_sec.toFixed(1)}s</td>
                  <td>{detection.frame}</td>
                  <td>
                    <span 
                      className="disease-tag"
                      style={{ 
                        backgroundColor: getDiseaseColor(detection.disease) + '20',
                        color: getDiseaseColor(detection.disease)
                      }}
                    >
                      {detection.disease}
                    </span>
                  </td>
                  <td>{detection.confidence.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          {result.detections.length > 10 && (
            <p className="table-footer">
              + {result.detections.length - 10} more detections
            </p>
          )}
        </div>
      )}

      {/* ANALYSIS VISUALIZATION & ANNOTATED VIDEO - NEW SECTION */}
      <div className="media-results">
        <h4>📊 Analysis Results</h4>
        
        {/* Analysis Visualization Image */}
        {result.analysis_image_url && (
          <div className="media-section">
            <h5>Statistical Analysis</h5>
            <div className="media-preview">
              <img 
                src={`${API_URL}${result.analysis_image_url}`} 
                alt="Analysis Results"
                className="analysis-image"
              />
            </div>
            <a 
              href={`${API_URL}${result.analysis_image_url}`}
              download="analysis_results.png"
              className="download-btn"
            >
              📥 Download Analysis Image
            </a>
          </div>
        )}

        {/* Annotated Video */}
        {result.annotated_video_url && (
          <div className="media-section">
            <h5>Annotated Video</h5>
            <div className="media-preview">
              <video 
                src={`${API_URL}${result.annotated_video_url}`}
                controls
                className="annotated-video"
              >
                Your browser does not support the video tag.
              </video>
            </div>
            <a 
              href={`${API_URL}${result.annotated_video_url}`}
              download="annotated_video.mp4"
              className="download-btn"
            >
              📥 Download Annotated Video
            </a>
          </div>
        )}
      </div>

      <div className="result-meta">
        <p><strong>Analysis Time:</strong> {new Date(result.timestamp).toLocaleString()}</p>
      </div>
    </div>
  );
}

// Helper function
function getDiseaseColor(disease) {
  const colors = {
    'blackspot': '#ef4444',
    'canker': '#f59e0b',
    'fresh': '#10b981',
    'grenning': '#3b82f6'
  };
  return colors[disease] || '#6b7280';
}

export default App;