import { useState, ChangeEvent, useEffect, useRef } from 'react';

interface AnalysisResult {
  chunk_index: number;
  text_chunk: string;
  analysis: string;
  is_complete?: boolean;
}

interface GoogleDocEditorProps {
  text: string;
  onTextChange: (text: string) => void;
  results: AnalysisResult[];
  isLoading: boolean;
}

const GoogleDocEditor: React.FC<GoogleDocEditorProps> = ({ 
  text, 
  onTextChange, 
  results, 
  isLoading 
}) => {
  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    onTextChange(e.target.value);
  };

  // Calculate the position of each note based on the chunk index
  const getCommentPosition = (index: number) => {
    // This is a simple approximation, in a real app you'd calculate
    // the exact position based on line numbers and text measurements
    return { top: `${(index * 20) + 10}%` };
  };

  return (
    <div className="google-doc-container">
      <div className="doc-content">
        <div className="doc-header">
          <div className="doc-title">Untitled document</div>
          <div className="toolbar">
            <div className="toolbar-group">
              <div className="toolbar-item">File</div>
              <div className="toolbar-item">Edit</div>
              <div className="toolbar-item">View</div>
              <div className="toolbar-item">Insert</div>
              <div className="toolbar-item">Format</div>
              <div className="toolbar-item">Tools</div>
              <div className="toolbar-item">Extensions</div>
              <div className="toolbar-item">Help</div>
            </div>
            <div className="toolbar-group formatting">
              <div className="toolbar-icon">B</div>
              <div className="toolbar-icon">I</div>
              <div className="toolbar-icon">U</div>
              <div className="toolbar-separator"></div>
              <div className="toolbar-icon">A</div>
              <div className="toolbar-icon">A</div>
              <div className="toolbar-separator"></div>
              <div className="toolbar-icon">⫶</div>
              <div className="toolbar-icon">⫷</div>
            </div>
          </div>
        </div>
        <div className="doc-body">
          <textarea
            className="doc-textarea"
            value={text}
            onChange={handleTextChange}
            placeholder="Type your text here..."
            disabled={isLoading}
          />
        </div>
      </div>
      <div className="doc-comments">
        <div className="comments-header">
          <div className="comments-title">Comments</div>
          {isLoading && <div className="comments-status">Analyzing...</div>}
        </div>
        {results.length > 0 && (
          <div className="comments-container">
            {[...results].sort((a, b) => a.chunk_index - b.chunk_index).map((result) => (
              <div 
                key={result.chunk_index} 
                className="comment-bubble"
                style={getCommentPosition(result.chunk_index)}
              >
                <div className="comment-content">
                  <div className="comment-header">
                    <div className="comment-avatar">AI</div>
                    <div className="comment-author">AI Assistant</div>
                  </div>
                  <div className="comment-text">
                    {result.analysis}
                    {!result.is_complete && result.analysis && (
                      <span className="typing-indicator">▌</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        {!isLoading && results.length === 0 && (
          <div className="no-comments">
            No comments yet. Click "Analyze Text" to generate comments.
          </div>
        )}
      </div>
    </div>
  );
};

export default GoogleDocEditor; 