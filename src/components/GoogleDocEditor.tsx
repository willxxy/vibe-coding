import { useState, ChangeEvent, useEffect, useRef, useMemo, memo, useCallback } from 'react';

interface AnalysisResult {
  chunk_index: number;
  text_chunk: string;
  analysis: string;
  is_complete?: boolean;
  session_id?: number;
}

interface GlobalAnalysis {
  tone: string;
  subject_matter: string;
  context_summary: string;
  session_id?: number;
}

interface GoogleDocEditorProps {
  text: string;
  onTextChange: (text: string) => void;
  results: AnalysisResult[];
  isLoading: boolean;
  globalAnalyses: GlobalAnalysis[];
  currentGlobalAnalysis: GlobalAnalysis | null;
  currentSession: number;
}

// Memoized comment bubble component to reduce re-renders but still show token streaming
const CommentBubble = memo(({ result, onHover, onLeave }: { 
  result: AnalysisResult, 
  onHover: (result: AnalysisResult) => void,
  onLeave: () => void
}) => {
  // Use a ref to track the previous analysis text length for cursor animation
  const prevLengthRef = useRef<number>(0);
  const [blinkCursor, setBlinkCursor] = useState<boolean>(true);
  
  // Blink cursor effect for streaming indication
  useEffect(() => {
    if (!result.is_complete) {
      const cursorInterval = setInterval(() => {
        setBlinkCursor(prev => !prev);
      }, 500);
      
      return () => clearInterval(cursorInterval);
    }
  }, [result.is_complete]);
  
  // Track when text changes to enhance the animation effect
  useEffect(() => {
    // Only animate when text is growing (streaming)
    if (result.analysis.length > prevLengthRef.current) {
      setBlinkCursor(true); // Reset blink on new content
      prevLengthRef.current = result.analysis.length;
    }
  }, [result.analysis]);

  return (
    <div 
      className="comment-bubble"
      onMouseEnter={() => onHover(result)}
      onMouseLeave={onLeave}
    >
      <div className="comment-content">
        <div className="comment-header">
          <div className="comment-avatar">AI</div>
          <div className="comment-author">AI Assistant</div>
          {!result.is_complete && (
            <div className="streaming-indicator">streaming...</div>
          )}
        </div>
        <div className="comment-text">
          {result.analysis}
          {!result.is_complete && result.analysis && (
            <span className={`typing-indicator ${blinkCursor ? 'visible' : 'hidden'}`}>▌</span>
          )}
          {result.is_complete && (
            <span className="completion-indicator">✓</span>
          )}
        </div>
      </div>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison to prevent unnecessary re-renders
  // Only re-render if the analysis text changed or completion status changed
  return prevProps.result.analysis === nextProps.result.analysis && 
         prevProps.result.is_complete === nextProps.result.is_complete;
});

// Memoized page component to avoid re-rendering all pages
const DocPage = memo(({ 
  pageContent, 
  index, 
  isLoading, 
  onTextChange,
  highlightRanges
}: { 
  pageContent: string;
  index: number;
  isLoading: boolean;
  onTextChange: (e: ChangeEvent<HTMLTextAreaElement>, index: number) => void;
  highlightRanges: Array<{start: number, end: number}>;
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [textWithHighlights, setTextWithHighlights] = useState<React.ReactNode[]>([]);
  
  // Create highlighted text spans
  useEffect(() => {
    if (highlightRanges.length === 0) {
      // No highlights, just use the normal text
      setTextWithHighlights([pageContent]);
      return;
    }

    // Sort highlight ranges by start position
    const sortedRanges = [...highlightRanges].sort((a, b) => a.start - b.start);
    
    // Calculate the absolute position where this page starts in the document
    const pageStartPos = index * 800; // Approximate position where this page starts
    const pageEndPos = pageStartPos + pageContent.length; // Where this page ends
    
    // Create an array of text segments with highlights
    const segments: React.ReactNode[] = [];
    let lastEnd = 0;
    
    // Filter to only ranges that affect this page
    const relevantRanges = sortedRanges.filter(range => 
      (range.start < pageEndPos && range.end > pageStartPos)
    );
    
    relevantRanges.forEach(range => {
      // Convert document positions to page-relative positions
      const localStart = Math.max(0, range.start - pageStartPos);
      const localEnd = Math.min(pageContent.length, range.end - pageStartPos);
      
      // Sanity check to ensure valid positions
      if (localEnd > localStart && localStart >= 0 && localEnd <= pageContent.length) {
        // Add normal text before this highlight
        if (localStart > lastEnd) {
          segments.push(pageContent.substring(lastEnd, localStart));
        }
        
        // Add highlighted text
        segments.push(
          <span key={`highlight-${localStart}-${localEnd}`} className="highlighted-text">
            {pageContent.substring(localStart, localEnd)}
          </span>
        );
        
        lastEnd = localEnd;
      }
    });
    
    // Add any remaining text after the last highlight
    if (lastEnd < pageContent.length) {
      segments.push(pageContent.substring(lastEnd));
    }
    
    setTextWithHighlights(segments);
  }, [pageContent, highlightRanges, index]);

  return (
    <div className="doc-page">
      <div className="textarea-with-highlights">
        <div className="highlights-layer">
          {/* This could be used for actual highlighting, but we're using the overlay approach instead */}
        </div>
        <textarea
          ref={textareaRef}
          className="doc-textarea"
          value={pageContent}
          onChange={(e) => onTextChange(e, index)}
          placeholder={index === 0 ? "Type your text here..." : ""}
          disabled={isLoading}
        />
        <div className="highlight-overlay">
          {textWithHighlights}
        </div>
      </div>
      <div className="page-number">{index + 1}</div>
    </div>
  );
});

// Memoized global analysis badge component
const GlobalAnalysisBadge = memo(({ globalAnalysis }: { globalAnalysis: GlobalAnalysis | null }) => {
  if (!globalAnalysis) return null;
  
  return (
    <div className="global-analysis-badge">
      <div className="global-analysis-item">
        <span className="badge-label">Tone:</span> 
        <span className="badge-value">{globalAnalysis.tone}</span>
      </div>
      <div className="global-analysis-item">
        <span className="badge-label">Subject:</span> 
        <span className="badge-value">{globalAnalysis.subject_matter}</span>
      </div>
      <div className="global-analysis-tooltip">
        <div className="tooltip-icon">i</div>
        <div className="tooltip-content">
          <p><strong>Context:</strong> {globalAnalysis.context_summary}</p>
        </div>
      </div>
    </div>
  );
});

const GoogleDocEditor: React.FC<GoogleDocEditorProps> = ({ 
  text, 
  onTextChange, 
  results, 
  isLoading,
  globalAnalyses,
  currentGlobalAnalysis,
  currentSession
}) => {
  const commentsContainerRef = useRef<HTMLDivElement>(null);
  const [pages, setPages] = useState<string[]>(['']);
  const [enableHighlighting, setEnableHighlighting] = useState<boolean>(true);
  const [highlightedResult, setHighlightedResult] = useState<AnalysisResult | null>(null);
  
  // Calculate highlight ranges when a result is hovered
  const highlightRanges = useMemo(() => {
    if (!enableHighlighting || !highlightedResult || !text) return [];
    
    // Find the text chunk in the document
    const chunkText = highlightedResult.text_chunk;
    if (!chunkText) return [];
    
    // Find all occurrences of the chunk text in the document
    const ranges: {start: number, end: number}[] = [];
    let startIndex = 0;
    let foundIndex;
    
    // Look for exact matches first
    foundIndex = text.indexOf(chunkText, startIndex);
    if (foundIndex !== -1) {
      ranges.push({
        start: foundIndex,
        end: foundIndex + chunkText.length
      });
    } else {
      // If no exact match, try a fuzzy approach for short chunks
      // This handles cases where line breaks or spaces might differ
      if (chunkText.length < 100) {
        // Create a simplified version of the chunk (lowercase, no extra spaces)
        const simplifiedChunk = chunkText.toLowerCase().replace(/\s+/g, ' ').trim();
        const simplifiedText = text.toLowerCase().replace(/\s+/g, ' ');
        
        foundIndex = simplifiedText.indexOf(simplifiedChunk);
        if (foundIndex !== -1) {
          // Use the length from the original text to preserve formatting
          ranges.push({
            start: foundIndex,
            end: foundIndex + chunkText.length
          });
        }
      }
    }
    
    return ranges;
  }, [enableHighlighting, highlightedResult, text]);
  
  // Split text into pages, approximately 800 characters per page
  useEffect(() => {
    if (!text) {
      setPages(['']);
      return;
    }
    
    const CHARS_PER_PAGE = 800;
    const pages = [];
    let remainingText = text;
    
    while (remainingText.length > 0) {
      // Try to find a good break point (paragraph or sentence end)
      let breakPoint = Math.min(remainingText.length, CHARS_PER_PAGE);
      
      // Look for paragraph break first
      const paraBreak = remainingText.lastIndexOf('\n\n', breakPoint);
      if (paraBreak !== -1 && paraBreak > CHARS_PER_PAGE / 2) {
        breakPoint = paraBreak + 2;
      } else {
        // Look for sentence end
        const sentenceBreak = remainingText.lastIndexOf('. ', breakPoint);
        if (sentenceBreak !== -1 && sentenceBreak > CHARS_PER_PAGE / 2) {
          breakPoint = sentenceBreak + 2;
        }
      }
      
      pages.push(remainingText.substring(0, breakPoint));
      remainingText = remainingText.substring(breakPoint);
    }
    
    // Ensure we have at least one page
    if (pages.length === 0) {
      pages.push('');
    }
    
    setPages(pages);
  }, [text]);
  
  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>, pageIndex: number) => {
    const newPages = [...pages];
    newPages[pageIndex] = e.target.value;
    onTextChange(newPages.join(''));
  };

  // Group results by session
  const resultsBySession = useMemo(() => {
    // Group results by session_id
    const groupedResults: Record<string, AnalysisResult[]> = {};
    
    // Sort all results by session and chunk_index
    const sortedResults = [...results].sort((a, b) => {
      // First sort by session (newest first)
      const sessionDiff = (b.session_id || 0) - (a.session_id || 0);
      if (sessionDiff !== 0) return sessionDiff;
      
      // Then sort by chunk_index 
      return a.chunk_index - b.chunk_index;
    });
    
    // Group by session
    sortedResults.forEach(result => {
      const sessionId = result.session_id?.toString() || 'unknown';
      if (!groupedResults[sessionId]) {
        groupedResults[sessionId] = [];
      }
      groupedResults[sessionId].push(result);
    });
    
    return groupedResults;
  }, [results]);

  // For current session, sort by chunk_index only
  const currentSessionResults = useMemo(() => {
    return results
      .filter(r => r.session_id === currentSession)
      .sort((a, b) => a.chunk_index - b.chunk_index);
  }, [results, currentSession]);

  // Find the global analysis for a given session
  const getGlobalAnalysisForSession = useCallback((sessionId: number) => {
    return globalAnalyses.find(ga => ga.session_id === sessionId) || null;
  }, [globalAnalyses]);

  // Scroll to the bottom when new tokens stream in for current session
  useEffect(() => {
    if (commentsContainerRef.current) {
      // Check if user is near the bottom before auto-scrolling
      const container = commentsContainerRef.current;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
      
      if (isNearBottom || currentSessionResults.some(r => !r.is_complete)) {
        container.scrollTop = container.scrollHeight;
      }
    }
  }, [currentSessionResults]);

  // Handle hovering a result
  const handleResultHover = useCallback((result: AnalysisResult) => {
    setHighlightedResult(result);
  }, []);
  
  // Handle leaving a result
  const handleResultLeave = useCallback(() => {
    setHighlightedResult(null);
  }, []);

  // Toggle highlighting
  const toggleHighlighting = useCallback(() => {
    setEnableHighlighting(prev => !prev);
    setHighlightedResult(null);
  }, []);

  // Add CSS for the streaming indication and highlighting
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      .streaming-indicator {
        font-size: 0.7rem;
        color: #2b6cb0;
        animation: pulse 2s infinite;
        margin-left: auto;
      }
      
      @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
      }
      
      .typing-indicator {
        display: inline-block;
        vertical-align: middle;
        transition: opacity 0.2s;
      }
      
      .typing-indicator.visible {
        opacity: 1;
      }
      
      .typing-indicator.hidden {
        opacity: 0;
      }
      
      .comment-header {
        display: flex;
        align-items: center;
      }
      
      .comment-text {
        white-space: pre-wrap;
        word-break: break-word;
      }
      
      .completion-indicator {
        color: #38a169;
        margin-left: 4px;
      }
      
      .global-analysis-tooltip {
        position: relative;
        display: inline-flex;
        margin-left: 8px;
      }
      
      .tooltip-icon {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #4299e1;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        cursor: pointer;
      }
      
      .tooltip-content {
        position: absolute;
        top: 100%;
        right: 0;
        width: 250px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        padding: 8px;
        z-index: 10;
        display: none;
      }
      
      .global-analysis-tooltip:hover .tooltip-content {
        display: block;
      }
      
      .textarea-with-highlights {
        position: relative;
        width: 100%;
        height: 100%;
      }
      
      .doc-textarea {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 2;
        background-color: transparent;
        color: #000;
        caret-color: #000;
      }
      
      .highlight-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
        padding: 8px;
        font-family: inherit;
        font-size: inherit;
        line-height: inherit;
        white-space: pre-wrap;
        word-wrap: break-word;
        box-sizing: border-box;
        color: transparent;
      }
      
      .highlighted-text {
        background-color: rgba(255, 230, 0, 0.3);
        border-radius: 2px;
      }
      
      .session-divider {
        margin: 20px 0;
        padding: 8px;
        background-color: #f7fafc;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
      }
      
      .highlight-toggle {
        margin-left: 12px;
        padding: 4px 8px;
        background-color: ${enableHighlighting ? '#4299e1' : '#e2e8f0'};
        color: ${enableHighlighting ? 'white' : '#4a5568'};
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.8rem;
        transition: all 0.2s;
      }
      
      .highlight-toggle:hover {
        opacity: 0.9;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, [enableHighlighting]);

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
          <div className="doc-pages">
            {pages.map((pageContent, index) => (
              <DocPage
                key={`page-${index}`}
                pageContent={pageContent}
                index={index}
                isLoading={isLoading}
                onTextChange={handleTextChange}
                highlightRanges={enableHighlighting ? highlightRanges : []}
              />
            ))}
          </div>
        </div>
      </div>
      <div className="doc-comments">
        <div className="comments-header">
          <div className="comments-title">
            Comments
            <GlobalAnalysisBadge globalAnalysis={currentGlobalAnalysis} />
            <button 
              className="highlight-toggle" 
              onClick={toggleHighlighting}
              title={enableHighlighting ? "Disable text highlighting" : "Enable text highlighting"}
            >
              {enableHighlighting ? "Highlighting On" : "Highlighting Off"}
            </button>
          </div>
          {isLoading && (
            <div className="comments-status">
              <div className="loading-spinner"></div>
              <span>Analyzing document...</span>
            </div>
          )}
        </div>
        
        {/* Show results grouped by session */}
        {Object.entries(resultsBySession).length > 0 && (
          <div className="comments-container" ref={commentsContainerRef}>
            {Object.entries(resultsBySession).map(([sessionId, sessionResults]) => (
              <div key={`session-${sessionId}`} className="analysis-session">
                {/* Show session divider for all except the current session */}
                {Number(sessionId) !== currentSession && (
                  <div className="session-divider">
                    <div>Analysis Session {sessionId}</div>
                    <GlobalAnalysisBadge globalAnalysis={getGlobalAnalysisForSession(Number(sessionId))} />
                  </div>
                )}
                
                {/* Show analysis results for this session */}
                {sessionResults.map(result => (
                  <CommentBubble 
                    key={`result-${sessionId}-${result.chunk_index}`} 
                    result={result} 
                    onHover={handleResultHover}
                    onLeave={handleResultLeave}
                  />
                ))}
              </div>
            ))}
          </div>
        )}
        
        {!isLoading && Object.entries(resultsBySession).length === 0 && (
          <div className="no-comments">
            No comments yet. Click "Analyze Text" to generate comments.
          </div>
        )}
      </div>
    </div>
  );
};

export default GoogleDocEditor; 