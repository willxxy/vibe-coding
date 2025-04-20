import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import './App.css'
import GoogleDocEditor from './components/GoogleDocEditor'

// Define types for the analysis results
interface AnalysisResult {
  chunk_index: number;
  text_chunk: string;
  analysis: string;
  is_complete?: boolean;
  session_id?: number; // Track which analysis session this result belongs to
}

interface GlobalAnalysis {
  tone: string;
  subject_matter: string;
  context_summary: string;
  session_id?: number; // Track which analysis session this belongs to
}

interface TokenUpdate {
  chunk_index: number;
  token: string;
  is_complete: boolean;
}

function App() {
  const [text, setText] = useState<string>('');
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [globalAnalyses, setGlobalAnalyses] = useState<GlobalAnalysis[]>([]); 
  const [currentGlobalAnalysis, setCurrentGlobalAnalysis] = useState<GlobalAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [progressMessage, setProgressMessage] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState<boolean>(false);
  const [retryCount, setRetryCount] = useState<number>(0);
  const [showCompletionNotice, setShowCompletionNotice] = useState<boolean>(false);
  const [analysisTime, setAnalysisTime] = useState<string>('');
  
  // References for managing analysis state
  const abortControllerRef = useRef<AbortController | null>(null);
  const expectedChunksRef = useRef<number>(0);
  const analysisSessionRef = useRef<number>(0);
  
  // Optimize token updates batching
  const pendingTokensRef = useRef<Map<number, string>>(new Map());
  const tokenUpdateTimeoutRef = useRef<number | null>(null);
  const tokenBatchSizeRef = useRef<number>(0);
  
  // Check if all analysis chunks are complete
  const checkIfAnalysisComplete = useCallback(() => {
    if (results.length === 0) return false;
    
    // Only check completion for the current session
    const currentSessionResults = results.filter(r => r.session_id === analysisSessionRef.current);
    if (currentSessionResults.length === 0) return false;
    
    // Check that all expected chunks exist and are marked complete
    const allComplete = currentSessionResults.every(result => result.is_complete === true);
    const hasAllExpectedChunks = currentSessionResults.length >= expectedChunksRef.current;
    
    return allComplete && hasAllExpectedChunks && expectedChunksRef.current > 0;
  }, [results]);
  
  // Effect for handling analysis completion
  useEffect(() => {
    if (isLoading && checkIfAnalysisComplete()) {
      setIsLoading(false);
      setShowCompletionNotice(true);
      setTimeout(() => setShowCompletionNotice(false), 3000);
    }
  }, [results, isLoading, checkIfAnalysisComplete]);

  // Handle cancellation of ongoing requests
  const closeConnection = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    
    // Clear any pending token updates
    if (tokenUpdateTimeoutRef.current) {
      window.clearTimeout(tokenUpdateTimeoutRef.current);
      tokenUpdateTimeoutRef.current = null;
    }
    
    setIsLoading(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      closeConnection();
    };
  }, [closeConnection]);

  // Flush pending tokens to state efficiently
  const flushTokenUpdates = useCallback(() => {
    if (pendingTokensRef.current.size === 0) return;
    
    setResults(prevResults => {
      // Create a new array only if we actually make changes
      const newResults = [...prevResults];
      let modified = false;
      
      pendingTokensRef.current.forEach((tokens, chunkIndex) => {
        // Find result with current session
        const resultIndex = newResults.findIndex(r => 
          r.chunk_index === chunkIndex && r.session_id === analysisSessionRef.current
        );
        
        if (resultIndex >= 0) {
          newResults[resultIndex] = {
            ...newResults[resultIndex],
            analysis: newResults[resultIndex].analysis + tokens
          };
          modified = true;
        }
      });
      
      pendingTokensRef.current.clear();
      tokenBatchSizeRef.current = 0;
      return modified ? newResults : prevResults;
    });
    
    tokenUpdateTimeoutRef.current = null;
  }, []);

  // Schedule token flush with adaptive timing
  const scheduleTokenFlush = useCallback(() => {
    // For very small token batches, use a longer timeout for efficiency
    let flushDelay = 10; // base delay in ms
    
    // Adaptive timing based on batch size
    if (tokenBatchSizeRef.current > 100) {
      flushDelay = 5; // Flush quickly for large batches
    } else if (tokenBatchSizeRef.current < 20) {
      flushDelay = 30; // Wait longer for small batches
    }
    
    // If we already have a timeout scheduled, don't schedule another
    if (tokenUpdateTimeoutRef.current !== null) return;
    
    tokenUpdateTimeoutRef.current = window.setTimeout(() => {
      flushTokenUpdates();
    }, flushDelay);
  }, [flushTokenUpdates]);

  // Optimized token update handler
  const handleTokenUpdate = useCallback((update: TokenUpdate) => {
    const { chunk_index, token, is_complete } = update;
    
    // Immediately update completion status if needed
    if (is_complete) {
      setResults(prevResults => 
        prevResults.map(result => {
          if (result.chunk_index === chunk_index && result.session_id === analysisSessionRef.current) {
            return { ...result, is_complete: true };
          }
          return result;
        })
      );
      
      // Force flush any pending tokens for this chunk
      const pendingToken = pendingTokensRef.current.get(chunk_index);
      if (pendingToken) {
        const currentToken = pendingToken;
        pendingTokensRef.current.delete(chunk_index);
        tokenBatchSizeRef.current -= currentToken.length;
        
        // Update with the final token
        setResults(prevResults => {
          const resultIndex = prevResults.findIndex(r => 
            r.chunk_index === chunk_index && r.session_id === analysisSessionRef.current
          );
          
          if (resultIndex === -1) return prevResults;
          
          const newResults = [...prevResults];
          newResults[resultIndex] = {
            ...newResults[resultIndex],
            analysis: newResults[resultIndex].analysis + currentToken,
            is_complete: true
          };
          return newResults;
        });
      }
      return;
    }
    
    // Batch token updates
    const current = pendingTokensRef.current.get(chunk_index) || '';
    pendingTokensRef.current.set(chunk_index, current + token);
    tokenBatchSizeRef.current += token.length;
    
    // Schedule the token flush
    scheduleTokenFlush();
  }, [scheduleTokenFlush]);

  // Main analyze function
  const handleAnalyze = async () => {
    closeConnection();
    setIsLoading(true);
    setError(null);
    setRetrying(false);
    setProgressMessage('Initializing analysis...');
    setAnalysisTime('');
    
    // Increment the analysis session for each new analysis
    analysisSessionRef.current += 1;
    const currentSession = analysisSessionRef.current;
    
    // Reset current global analysis
    setCurrentGlobalAnalysis(null);
    
    // Reset expected chunks counter
    expectedChunksRef.current = 0;
    
    // Clear any pending token updates
    pendingTokensRef.current.clear();
    if (tokenUpdateTimeoutRef.current) {
      window.clearTimeout(tokenUpdateTimeoutRef.current);
      tokenUpdateTimeoutRef.current = null;
    }

    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    try {
      const response = await fetch('http://localhost:5001/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({ 
          text,
          chunk_size: 3
        }),
        signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("Response body is missing.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let streamEnded = false;

      while (!streamEnded) {
        const { done, value } = await reader.read();
        
        if (done) {
          setIsLoading(false);
          break;
        }

        // If the session has changed, ignore further processing
        if (currentSession !== analysisSessionRef.current) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        
        // Handle event boundaries
        const events = buffer.split('\n\n');
        // Keep the last partial event in the buffer
        buffer = events.pop() || '';

        for (const event of events) {
          if (!event.trim()) continue;

          // Parse SSE format
          const lines = event.split('\n');
          let eventType = '';
          let data = '';
          
          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.substring(6).trim();
            } else if (line.startsWith('data:')) {
              data = line.substring(5).trim();
            }
          }

          if (!eventType || !data) continue;

          try {
            const parsedData = JSON.parse(data);

            // Ignore events if session has changed
            if (currentSession !== analysisSessionRef.current) {
              break;
            }

            switch (eventType) {
              case 'global_analysis':
                // Add session ID to the global analysis
                const analysisWithSession = {
                  ...parsedData,
                  session_id: currentSession
                };
                
                // Update both current and stored global analyses
                setCurrentGlobalAnalysis(analysisWithSession);
                setGlobalAnalyses(prev => [...prev, analysisWithSession]);
                break;
                
              case 'progress':
                // Handle progress updates
                if (parsedData && parsedData.message) {
                  setProgressMessage(parsedData.message);
                }
                break;
                
              case 'chunk':
                // Update expected chunks counter
                if (parsedData.chunk_index >= expectedChunksRef.current) {
                  expectedChunksRef.current = parsedData.chunk_index + 1;
                }
                
                // Add session ID to the chunk data
                const chunkWithSession = {
                  ...parsedData,
                  session_id: currentSession
                };
                
                setResults(prevResults => {
                  // Find if we already have a result for this chunk in this session
                  const existingIndex = prevResults.findIndex(r => 
                    r.chunk_index === parsedData.chunk_index && r.session_id === currentSession
                  );
                  
                  if (existingIndex >= 0) {
                    // Only update if necessary
                    if (
                      prevResults[existingIndex].is_complete !== chunkWithSession.is_complete ||
                      prevResults[existingIndex].text_chunk !== chunkWithSession.text_chunk ||
                      prevResults[existingIndex].analysis !== chunkWithSession.analysis
                    ) {
                      const newResults = [...prevResults];
                      newResults[existingIndex] = chunkWithSession;
                      return newResults;
                    }
                    return prevResults;
                  } else {
                    // Add new chunk
                    return [...prevResults, chunkWithSession];
                  }
                });
                break;
                
              case 'token':
                // Use optimized token handling
                handleTokenUpdate({
                  chunk_index: parsedData.chunk_index,
                  token: parsedData.token,
                  is_complete: parsedData.is_complete
                });
                break;
                
              case 'error':
                setError(parsedData.error);
                closeConnection();
                return;
                
              case 'end':
                // Final flush of any pending tokens
                flushTokenUpdates();
                
                streamEnded = true;
                setIsLoading(false);
                setProgressMessage('');
                
                // Extract time information if available
                if (parsedData && parsedData.time_taken) {
                  setAnalysisTime(parsedData.time_taken);
                }
                
                setShowCompletionNotice(true);
                setTimeout(() => setShowCompletionNotice(false), 3000);
                
                if (abortControllerRef.current) {
                  abortControllerRef.current = null;
                }
                break;
                
              default:
                console.warn(`Unhandled event type: ${eventType}`);
            }
          } catch (e) {
            console.error("Failed to parse SSE JSON data:", data, e);
          }
        }
      }
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') {
        console.log('Fetch aborted');
      } else {
        console.error("Analysis failed:", e);
        setError(e instanceof Error ? e.message : 'An unknown error occurred.');
      }
    } finally {
      setIsLoading(false);
      setProgressMessage('');
      abortControllerRef.current = null;
    }
  };

  // Retry mechanism with progressive backoff
  const handleRetry = useCallback(() => {
    setRetrying(true);
    setRetryCount(prev => prev + 1);
    setError(null);
    
    // Progressive backoff for retries
    const backoffTime = Math.min(1000 * retryCount, 5000);
    
    setTimeout(() => {
      handleAnalyze();
      setRetrying(false);
    }, backoffTime);
  }, [retryCount]);
  
  // Clear all analyses
  const handleClearAnalyses = useCallback(() => {
    setResults([]);
    setGlobalAnalyses([]);
    setCurrentGlobalAnalysis(null);
    expectedChunksRef.current = 0;
  }, []);

  return (
    <div className="app-container">
      <h1>Google Doc Style Editor</h1>
      
      <GoogleDocEditor 
        text={text}
        onTextChange={setText}
        results={results}
        isLoading={isLoading}
        progressMessage={progressMessage}
        globalAnalyses={globalAnalyses}
        currentGlobalAnalysis={currentGlobalAnalysis}
        currentSession={analysisSessionRef.current}
      />
      
      <div className="action-buttons">
        <button onClick={handleAnalyze} disabled={isLoading || retrying || !text.trim()}>
          {isLoading ? 'Analyzing...' : retrying ? `Retrying (${retryCount})...` : 'Analyze Text'}
        </button>
        {isLoading && <button onClick={closeConnection}>Cancel</button>}
        {results.length > 0 && (
          <button onClick={handleClearAnalyses} className="clear-button">
            Clear All Analyses
          </button>
        )}
        {error && (
          <div className="error-container">
            <p className="error-message">Error: {error}</p>
            <button onClick={handleRetry} className="retry-button">
              Retry
            </button>
          </div>
        )}
        
        {showCompletionNotice && (
          <div className="completion-notification">
            Analysis complete! {analysisTime ? `(${analysisTime})` : '✓'}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
