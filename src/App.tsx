import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import GoogleDocEditor from './components/GoogleDocEditor'

// Define types for the analysis results
interface AnalysisResult {
  chunk_index: number;
  text_chunk: string;
  analysis: string;
  is_complete?: boolean;
}

interface TokenUpdate {
  chunk_index: number;
  token: string;
  is_complete: boolean;
}

interface GlobalAnalysis {
  tone: string;
  subject_matter: string;
  context_summary: string;
}

function App() {
  const [text, setText] = useState<string>('');
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [globalAnalysis, setGlobalAnalysis] = useState<GlobalAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const expectedChunksRef = useRef<number>(0);
  
  const checkIfAnalysisComplete = useCallback(() => {
    if (results.length === 0) return false;
    
    // Check that all expected chunks exist and are marked complete
    const allComplete = results.every(result => result.is_complete === true);
    const hasAllExpectedChunks = results.length >= expectedChunksRef.current;
    
    // Added more detailed logging for debugging
    if (allComplete && hasAllExpectedChunks) {
      console.log("Analysis complete check: all chunks complete and have all expected chunks");
      return true;
    }
    
    return false;
  }, [results]);
  
  useEffect(() => {
    if (isLoading && checkIfAnalysisComplete()) {
      console.log("All chunks complete, setting isLoading to false");
      setIsLoading(false);
    }
  }, [results, isLoading, checkIfAnalysisComplete]);

  // Safety timer to ensure isLoading gets reset after a timeout
  useEffect(() => {
    let safetyTimer: number | null = null;
    
    if (isLoading) {
      // If still loading after 5 seconds with no activity, force reset
      safetyTimer = window.setTimeout(() => {
        console.log("Safety timeout triggered: resetting isLoading state");
        setIsLoading(false);
      }, 5000); // Reduced from 10000ms to 5000ms
    }
    
    return () => {
      if (safetyTimer !== null) {
        clearTimeout(safetyTimer);
      }
    };
  }, [isLoading]);

  const closeConnection = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      console.log("Previous fetch aborted.");
    }
    console.log("Setting isLoading to false from closeConnection");
    setIsLoading(false);
  };

  useEffect(() => {
    return () => {
      closeConnection();
    };
  }, []);

  const handleAnalyze = async () => {
    closeConnection();
    console.log("Setting isLoading to true for analysis");
    setIsLoading(true);
    setError(null);
    setResults([]);
    setGlobalAnalysis(null);
    expectedChunksRef.current = 0;

    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    try {
      const response = await fetch('http://localhost:5001/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({ text }),
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

      while (true) {
        if (streamEnded) {
          console.log("Breaking out of stream loop due to end event");
          setIsLoading(false);
          break;
        }

        const { done, value } = await reader.read();
        if (done) {
          console.log("Stream finished.");
          console.log("Setting isLoading to false from stream done");
          setIsLoading(false);
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          const eventMatch = line.match(/^event:\s*(.*)$/m);
          const dataMatch = line.match(/^data:\s*(.*)$/m);

          const event = eventMatch ? eventMatch[1].trim() : null;
          const dataString = dataMatch ? dataMatch[1].trim() : null;

          if (dataString && (event === 'global_analysis' || event === 'chunk' || event === 'token' || event === 'error' || event === 'chunk_complete')) {
            try {
              const data = JSON.parse(dataString);

              if (event === 'global_analysis') {
                console.log("Received global analysis:", data);
                setGlobalAnalysis(data);
              } else if (event === 'chunk') {
                console.log("Received chunk:", data.chunk_index, "is_complete:", data.is_complete);
                
                expectedChunksRef.current = Math.max(expectedChunksRef.current, data.chunk_index + 1);
                
                if (data.is_complete) {
                  setResults(prevResults => {
                    const exists = prevResults.some(r => r.chunk_index === data.chunk_index);
                    
                    if (exists) {
                      return prevResults.map(r => 
                        r.chunk_index === data.chunk_index ? data : r
                      );
                    } else {
                      return [...prevResults, data];
                    }
                  });
                } else {
                  setResults(prevResults => {
                    const exists = prevResults.some(r => r.chunk_index === data.chunk_index);
                    if (!exists) {
                      return [...prevResults, data];
                    }
                    return prevResults;
                  });
                }
              } else if (event === 'token') {
                setResults(prevResults => {
                  return prevResults.map(result => {
                    if (result.chunk_index === data.chunk_index) {
                      return {
                        ...result,
                        analysis: result.analysis + data.token
                      };
                    }
                    return result;
                  });
                });
              } else if (event === 'chunk_complete') {
                console.log("Received chunk_complete for chunk:", data.chunk_index);
                setResults(prevResults => {
                  return prevResults.map(result => {
                    if (result.chunk_index === data.chunk_index) {
                      return {
                        ...result,
                        is_complete: true
                      };
                    }
                    return result;
                  });
                });
              } else if (event === 'error') {
                console.error("Received stream error:", data.error);
                setError(data.error);
                closeConnection();
                return;
              }
            } catch (e) {
              console.error("Failed to parse SSE JSON data:", dataString, e);
            }
          } else if (event === 'end') {
              console.log("Received end event from stream.");
              console.log("Setting isLoading to false from end event");
              setIsLoading(false);
              // Immediately close the connection and set streamEnded to true
              streamEnded = true;
              if (abortControllerRef.current) {
                abortControllerRef.current.abort();
                abortControllerRef.current = null;
              }
              // Break out of the processing loop immediately
              break;
          } else if (dataString) {
              console.warn(`Received data for unhandled or unknown event type '${event}':`, dataString);
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
      console.log("Setting isLoading to false from finally block");
      setIsLoading(false);
      if (abortControllerRef.current && !abortControllerRef.current.signal.aborted) {
        abortControllerRef.current = null;
      }
    }
  };

  return (
    <div className="app-container">
      <h1>Google Doc Style Editor</h1>
      
      {/* Global analysis is hidden from UI but still used internally */}
      
      <GoogleDocEditor 
        text={text}
        onTextChange={setText}
        results={results}
        isLoading={isLoading}
        globalAnalysis={globalAnalysis}
      />
      
      <div className="action-buttons">
        <button onClick={handleAnalyze} disabled={isLoading || !text.trim()}>
          {isLoading ? 'Analyzing...' : 'Analyze Text'}
        </button>
        {isLoading && <button onClick={closeConnection}>Cancel</button>}
        {error && <p className="error-message">Error: {error}</p>}
      </div>
    </div>
  )
}

export default App
