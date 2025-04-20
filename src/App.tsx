import { useState, useEffect, useRef } from 'react'
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

function App() {
  const [text, setText] = useState<string>('');
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [overallTone, setOverallTone] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const closeConnection = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      console.log("Previous fetch aborted.");
    }
    setIsLoading(false);
  };

  useEffect(() => {
    return () => {
      closeConnection();
    };
  }, []);

  const handleAnalyze = async () => {
    closeConnection();
    setIsLoading(true);
    setError(null);
    setResults([]);
    setOverallTone(null);

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

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log("Stream finished.");
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

          // Only attempt to parse if we have data AND it's an event type we expect JSON from
          if (dataString && (event === 'tone' || event === 'chunk' || event === 'token' || event === 'error')) {
            try {
              const data = JSON.parse(dataString);

              if (event === 'tone') {
                console.log("Received tone:", data.overall_tone);
                setOverallTone(data.overall_tone);
              } else if (event === 'chunk') {
                console.log("Received chunk:", data.chunk_index, "is_complete:", data.is_complete);
                
                if (data.is_complete) {
                  // Final update for this chunk with complete analysis
                  setResults(prevResults => {
                    // Find and replace the existing chunk if it exists
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
                  // Initial chunk setup
                  setResults(prevResults => {
                    // Only add if it doesn't exist yet
                    const exists = prevResults.some(r => r.chunk_index === data.chunk_index);
                    if (!exists) {
                      return [...prevResults, data];
                    }
                    return prevResults;
                  });
                }
              } else if (event === 'token') {
                // Update the current chunk with the new token
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
              } else if (event === 'error') {
                console.error("Received stream error:", data.error);
                setError(data.error);
                closeConnection(); // Stop processing on stream error
                return; // Exit the loop
              }
            } catch (e) {
              console.error("Failed to parse SSE JSON data:", dataString, e);
              // Optionally set an error state here too
              // setError("Failed to parse stream data.");
              // closeConnection();
              // return;
            }
          } else if (event === 'end') {
              console.log("Received end event from stream.");
              // Backend signalled end, reader.read() loop will handle closing.
          } else if (dataString) {
              // Log unexpected event data
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
      if (abortControllerRef.current && !abortControllerRef.current.signal.aborted) {
        setIsLoading(false);
      }
      if (abortControllerRef.current && !abortControllerRef.current.signal.aborted) {
        abortControllerRef.current = null;
      }
    }
  };

  return (
    <div className="app-container">
      <h1>Google Doc Style Editor</h1>
      {overallTone && <p className="tone-indicator">Overall Tone: <strong>{overallTone}</strong></p>}
      
      <GoogleDocEditor 
        text={text}
        onTextChange={setText}
        results={results}
        isLoading={isLoading}
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
