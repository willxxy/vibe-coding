import { useState, useEffect, useRef, useCallback } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import './App.css'
import DocumentList, { Document } from './components/DocumentList'
import DocumentEditor from './components/DocumentEditor'
import { v4 as uuidv4 } from 'uuid'

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

// Storage keys
const DOCUMENTS_STORAGE_KEY = 'vibe-docs-documents';

function App() {
  // Document state
  const [documents, setDocuments] = useState<Document[]>([]);
  
  // Analysis state
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

  // Load documents from localStorage on app start
  useEffect(() => {
    const storedDocuments = localStorage.getItem(DOCUMENTS_STORAGE_KEY);
    if (storedDocuments) {
      try {
        const parsedDocs = JSON.parse(storedDocuments);
        // Convert string dates back to Date objects
        const docsWithDateObjects = parsedDocs.map((doc: any) => ({
          ...doc,
          lastModified: new Date(doc.lastModified),
          created: new Date(doc.created)
        }));
        setDocuments(docsWithDateObjects);
      } catch (e) {
        console.error('Error parsing stored documents:', e);
      }
    }
  }, []);

  // Save documents to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem(DOCUMENTS_STORAGE_KEY, JSON.stringify(documents));
  }, [documents]);

  // Create a new document
  const handleCreateDocument = () => {
    const newDoc: Document = {
      id: uuidv4(),
      title: 'Untitled Document',
      content: '',
      lastModified: new Date(),
      created: new Date()
    };
    
    setDocuments(prev => [newDoc, ...prev]);
    return newDoc.id;
  };

  // Delete a document
  const handleDeleteDocument = (id: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id));
  };

  // Update a document
  const handleUpdateDocument = (id: string, content: string, title?: string) => {
    setDocuments(prev => 
      prev.map(doc => {
        if (doc.id === id) {
          return {
            ...doc,
            content,
            title: title !== undefined ? title : doc.title,
            lastModified: new Date()
          };
        }
        return doc;
      })
    );
  };

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

  return (
    <Router>
      <div className="app-container">
        <Routes>
          <Route path="/" element={
            <DocumentList 
              documents={documents}
              onCreateDocument={() => {
                const newId = handleCreateDocument();
                return newId;
              }}
              onDeleteDocument={handleDeleteDocument}
            />
          } />
          <Route path="/document/:id" element={
            <DocumentEditor
              documents={documents}
              updateDocument={handleUpdateDocument}
              currentGlobalAnalysis={currentGlobalAnalysis}
              globalAnalyses={globalAnalyses}
            />
          } />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App
