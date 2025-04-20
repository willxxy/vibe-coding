import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

// Document interface
export interface Document {
  id: string;
  title: string;
  content: string;
  lastModified: Date;
  created: Date;
}

interface DocumentListProps {
  documents: Document[];
  onCreateDocument: () => void;
  onDeleteDocument: (id: string) => void;
}

const DocumentList: React.FC<DocumentListProps> = ({ 
  documents, 
  onCreateDocument,
  onDeleteDocument 
}) => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  
  // Filter documents based on search term
  const filteredDocuments = documents.filter(doc => 
    doc.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    }).format(date);
  };

  const handleDocumentClick = (id: string) => {
    navigate(`/document/${id}`);
  };

  return (
    <div className="document-list-container">
      <div className="document-list-header">
        <h1>My Documents</h1>
        <div className="document-controls">
          <div className="search-container">
            <input
              type="text"
              placeholder="Search documents..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
            {searchTerm && (
              <button 
                className="clear-search" 
                onClick={() => setSearchTerm('')}
              >
                ×
              </button>
            )}
          </div>
          <button 
            className="create-document-btn"
            onClick={onCreateDocument}
          >
            + New Document
          </button>
        </div>
      </div>

      {filteredDocuments.length === 0 ? (
        <div className="no-documents">
          {searchTerm ? (
            <p>No documents match your search.</p>
          ) : (
            <p>No documents yet. Create your first document to get started!</p>
          )}
        </div>
      ) : (
        <div className="document-grid">
          {filteredDocuments.map(doc => (
            <div 
              key={doc.id} 
              className="document-card"
              onClick={() => handleDocumentClick(doc.id)}
            >
              <div className="document-preview">
                {doc.content ? (
                  <p className="preview-content">
                    {doc.content.substring(0, 150)}
                    {doc.content.length > 150 && '...'}
                  </p>
                ) : (
                  <p className="empty-preview">Empty document</p>
                )}
              </div>
              <div className="document-info">
                <h3 className="document-title">{doc.title || 'Untitled Document'}</h3>
                <p className="document-date">Last modified: {formatDate(doc.lastModified)}</p>
                <button 
                  className="delete-document-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm(`Are you sure you want to delete "${doc.title || 'Untitled Document'}"?`)) {
                      onDeleteDocument(doc.id);
                    }
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DocumentList; 