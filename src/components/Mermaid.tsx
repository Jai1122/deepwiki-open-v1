import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
// We'll use dynamic import for svg-pan-zoom

// Initialize mermaid with defaults - Japanese aesthetic
mermaid.initialize({
  startOnLoad: true,
  theme: 'neutral',
  securityLevel: 'loose',
  suppressErrorRendering: true,
  logLevel: 'error',
  maxTextSize: 100000, // Increase text size limit
  htmlLabels: true,
  flowchart: {
    htmlLabels: true,
    curve: 'basis',
    nodeSpacing: 60,
    rankSpacing: 60,
    padding: 20,
  },
  themeCSS: `
    /* Colorful Japanese aesthetic styles for all diagrams */
    .node rect, .node circle, .node ellipse, .node polygon, .node path {
      fill: #e8f4fd;
      stroke: #4a90e2;
      stroke-width: 2px;
    }
    
    /* Different colors for different node types */
    .node:nth-child(4n+1) rect, .node:nth-child(4n+1) circle, .node:nth-child(4n+1) ellipse, .node:nth-child(4n+1) polygon, .node:nth-child(4n+1) path {
      fill: #fef3e8;
      stroke: #e67e22;
    }
    .node:nth-child(4n+2) rect, .node:nth-child(4n+2) circle, .node:nth-child(4n+2) ellipse, .node:nth-child(4n+2) polygon, .node:nth-child(4n+2) path {
      fill: #f0fdf4;
      stroke: #22c55e;
    }
    .node:nth-child(4n+3) rect, .node:nth-child(4n+3) circle, .node:nth-child(4n+3) ellipse, .node:nth-child(4n+3) polygon, .node:nth-child(4n+3) path {
      fill: #fdf2f8;
      stroke: #e879f9;
    }
    .node:nth-child(4n+4) rect, .node:nth-child(4n+4) circle, .node:nth-child(4n+4) ellipse, .node:nth-child(4n+4) polygon, .node:nth-child(4n+4) path {
      fill: #fff7ed;
      stroke: #f97316;
    }
    
    .edgePath .path {
      stroke: #6366f1;
      stroke-width: 2px;
    }
    .edgeLabel {
      background-color: rgba(255, 255, 255, 0.9);
      color: #1e293b;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 500;
      p {
        background-color: transparent !important;
      }
    }
    .label {
      color: #1e293b;
      font-weight: 500;
    }
    .cluster rect {
      fill: #f1f5f9;
      stroke: #64748b;
      stroke-width: 2px;
    }

    /* Colorful sequence diagram specific styles */
    .actor {
      fill: #dbeafe;
      stroke: #3b82f6;
      stroke-width: 2px;
    }
    .actor:nth-of-type(2n) {
      fill: #dcfce7;
      stroke: #22c55e;
    }
    .actor:nth-of-type(3n) {
      fill: #fef3c7;
      stroke: #f59e0b;
    }
    text.actor {
      fill: #1e293b;
      stroke: none;
      font-weight: 600;
    }
    .messageText {
      fill: #1e293b;
      stroke: none;
      font-weight: 500;
    }
    .messageLine0, .messageLine1 {
      stroke: #6366f1;
      stroke-width: 2px;
    }
    .messageLine0.note, .messageLine1.note {
      stroke: #ef4444;
      stroke-width: 2px;
    }
    .noteText {
      fill: #1e293b;
      font-weight: 500;
    }
    .note {
      fill: #fef2f2;
      stroke: #ef4444;
      stroke-width: 2px;
    }

    /* Colorful dark mode overrides - will be applied with data-theme="dark" */
    [data-theme="dark"] .node rect,
    [data-theme="dark"] .node circle,
    [data-theme="dark"] .node ellipse,
    [data-theme="dark"] .node polygon,
    [data-theme="dark"] .node path {
      fill: #1e293b;
      stroke: #60a5fa;
    }
    
    /* Different colors for different node types in dark mode */
    [data-theme="dark"] .node:nth-child(4n+1) rect,
    [data-theme="dark"] .node:nth-child(4n+1) circle,
    [data-theme="dark"] .node:nth-child(4n+1) ellipse,
    [data-theme="dark"] .node:nth-child(4n+1) polygon,
    [data-theme="dark"] .node:nth-child(4n+1) path {
      fill: #292524;
      stroke: #fb923c;
    }
    [data-theme="dark"] .node:nth-child(4n+2) rect,
    [data-theme="dark"] .node:nth-child(4n+2) circle,
    [data-theme="dark"] .node:nth-child(4n+2) ellipse,
    [data-theme="dark"] .node:nth-child(4n+2) polygon,
    [data-theme="dark"] .node:nth-child(4n+2) path {
      fill: #14532d;
      stroke: #4ade80;
    }
    [data-theme="dark"] .node:nth-child(4n+3) rect,
    [data-theme="dark"] .node:nth-child(4n+3) circle,
    [data-theme="dark"] .node:nth-child(4n+3) ellipse,
    [data-theme="dark"] .node:nth-child(4n+3) polygon,
    [data-theme="dark"] .node:nth-child(4n+3) path {
      fill: #2e1065;
      stroke: #c084fc;
    }
    [data-theme="dark"] .node:nth-child(4n+4) rect,
    [data-theme="dark"] .node:nth-child(4n+4) circle,
    [data-theme="dark"] .node:nth-child(4n+4) ellipse,
    [data-theme="dark"] .node:nth-child(4n+4) polygon,
    [data-theme="dark"] .node:nth-child(4n+4) path {
      fill: #431407;
      stroke: #fb923c;
    }
    
    [data-theme="dark"] .edgePath .path {
      stroke: #a78bfa;
      stroke-width: 2px;
    }
    [data-theme="dark"] .edgeLabel {
      background-color: rgba(0, 0, 0, 0.8);
      color: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 500;
    }
    [data-theme="dark"] .label {
      color: #f1f5f9;
      font-weight: 500;
    }
    [data-theme="dark"] .cluster rect {
      fill: #1e293b;
      stroke: #64748b;
    }
    [data-theme="dark"] .flowchart-link {
      stroke: #a78bfa;
    }

    /* Colorful dark mode sequence diagram overrides */
    [data-theme="dark"] .actor {
      fill: #1e3a8a;
      stroke: #60a5fa;
      stroke-width: 2px;
    }
    [data-theme="dark"] .actor:nth-of-type(2n) {
      fill: #14532d;
      stroke: #4ade80;
    }
    [data-theme="dark"] .actor:nth-of-type(3n) {
      fill: #92400e;
      stroke: #fbbf24;
    }
    [data-theme="dark"] text.actor {
      fill: #f1f5f9;
      stroke: none;
      font-weight: 600;
    }
    [data-theme="dark"] .messageText {
      fill: #f1f5f9;
      stroke: none;
      font-weight: 500;
    }
    [data-theme="dark"] .messageLine0, [data-theme="dark"] .messageLine1 {
      stroke: #a78bfa;
      stroke-width: 2px;
    }
    [data-theme="dark"] .messageLine0.note, [data-theme="dark"] .messageLine1.note {
      stroke: #f87171;
      stroke-width: 2px;
    }
    [data-theme="dark"] .noteText {
      fill: #f1f5f9;
      font-weight: 500;
    }
    [data-theme="dark"] .note {
      fill: #7f1d1d;
      stroke: #f87171;
      stroke-width: 2px;
    }
    /* Additional styles for sequence diagram text */
    [data-theme="dark"] #sequenceNumber {
      fill: #f0f0f0;
    }
    [data-theme="dark"] text.sequenceText {
      fill: #f0f0f0;
      font-weight: 500;
    }
    [data-theme="dark"] text.loopText, [data-theme="dark"] text.loopText tspan {
      fill: #f0f0f0;
    }
    /* Add a subtle background to message text for better readability */
    [data-theme="dark"] .messageText, [data-theme="dark"] text.sequenceText {
      paint-order: stroke;
      stroke: #1a1a1a;
      stroke-width: 2px;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    /* Force text elements to be properly colored */
    text[text-anchor][dominant-baseline],
    text[text-anchor][alignment-baseline],
    .nodeLabel,
    .edgeLabel,
    .label,
    text {
      fill: #777 !important;
    }

    [data-theme="dark"] text[text-anchor][dominant-baseline],
    [data-theme="dark"] text[text-anchor][alignment-baseline],
    [data-theme="dark"] .nodeLabel,
    [data-theme="dark"] .edgeLabel,
    [data-theme="dark"] .label,
    [data-theme="dark"] text {
      fill: #f0f0f0 !important;
    }

    /* Add clickable element styles with subtle transitions */
    .clickable {
      transition: all 0.3s ease;
    }
    .clickable:hover {
      transform: scale(1.03);
      cursor: pointer;
    }
    .clickable:hover > * {
      filter: brightness(0.95);
    }
  `,
  fontFamily: 'var(--font-geist-sans), var(--font-serif-jp), sans-serif',
  fontSize: 12,
});

// Function to preprocess Mermaid chart and fix common syntax errors
const preprocessMermaidChart = (chart: string): string => {
  let processed = chart;
  
  try {
    // Fix missing closing brackets in node definitions
    // Pattern: A[Text without closing bracket --> B
    processed = processed.replace(/([A-Z]\[[^\]]*?)(?=\s*-->)/g, '$1]');
    
    // Fix malformed arrows - ensure proper spacing
    processed = processed.replace(/([A-Z])\s*-->\s*([A-Z])/g, '$1 --> $2');
    
    // Fix node definitions with parentheses that might break parsing
    // Replace parentheses in node labels with square brackets or quotes
    processed = processed.replace(/\[([^\]]*)\(([^)]*)\)([^\]]*)\]/g, (match, before, inside, after) => {
      // Replace parentheses with square brackets to avoid parse errors
      return `[${before}[${inside}]${after}]`;
    });
    
    // Additional fix: Handle parentheses in node labels more aggressively
    // Look for patterns like: A[Text (content) more text]
    processed = processed.replace(/(\[[^\]]*)\(([^)]*)\)([^\]]*\])/g, (match, before, inside, after) => {
      // Replace with safer alternatives
      return `${before} - ${inside}${after}`;
    });
    
    // Fix special characters that can break parsing
    processed = processed.replace(/\[([^\]]*)\&([^\]]*)\]/g, '[$1and$2]');
    processed = processed.replace(/\[([^\]]*)<([^\]]*)\]/g, '[$1 less than $2]');
    processed = processed.replace(/\[([^\]]*)\>([^\]]*)\]/g, '[$1 greater than $2]');
    
    // Ensure proper graph declaration
    if (!processed.trim().startsWith('graph ') && 
        !processed.trim().startsWith('flowchart ') && 
        !processed.trim().startsWith('sequenceDiagram') &&
        !processed.trim().startsWith('classDiagram') &&
        !processed.trim().startsWith('erDiagram')) {
      processed = `graph TD\n${processed}`;
    }
    
    // Remove any trailing incomplete lines that might cause parsing errors
    const lines = processed.split('\n');
    const validLines = lines.filter(line => {
      const trimmed = line.trim();
      // Keep empty lines and comments
      if (!trimmed || trimmed.startsWith('%%')) return true;
      // Keep graph declarations
      if (trimmed.startsWith('graph ') || trimmed.startsWith('flowchart ') || 
          trimmed.startsWith('sequenceDiagram') || trimmed.startsWith('classDiagram')) return true;
      
      // More comprehensive validation for flowchart lines
      // Valid node definition: A[Label]
      if (trimmed.match(/^[A-Za-z0-9_]+\[[^\]]*\]$/)) return true;
      // Valid connection: A --> B or A[Label] --> B[Label]
      if (trimmed.match(/^[A-Za-z0-9_]+(\[[^\]]*\])?\s*-->\s*[A-Za-z0-9_]+(\[[^\]]*\])?$/)) return true;
      // Valid arrow with label: A -->|label| B
      if (trimmed.match(/^[A-Za-z0-9_]+\s*-->\|[^|]*\|\s*[A-Za-z0-9_]+$/)) return true;
      
      // Sequence diagram elements
      if (trimmed.includes('participant ') || trimmed.includes('Note ') || 
          trimmed.match(/^[A-Za-z0-9_]+-?-?>[A-Za-z0-9_]+:/)) return true;
      
      // Style definitions
      if (trimmed.startsWith('style ') || trimmed.startsWith('class ')) return true;
      
      // Subgraph definitions
      if (trimmed.startsWith('subgraph ') || trimmed === 'end') return true;
      
      // If line contains problematic characters or patterns, reject it
      if (trimmed.includes('PS') && !trimmed.includes('[') && !trimmed.includes(']')) return false;
      if (trimmed.match(/[^\w\s\[\]().,:-]/)) return false;
      
      // Be more permissive for other valid-looking content
      return trimmed.length > 0;
    });
    
    processed = validLines.join('\n');
    
    // Final cleanup - remove any duplicate graph declarations
    const graphDeclarations = processed.match(/^(graph |flowchart |sequenceDiagram|classDiagram)/gm);
    if (graphDeclarations && graphDeclarations.length > 1) {
      // Keep only the first declaration
      const firstDeclaration = graphDeclarations[0];
      processed = processed.replace(/^(graph |flowchart |sequenceDiagram|classDiagram).*$/gm, '');
      processed = `${firstDeclaration}\n${processed}`;
    }
    
    // Final sanitization pass - remove any remaining problematic characters
    processed = processed
      .replace(/[""'']/g, '"')  // Normalize quotes
      .replace(/[—–]/g, '-')   // Normalize dashes
      .replace(/\s+/g, ' ')    // Normalize whitespace
      .replace(/\n\s*\n/g, '\n'); // Remove empty lines
    
  } catch (error) {
    console.warn('Error preprocessing Mermaid chart:', error);
    // Return original chart if preprocessing fails
    return chart;
  }
  
  return processed.trim();
};

interface MermaidProps {
  chart: string;
  className?: string;
  zoomingEnabled?: boolean;
}

// Full screen modal component for the diagram
const FullScreenModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}> = ({ isOpen, onClose, children }) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  // Handle click outside to close
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
    }

    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, [isOpen, onClose]);

  // Reset zoom when modal opens
  useEffect(() => {
    if (isOpen) {
      setZoom(1);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 p-4">
      <div
        ref={modalRef}
        className="bg-[var(--card-bg)] rounded-lg shadow-custom max-w-5xl max-h-[90vh] w-full overflow-hidden flex flex-col card-japanese"
      >
        {/* Modal header with controls */}
        <div className="flex items-center justify-between p-4 border-b border-[var(--border-color)]">
          <div className="font-medium text-[var(--foreground)] font-serif">図表表示</div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
                className="text-[var(--foreground)] hover:bg-[var(--accent-primary)]/10 p-2 rounded-md border border-[var(--border-color)] transition-colors"
                aria-label="Zoom out"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                  <line x1="8" y1="11" x2="14" y2="11"></line>
                </svg>
              </button>
              <span className="text-sm text-[var(--muted)]">{Math.round(zoom * 100)}%</span>
              <button
                onClick={() => setZoom(Math.min(2, zoom + 0.1))}
                className="text-[var(--foreground)] hover:bg-[var(--accent-primary)]/10 p-2 rounded-md border border-[var(--border-color)] transition-colors"
                aria-label="Zoom in"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                  <line x1="11" y1="8" x2="11" y2="14"></line>
                  <line x1="8" y1="11" x2="14" y2="11"></line>
                </svg>
              </button>
              <button
                onClick={() => setZoom(1)}
                className="text-[var(--foreground)] hover:bg-[var(--accent-primary)]/10 p-2 rounded-md border border-[var(--border-color)] transition-colors"
                aria-label="Reset zoom"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"></path>
                  <path d="M21 3v5h-5"></path>
                </svg>
              </button>
            </div>
            <button
              onClick={onClose}
              className="text-[var(--foreground)] hover:bg-[var(--accent-primary)]/10 p-2 rounded-md border border-[var(--border-color)] transition-colors"
              aria-label="Close"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        </div>

        {/* Modal content with zoom */}
        <div className="overflow-auto p-6 flex-1 flex items-center justify-center bg-[var(--background)]/50">
          <div
            style={{
              transform: `scale(${zoom})`,
              transformOrigin: 'center center',
              transition: 'transform 0.3s ease-out'
            }}
          >
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

const Mermaid: React.FC<MermaidProps> = ({ chart, className = '', zoomingEnabled = false }) => {
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const mermaidRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const idRef = useRef(`mermaid-${Math.random().toString(36).substring(2, 9)}`);
  const isDarkModeRef = useRef(
    typeof window !== 'undefined' &&
    window.matchMedia &&
    window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  // Initialize pan-zoom functionality when SVG is rendered
  useEffect(() => {
    if (svg && zoomingEnabled && containerRef.current) {
      const initializePanZoom = async () => {
        const svgElement = containerRef.current?.querySelector("svg");
        if (svgElement) {
          // Remove any max-width constraints
          svgElement.style.maxWidth = "none";
          svgElement.style.width = "100%";
          svgElement.style.height = "100%";

          try {
            // Dynamically import svg-pan-zoom only when needed in the browser
            const svgPanZoom = (await import("svg-pan-zoom")).default;

            svgPanZoom(svgElement, {
              zoomEnabled: true,
              controlIconsEnabled: true,
              fit: true,
              center: true,
              minZoom: 0.1,
              maxZoom: 10,
              zoomScaleSensitivity: 0.3,
            });
          } catch (error) {
            console.error("Failed to load svg-pan-zoom:", error);
          }
        }
      };

      // Wait for the SVG to be rendered
      setTimeout(() => {
        void initializePanZoom();
      }, 100);
    }
  }, [svg, zoomingEnabled]);

  useEffect(() => {
    if (!chart) return;

    let isMounted = true;

    const renderChart = async () => {
      if (!isMounted) return;

      let processedChart = '';
      try {
        setError(null);
        setSvg('');

        // Preprocess chart to fix common syntax errors
        processedChart = preprocessMermaidChart(chart);
        const { svg: renderedSvg } = await mermaid.render(idRef.current, processedChart);

        if (!isMounted) return;

        let processedSvg = renderedSvg;
        if (isDarkModeRef.current) {
          processedSvg = processedSvg.replace('<svg ', '<svg data-theme="dark" ');
        }

        setSvg(processedSvg);

        // Call mermaid.contentLoaded to ensure proper initialization
        setTimeout(() => {
          mermaid.contentLoaded();
        }, 50);
      } catch (err) {
        console.error('Mermaid rendering error:', err);
        console.log('Original chart:', chart);
        console.log('Processed chart:', processedChart);

        // Try a simplified fallback approach
        let fallbackAttempted = false;
        if (processedChart.includes('[') && processedChart.includes(']')) {
          try {
            // Create a super-simplified diagram by extracting just node names
            const nodeMatches = processedChart.match(/([A-Za-z0-9_]+)\[([^\]]+)\]/g);
            const connectionMatches = processedChart.match(/([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)/g);
            
            if (nodeMatches && nodeMatches.length > 0) {
              let simplifiedChart = 'graph TD\n';
              
              // Add sanitized node definitions
              nodeMatches.forEach(match => {
                const nodeMatch = match.match(/([A-Za-z0-9_]+)\[([^\]]+)\]/);
                if (nodeMatch) {
                  const nodeId = nodeMatch[1];
                  const nodeLabel = nodeMatch[2]
                    .replace(/[()]/g, '')  // Remove parentheses
                    .replace(/[^\w\s-]/g, '') // Keep only word chars, spaces, hyphens
                    .substring(0, 50);  // Limit length
                  simplifiedChart += `    ${nodeId}[${nodeLabel}]\n`;
                }
              });
              
              // Add connections
              if (connectionMatches) {
                connectionMatches.forEach(match => {
                  const connectionMatch = match.match(/([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)/);
                  if (connectionMatch) {
                    simplifiedChart += `    ${connectionMatch[1]} --> ${connectionMatch[2]}\n`;
                  }
                });
              }
              
              // Try to render the simplified chart
              const { svg: fallbackSvg } = await mermaid.render(idRef.current + '_fallback', simplifiedChart);
              if (isMounted) {
                setSvg(fallbackSvg);
                console.log('Successfully rendered simplified fallback diagram');
                return;
              }
              fallbackAttempted = true;
            }
          } catch (fallbackErr) {
            console.log('Fallback diagram also failed:', fallbackErr);
          }
        }

        const errorMessage = err instanceof Error ? err.message : String(err);

        if (isMounted) {
          setError(`Failed to render diagram: ${errorMessage}`);

          if (mermaidRef.current) {
            mermaidRef.current.innerHTML = `
              <div class="text-red-500 dark:text-red-400 text-xs mb-1">Syntax error in diagram${fallbackAttempted ? ' (fallback also failed)' : ''}</div>
              <details class="text-xs">
                <summary class="cursor-pointer hover:text-blue-500">Show original chart</summary>
                <pre class="mt-2 overflow-auto p-2 bg-gray-100 dark:bg-gray-800 rounded border">${chart}</pre>
              </details>
              <details class="text-xs mt-2">
                <summary class="cursor-pointer hover:text-blue-500">Show processed chart</summary>
                <pre class="mt-2 overflow-auto p-2 bg-gray-100 dark:bg-gray-800 rounded border">${processedChart}</pre>
              </details>
            `;
          }
        }
      }
    };

    renderChart();

    return () => {
      isMounted = false;
    };
  }, [chart]);

  const handleDiagramClick = () => {
    if (!error && svg) {
      setIsFullscreen(true);
    }
  };

  if (error) {
    return (
      <div className={`border border-[var(--highlight)]/30 rounded-md p-4 bg-[var(--highlight)]/5 ${className}`}>
        <div className="flex items-center mb-3">
          <div className="text-[var(--highlight)] text-xs font-medium flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            図表レンダリングエラー
          </div>
        </div>
        <div ref={mermaidRef} className="text-xs overflow-auto"></div>
        <div className="mt-3 text-xs text-[var(--muted)] font-serif">
          図表に構文エラーがあり、レンダリングできません。
        </div>
      </div>
    );
  }

  if (!svg) {
    return (
      <div className={`flex justify-center items-center p-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-[var(--accent-primary)]/70 rounded-full animate-pulse"></div>
          <div className="w-2 h-2 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-75"></div>
          <div className="w-2 h-2 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-150"></div>
          <span className="text-[var(--muted)] text-xs ml-2 font-serif">図表を描画中...</span>
        </div>
      </div>
    );
  }

  return (
    <>
      <div
        ref={containerRef}
        className={`w-full max-w-full ${zoomingEnabled ? "h-[600px] p-4" : ""}`}
      >
        <div
          className={`relative group ${zoomingEnabled ? "h-full rounded-lg border-2 border-black" : ""}`}
        >
          <div
            className={`flex justify-center overflow-auto text-center my-2 cursor-pointer hover:shadow-md transition-shadow duration-200 rounded-md ${className} ${zoomingEnabled ? "h-full" : ""}`}
            dangerouslySetInnerHTML={{ __html: svg }}
            onClick={zoomingEnabled ? undefined : handleDiagramClick}
            title={zoomingEnabled ? undefined : "Click to view fullscreen"}
          />

          {!zoomingEnabled && (
            <div className="absolute top-2 right-2 bg-gray-700/70 dark:bg-gray-900/70 text-white p-1.5 rounded-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center gap-1.5 text-xs shadow-md pointer-events-none">
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <line x1="11" y1="8" x2="11" y2="14"></line>
                <line x1="8" y1="11" x2="14" y2="11"></line>
              </svg>
              <span>Click to zoom</span>
            </div>
          )}
        </div>
      </div>

      {!zoomingEnabled && (
        <FullScreenModal
          isOpen={isFullscreen}
          onClose={() => setIsFullscreen(false)}
        >
          <div dangerouslySetInnerHTML={{ __html: svg }} />
        </FullScreenModal>
      )}
    </>
  );
};



export default Mermaid;