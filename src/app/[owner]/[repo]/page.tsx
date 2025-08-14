/* eslint-disable @typescript-eslint/no-unused-vars */
'use client';

import Ask from '@/components/Ask';
import Markdown from '@/components/Markdown';
// ModelSelectionModal removed - not used in this page
import ThemeToggle from '@/components/theme-toggle';
import WikiTreeView from '@/components/WikiTreeView';
import { useLanguage } from '@/contexts/LanguageContext';
import { RepoInfo } from '@/types/repoinfo';
import getRepoUrl from '@/utils/getRepoUrl';
import { extractUrlDomain, extractUrlPath } from '@/utils/urlDecoder';
import Link from 'next/link';
import { useParams, useSearchParams } from 'next/navigation';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaBitbucket, FaBookOpen, FaComments, FaDownload, FaExclamationTriangle, FaFileExport, FaFolder, FaHome, FaSync, FaTimes } from 'react-icons/fa';
import { createChatWebSocket, closeWebSocket, ChatCompletionRequest, createCleanWikiWebSocket } from '@/utils/websocketClient';

// Type definitions
interface WikiSection {
  id: string;
  title: string;
  pages: string[];
  subsections?: string[];
}

interface WikiPage {
  id: string;
  title: string;
  content: string;
  filePaths: string[];
  importance: 'high' | 'medium' | 'low';
  relatedPages: string[];
  parentId?: string;
  isSection?: boolean;
  children?: string[];
}

interface WikiStructure {
  id: string;
  title: string;
  description: string;
  pages: WikiPage[];
  sections: WikiSection[];
  rootSections: string[];
}

type PageState =
  | 'idle'
  | 'checking_cache'
  | 'fetching_repo_structure'
  | 'determining_wiki_structure'
  | 'generating_page_content'
  | 'ready'
  | 'error';


// Styles remain the same
const wikiStyles = `
  /* Main wiki content uses consistent fonts */
  .prose { 
    font-family: var(--font-geist-sans), sans-serif;
    @apply text-[var(--foreground)] max-w-none;
  }
  .prose code { 
    font-family: var(--font-geist-mono), monospace;
    @apply bg-[var(--background)]/70 px-1.5 py-0.5 rounded text-xs border border-[var(--border-color)]; 
  }
  .prose pre { 
    font-family: var(--font-geist-mono), monospace;
    @apply bg-[var(--background)]/80 text-[var(--foreground)] rounded-md p-4 overflow-x-auto border border-[var(--border-color)] shadow-sm; 
  }
  .prose h1, .prose h2, .prose h3, .prose h4 { 
    font-family: var(--font-serif-jp), serif;
    @apply text-[var(--foreground)] font-medium;
  }
  .prose p { 
    font-family: var(--font-geist-sans), sans-serif;
    @apply text-[var(--foreground)] leading-relaxed; 
  }
  .prose a { @apply text-[var(--accent-primary)] hover:text-[var(--highlight)] transition-colors no-underline border-b border-[var(--border-color)] hover:border-[var(--accent-primary)]; }
  .prose blockquote { @apply border-l-4 border-[var(--accent-primary)]/30 bg-[var(--background)]/30 pl-4 py-1 italic; }
  .prose ul, .prose ol { 
    font-family: var(--font-geist-sans), sans-serif;
    @apply text-[var(--foreground)]; 
  }
  .prose table { @apply border-collapse border border-[var(--border-color)]; }
  .prose th { 
    font-family: var(--font-serif-jp), serif;
    @apply bg-[var(--background)]/70 text-[var(--foreground)] p-2 border border-[var(--border-color)] font-medium; 
  }
  .prose td { 
    font-family: var(--font-geist-sans), sans-serif;
    @apply p-2 border border-[var(--border-color)]; 
  }
  /* Ensure buttons and UI elements use consistent fonts */
  button, .button {
    font-family: var(--font-geist-sans), sans-serif;
  }
`;

// --- Helper functions for API calls ---
const createBitbucketHeaders = (bitbucketToken: string): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (bitbucketToken) {
    // Bitbucket app password is passed as a token
    // The backend will combine it with the username from the repo URL
    headers['Authorization'] = `Bearer ${bitbucketToken}`;
  }
  return headers;
};


export default function RepoWikiPage() {
  // --- Hooks and Params ---
  const params = useParams();
  const searchParams = useSearchParams();
  const { messages, language } = useLanguage();

  // --- Repo Info ---
  const owner = params.owner as string;
  const repo = params.repo as string;
  const token = searchParams.get('token') || '';
  const repoUrl = searchParams.get('repo_url') ? decodeURIComponent(searchParams.get('repo_url') || '') : undefined;
  const repoType = repoUrl?.includes('bitbucket.org') ? 'bitbucket' : searchParams.get('type') || 'bitbucket';

  const repoInfo = useMemo<RepoInfo>(() => ({
    owner,
    repo,
    type: repoType,
    token: token || null,
    localPath: searchParams.get('local_path') ? decodeURIComponent(searchParams.get('local_path') || '') : null,
    repoUrl: repoUrl || null
  }), [owner, repo, repoType, searchParams, token, repoUrl]);

  // --- State Machine ---
  const [pageState, setPageState] = useState<PageState>('idle');
  const [loadingMessage, setLoadingMessage] = useState<string | undefined>('');
  const [error, setError] = useState<string | null>(null);

  // --- Data State ---
  const [wikiStructure, setWikiStructure] = useState<WikiStructure | undefined>();
  const [generatedPages, setGeneratedPages] = useState<Record<string, WikiPage>>({});
  const [pagesInProgress, setPagesInProgress] = useState(new Set<string>());
  const [currentPageId, setCurrentPageId] = useState<string | undefined>();
  const [fileTree, setFileTree] = useState('');
  const [readme, setReadme] = useState('');
  const [defaultBranch, setDefaultBranch] = useState<string>('main');

  // --- Modal and UI State ---
  const [isAskModalOpen, setIsAskModalOpen] = useState(false);
  // ModelSelectionModal state removed - not used in this page
  // Wiki type is now fixed to concise mode (comprehensive mode removed)
  const isComprehensiveView = false;
  const [isExportDropdownOpen, setIsExportDropdownOpen] = useState(false);

  // --- Model Config State ---
  const [selectedProvider, setSelectedProvider] = useState(searchParams.get('provider') || 'vllm');
  const [selectedModel, setSelectedModel] = useState(searchParams.get('model') || '/app/models/Qwen2.5-VL-7B-Instruct');
  const [isCustomModel, setIsCustomModel] = useState(searchParams.get('is_custom_model') === 'true');
  const [customModel, setCustomModel] = useState(searchParams.get('custom_model') || '');

  // --- Refs ---
  const webSocketRef = useRef<WebSocket | null>(null);
  const pageQueueRef = useRef<WikiPage[]>([]);
  const activeRequestsRef = useRef(0);
  const initialLoadRef = useRef(true);
  const askComponentRef = useRef<{ clearConversation: () => void } | null>(null);


  // --- Derived State ---
  const isLoading = pageState !== 'ready' && pageState !== 'error';

  // --- Utility Functions ---
  const generateFileUrl = useCallback((filePath: string): string => {
    if (repoInfo.type === 'local' || !repoInfo.repoUrl) return filePath;
    const baseUrl = repoInfo.repoUrl;
    try {
      const url = new URL(baseUrl);
      if (url.hostname.includes('bitbucket')) return `${baseUrl}/src/${defaultBranch}/${filePath}`;
    } catch { /* fallback */ }
    return filePath;
  }, [repoInfo, defaultBranch]);

  const handleError = (errorMessage: string, state: PageState = 'error') => {
    console.error(errorMessage);
    setError(errorMessage);
    setPageState(state as PageState);
    setLoadingMessage(undefined);
  };

  // --- Export and Save Handlers ---
  const handleExportWiki = async (format: 'markdown' | 'json') => {
    try {
      setIsExportDropdownOpen(false);
      
      if (!wikiStructure || Object.keys(generatedPages).length === 0) {
        alert('No wiki content to export. Please generate wiki pages first.');
        return;
      }

      // Prepare export data
      const exportData = {
        repo_url: repoInfo.repoUrl || `${repoInfo.type}:${repoInfo.owner}/${repoInfo.repo}`,
        pages: Object.values(generatedPages),
        format: format
      };

      console.log('Exporting wiki:', exportData);

      // Call the export API
      const response = await fetch('/api/export/wiki', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(exportData),
      });

      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }

      // Download the file
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      
      // Get filename from response headers or create one
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${repoInfo.owner}_${repoInfo.repo}_wiki.${format}`;
      if (contentDisposition) {
        const matches = /filename="([^"]*)"/.exec(contentDisposition);
        if (matches && matches[1]) {
          filename = matches[1];
        }
      }
      
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      console.log(`Wiki exported as ${format} successfully`);
    } catch (error) {
      console.error('Export error:', error);
      alert(`Failed to export wiki: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleSaveWiki = async () => {
    try {
      if (!wikiStructure || Object.keys(generatedPages).length === 0) {
        alert('No wiki content to save. Please generate wiki pages first.');
        return;
      }

      // Prepare save data for the wiki cache API
      const saveData = {
        repo: repoInfo,
        language: language,
        wiki_structure: wikiStructure,
        generated_pages: generatedPages,
        provider: selectedProvider || 'vllm',
        model: selectedModel || '/app/models/Qwen2.5-VL-7B-Instruct'
      };

      console.log('Saving wiki to cache:', saveData);

      // Call the save API
      const response = await fetch('/api/wiki_cache', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(saveData),
      });

      if (!response.ok) {
        throw new Error(`Save failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Wiki saved successfully:', result);
      alert('Wiki saved successfully! You can access it later from the processed projects page.');
    } catch (error) {
      console.error('Save error:', error);
      alert(`Failed to save wiki: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  // --- State Machine Effects ---

  // Effect for initial load and refresh - check cache first
  useEffect(() => {
    if (initialLoadRef.current) {
      initialLoadRef.current = false;
      // Check if we should load from cache first
      const shouldLoadFromCache = !searchParams.get('force_regenerate');
      if (shouldLoadFromCache) {
        setPageState('checking_cache');
      } else {
        setPageState('fetching_repo_structure');
      }
    }
  }, [searchParams]);

  // 0. Checking Cache
  useEffect(() => {
    if (pageState !== 'checking_cache') return;

    const checkCache = async () => {
      setLoadingMessage('Checking for saved wiki...');
      try {
        const params = new URLSearchParams({
          owner: repoInfo.owner,
          repo: repoInfo.repo,
          repo_type: repoInfo.type,
          language: searchParams.get('language') || language
        });

        const response = await fetch(`/api/wiki_cache?${params}`);
        
        if (response.ok) {
          const cachedData = await response.json();
          
          // Load cached wiki structure and pages
          if (cachedData.wiki_structure && cachedData.generated_pages) {
            console.log('Loading wiki from cache:', cachedData);
            setWikiStructure(cachedData.wiki_structure);
            setGeneratedPages(cachedData.generated_pages);
            setPagesInProgress(new Set());
            setPageState('ready');
            setLoadingMessage(undefined);
            return;
          }
        }
        
        // If cache miss or error, proceed with normal generation
        console.log('Cache miss or error, proceeding with generation');
        setPageState('fetching_repo_structure');
        
      } catch (error) {
        console.log('Cache check failed, proceeding with generation:', error);
        setPageState('fetching_repo_structure');
      }
    };

    checkCache();
  }, [pageState, repoInfo, language, searchParams]);

  // 1. Fetching Repository Structure
  useEffect(() => {
    if (pageState !== 'fetching_repo_structure') return;

    const fetchRepoData = async () => {
      setLoadingMessage(messages.loading?.fetchingStructure || 'Fetching repository structure...');
      let fileTreeData = '';
      let readmeContent = '';

      try {
        if (repoInfo.type === 'local' && repoInfo.localPath) {
            const response = await fetch(`/local_repo/structure?path=${encodeURIComponent(repoInfo.localPath)}`);
            if (!response.ok) {
                const errorData = await response.text();
                throw new Error(`Local repository API error (${response.status}): ${errorData}`);
            }
            const data = await response.json();
            fileTreeData = data.file_tree;
            readmeContent = data.readme;
            setDefaultBranch('main'); // Local repos don't have a concept of a remote branch
        } else if (repoInfo.type === 'bitbucket') {
            // For Bitbucket repositories, we skip the structure fetching step
            // The backend will handle repository cloning and analysis during wiki generation
            // Set minimal data to proceed to wiki generation
            fileTreeData = 'Repository structure will be analyzed by backend during processing';
            readmeContent = 'README content will be retrieved during backend processing';
            setDefaultBranch('main'); // Default branch
        }

        setFileTree(fileTreeData);
        setReadme(readmeContent);
        setPageState('determining_wiki_structure');
      } catch (err) {
        handleError(err instanceof Error ? err.message : 'Unknown error fetching repository data.');
      }
    };

    fetchRepoData();
  }, [pageState, repoInfo, token, owner, repo, messages.loading]);


  // 2. Generate Comprehensive Wiki - Use clean backend API 
  useEffect(() => {
    if (pageState !== 'determining_wiki_structure') return;

    setLoadingMessage('Generating comprehensive wiki documentation...');

    // Use clean architecture - only send repository details, backend handles all prompts
    const cleanRequest = {
      repo_url: getRepoUrl(repoInfo),
      repo_type: repoInfo.type,
      provider: selectedProvider || 'vllm',
      model: isCustomModel ? customModel : (selectedModel || '/app/models/Qwen2.5-VL-7B-Instruct'),
      token: token,
      local_path: repoInfo.localPath,
      language: language
    };

    let responseBuffer = '';
    let chunkCount = 0;
    closeWebSocket(webSocketRef.current);
    
    // Connect to the clean wiki generation WebSocket endpoint
    webSocketRef.current = createCleanWikiWebSocket(
      cleanRequest,
      (chunk) => { 
        chunkCount++;
        responseBuffer += chunk;
        if (chunkCount % 1000 === 0) {
          console.log(`Received ${chunkCount} chunks, buffer length: ${responseBuffer.length}`);
        }
        // Log first few chunks to see what we're getting
        if (chunkCount <= 5) {
          console.log(`Chunk ${chunkCount}:`, JSON.stringify(chunk.substring(0, 100)));
        }
        // Log chunks that contain wiki_structure
        if (chunk.includes('wiki_structure')) {
          console.log(`Wiki structure found in chunk ${chunkCount}:`, chunk.substring(0, 200));
        }
      },
      (status, msg) => {
        console.log(`Structure status: ${status} - ${msg}`);
        if (status === 'error') {
          handleError(`WebSocket error while determining structure: ${msg}`);
        } else if (status === 'timeout') {
          handleError(`Connection timeout: ${msg}. Please check if the API server is running on port 8001.`);
        }
      },
      (err) => handleError(`WebSocket error while determining structure: ${err}`),
      () => { // onComplete
        try {
          console.log(`Comprehensive wiki generation completed. Length: ${responseBuffer.length} characters`);
          
          // Validate response completeness
          if (!responseBuffer.trim()) {
            throw new Error("Empty response received from server.");
          }

          // Parse the comprehensive response into logical wiki sections
          console.log(`Processing comprehensive wiki response: ${responseBuffer.length} characters`);
          
          // Split the response into logical sections based on # headers (DeepWiki chapter format)
          const sectionRegex = /^#\s+(\d+\s*-\s*.+)$/gm;
          const sectionMatches = [...responseBuffer.matchAll(sectionRegex)];
          
          console.log(`Found ${sectionMatches.length} major sections with DeepWiki format`);
          
          const pages: WikiPage[] = [];
          const sections: WikiSection[] = [];
          
          // If no DeepWiki-style chapters found, fallback to old ## header parsing
          if (sectionMatches.length === 0) {
            console.log("No DeepWiki chapters found, trying fallback ## header parsing");
            const fallbackRegex = /^##\s+(.+)$/gm;
            const fallbackMatches = [...responseBuffer.matchAll(fallbackRegex)];
            console.log(`Found ${fallbackMatches.length} sections with fallback parsing`);
            
            if (fallbackMatches.length > 0) {
              // Use the old parsing logic as fallback
              const firstSectionIndex = fallbackMatches[0].index!;
              const overviewContent = responseBuffer.substring(0, firstSectionIndex).trim();
              
              if (overviewContent) {
                pages.push({
                  id: 'overview',
                  title: 'System Overview',
                  content: overviewContent,
                  filePaths: [],
                  importance: 'high' as const,
                  relatedPages: []
                });
              }
              
              // Process fallback sections
              for (let i = 0; i < fallbackMatches.length; i++) {
                const match = fallbackMatches[i];
                const title = match[1];
                const id = title.toLowerCase()
                  .replace(/[^a-z0-9\s]/g, '')
                  .replace(/\s+/g, '-')
                  .replace(/^-|-$/g, '');
                
                const startIndex = match.index!;
                const endIndex = i < fallbackMatches.length - 1 ? fallbackMatches[i + 1].index! : responseBuffer.length;
                const content = responseBuffer.substring(startIndex, endIndex).trim();
                
                const hasArchitecture = content.toLowerCase().includes('architecture') || content.includes('mermaid');
                const hasImplementation = content.toLowerCase().includes('implementation') || content.toLowerCase().includes('code');
                const importance = hasArchitecture || hasImplementation ? 'high' as const : 'medium' as const;
                
                pages.push({
                  id,
                  title,
                  content,
                  filePaths: [],
                  importance,
                  relatedPages: []
                });
              }
            }
          } else {
            // DeepWiki format detected - process numbered chapters
            console.log("Processing DeepWiki-style numbered chapters");
            
            // Add the main overview page with everything before the first chapter
            const firstSectionIndex = sectionMatches[0].index!;
            const overviewContent = responseBuffer.substring(0, firstSectionIndex).trim();
            
            if (overviewContent) {
              pages.push({
                id: 'overview',
                title: 'System Overview',
                content: overviewContent,
                filePaths: [],
                importance: 'high' as const,
                relatedPages: []
              });
            }
            
            // Process each major section as a separate page
            for (let i = 0; i < sectionMatches.length; i++) {
              const match = sectionMatches[i];
              const fullTitle = match[1]; // e.g., "1 - Getting Started"
              const title = fullTitle.replace(/^\d+\s*-\s*/, '').trim(); // Extract "Getting Started"
              const id = title.toLowerCase()
                .replace(/[^a-z0-9\s]/g, '')
                .replace(/\s+/g, '-')
                .replace(/^-|-$/g, '');
              
              // Extract content for this section
              const startIndex = match.index!;
              const endIndex = i < sectionMatches.length - 1 ? sectionMatches[i + 1].index! : responseBuffer.length;
              const content = responseBuffer.substring(startIndex, endIndex).trim();
              
              // Determine importance based on section content
              const hasArchitecture = content.toLowerCase().includes('architecture') || content.includes('mermaid');
              const hasImplementation = content.toLowerCase().includes('implementation') || content.toLowerCase().includes('code');
              const importance = hasArchitecture || hasImplementation ? 'high' as const : 'medium' as const;
              
              pages.push({
                id,
                title,
                content,
                filePaths: [],
                importance,
                relatedPages: []
              });
            }
            
            // Create a single section with pages in their original order
            // This preserves the logical flow from the AI-generated comprehensive response
            const allPageIds = pages.map(p => p.id);
            
            sections.push({
              id: 'main-documentation',
              title: 'Documentation',
              pages: allPageIds,
              subsections: []
            });
          }
          
          // Final validation - if no pages created, create single comprehensive page
          if (pages.length === 0) {
            console.log("No sections found with either method, creating single comprehensive wiki page");
            pages.push({
              id: 'comprehensive-overview',
              title: `${repo} Comprehensive Documentation`,
              content: responseBuffer,
              filePaths: [],
              importance: 'high' as const,
              relatedPages: []
            });
            
            sections.push({
              id: 'main',
              title: 'Documentation',
              pages: ['comprehensive-overview'],
              subsections: []
            });
          }
          
          // Create the final wiki structure
          const structure: WikiStructure = {
            id: 'wiki',
            title: `${repo} Wiki`,
            description: `Comprehensive documentation for ${owner}/${repo}`,
            pages,
            sections,
            rootSections: sections.map(s => s.id)
          };

          // Create the generated pages map
          const generatedPagesMap: Record<string, WikiPage> = {};
          pages.forEach(page => {
            generatedPagesMap[page.id] = page;
          });

          console.log(`Created wiki with ${pages.length} pages and ${sections.length} sections`);
          
          setWikiStructure(structure);
          setGeneratedPages(generatedPagesMap);
          setPagesInProgress(new Set());
          setPageState('ready');
          
        } catch (err) {
          handleError(err instanceof Error ? err.message : 'Failed to process comprehensive wiki response.');
        }
      }
    );
  }, [pageState, repoInfo, fileTree, readme, messages.loading, isComprehensiveView, language, selectedProvider, selectedModel, isCustomModel, customModel, token]);

  // Set current page when wiki structure is ready
  useEffect(() => {
    if (wikiStructure && wikiStructure.pages.length > 0 && !currentPageId) {
      setCurrentPageId(wikiStructure.pages[0].id);
    }
  }, [wikiStructure, currentPageId]);


  // Page generation is now handled in the single wiki generation request above
  // No separate page generation step needed

  // --- UI Event Handlers ---
  const handleRefresh = () => {
    setError(null);
    setWikiStructure(undefined);
    setGeneratedPages({});
    setPagesInProgress(new Set());
    pageQueueRef.current = [];
    activeRequestsRef.current = 0;
    // Force regeneration by skipping cache
    setPageState('fetching_repo_structure');
    
    // Add force_regenerate parameter to URL to bypass cache on page reload
    const currentUrl = new URL(window.location.href);
    currentUrl.searchParams.set('force_regenerate', 'true');
    window.history.replaceState({}, '', currentUrl.toString());
  };

  const handlePageSelect = (pageId: string) => {
    setCurrentPageId(pageId);
  };

  // close the modal when escape is pressed
  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsAskModalOpen(false);
      }
    };
    if (isAskModalOpen) {
      window.addEventListener('keydown', handleEsc);
    }
    return () => {
      window.removeEventListener('keydown', handleEsc);
    };
  }, [isAskModalOpen]);


  // --- Render Logic ---
  const currentPage = useMemo(() => {
    if (!wikiStructure || !currentPageId) return null;
    return generatedPages[currentPageId] || wikiStructure.pages.find(p => p.id === currentPageId);
  }, [currentPageId, wikiStructure, generatedPages]);

  if (pageState !== 'ready') {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        {error ? (
          <>
            <div className="text-red-500 text-lg">Error: {error}</div>
            <button onClick={handleRefresh} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded">
              Try Again
            </button>
          </>
        ) : (
          <>
            <div className="animate-pulse text-lg">{loadingMessage}</div>
            {pageState === 'generating_page_content' && wikiStructure && (
              <div className="w-full max-w-md mt-4">
                <div className="bg-gray-200 rounded-full h-2.5">
                  <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${(1 - pagesInProgress.size / (wikiStructure.pages.length || 1)) * 100}%` }}></div>
                </div>
                <p className="text-center mt-2">{wikiStructure.pages.length - pagesInProgress.size} / {wikiStructure.pages.length} pages complete</p>
              </div>
            )}
          </>
        )}
      </div>
    );
  }

  return (
    <div className="h-screen bg-white dark:bg-gray-900 p-4 md:p-8 flex flex-col">
      <style>{wikiStyles}</style>
      <header className="max-w-[90%] xl:max-w-[1400px] mx-auto mb-8 h-fit w-full">
        <div className="flex items-center justify-between">
            <Link href="/" className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-1.5 transition-colors">
              <FaHome /> {messages.repoPage?.home || 'Home'}
            </Link>
            
            <div className="flex items-center gap-3">
              {/* Export Dropdown */}
              <div className="relative export-dropdown">
                <button
                  onClick={() => setIsExportDropdownOpen(!isExportDropdownOpen)}
                  className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors flex items-center gap-2"
                  disabled={!wikiStructure || Object.keys(generatedPages).length === 0}
                >
                  <FaFileExport />
                  Export Wiki
                </button>
                
                {isExportDropdownOpen && (
                  <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg z-50">
                    <button
                      onClick={() => handleExportWiki('markdown')}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
                    >
                      <FaDownload />
                      Export as Markdown
                    </button>
                    <button
                      onClick={() => handleExportWiki('json')}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
                    >
                      <FaDownload />
                      Export as JSON
                    </button>
                  </div>
                )}
              </div>
              
              {/* Save Wiki Button */}
              <button
                onClick={handleSaveWiki}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center gap-2"
                disabled={!wikiStructure || Object.keys(generatedPages).length === 0}
              >
                <FaSync />
                Save Wiki
              </button>
              
              <button
                onClick={handleRefresh}
                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors flex items-center gap-2"
                title="Regenerate wiki (ignore cache)"
              >
                <FaSync />
                Refresh
              </button>
              
              <button
                  onClick={() => setIsAskModalOpen(true)}
                  className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors flex items-center gap-2"
              >
                  <FaComments />
                  {messages.repoPage?.askButton || 'Ask about this repo'}
              </button>
              
              <ThemeToggle />
            </div>
        </div>
      </header>
      <main className="flex-1 max-w-[90%] xl:max-w-[1400px] mx-auto overflow-y-auto grid grid-cols-12 gap-8">
        <aside className="col-span-3">
          {wikiStructure && (
            <WikiTreeView
              wikiStructure={wikiStructure}
              currentPageId={currentPageId}
              onPageSelect={handlePageSelect}
            />
          )}
        </aside>
        <section id="wiki-content" className="col-span-9 overflow-y-auto">
          {currentPage ? (
            <article className="prose max-w-none">
              <h1>{currentPage.title}</h1>
              <Markdown content={currentPage.content} />
            </article>
          ) : (
            <div>Select a page to view its content.</div>
          )}
        </section>
      </main>
      
      {isAskModalOpen && (
        <div className="fixed inset-0 bg-black/50 z-40 flex items-center justify-center">
          <div className="bg-[var(--background)] rounded-lg shadow-2xl w-full max-w-4xl h-[90vh] flex flex-col">
            <div className="p-4 border-b border-[var(--border-color)] flex justify-between items-center">
                <h2 className="text-lg font-semibold">Ask about {repoInfo.owner}/{repoInfo.repo}</h2>
                <button onClick={() => setIsAskModalOpen(false)} className="p-1 rounded-full hover:bg-[var(--background)]/50">
                    <FaTimes />
                </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              <Ask
                repoInfo={repoInfo}
                provider={selectedProvider}
                model={selectedModel}
                isCustomModel={isCustomModel}
                customModel={customModel}
                language={language}
                onRef={(ref) => (askComponentRef.current = ref)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}