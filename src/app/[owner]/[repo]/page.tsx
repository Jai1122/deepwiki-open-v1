/* eslint-disable @typescript-eslint/no-unused-vars */
'use client';

import Ask from '@/components/Ask';
import Markdown from '@/components/Markdown';
import ModelSelectionModal from '@/components/ModelSelectionModal';
import ThemeToggle from '@/components/theme-toggle';
import WikiTreeView from '@/components/WikiTreeView';
import { useLanguage } from '@/contexts/LanguageContext';
import { RepoInfo } from '@/types/repoinfo';
import getRepoUrl from '@/utils/getRepoUrl';
import { extractUrlDomain, extractUrlPath } from '@/utils/urlDecoder';
import Link from 'next/link';
import { useParams, useSearchParams } from 'next/navigation';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaBitbucket, FaBookOpen, FaComments, FaDownload, FaExclamationTriangle, FaFileExport, FaFolder, FaGithub, FaGitlab, FaHome, FaSync, FaTimes } from 'react-icons/fa';
import { createChatWebSocket, closeWebSocket, ChatCompletionRequest } from '@/utils/websocketClient';

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
const createGithubHeaders = (githubToken: string): HeadersInit => {
  const headers: HeadersInit = { 'Accept': 'application/vnd.github.v3+json' };
  if (githubToken) headers['Authorization'] = `Bearer ${githubToken}`;
  return headers;
};

const createGitlabHeaders = (gitlabToken: string): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (gitlabToken) headers['PRIVATE-TOKEN'] = gitlabToken;
  return headers;
};

const createBitbucketHeaders = (bitbucketToken: string): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (bitbucketToken) headers['Authorization'] = `Bearer ${bitbucketToken}`;
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
  const repoType = repoUrl?.includes('bitbucket.org') ? 'bitbucket' : repoUrl?.includes('gitlab.com') ? 'gitlab' : repoUrl?.includes('github.com') ? 'github' : searchParams.get('type') || 'github';

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
  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);
  const [isComprehensiveView, setIsComprehensiveView] = useState(searchParams.get('comprehensive') !== 'false');
  const [isExportDropdownOpen, setIsExportDropdownOpen] = useState(false);

  // --- Model Config State ---
  const [selectedProvider, setSelectedProvider] = useState(searchParams.get('provider') || '');
  const [selectedModel, setSelectedModel] = useState(searchParams.get('model') || '');
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
      if (url.hostname.includes('github')) return `${baseUrl}/blob/${defaultBranch}/${filePath}`;
      if (url.hostname.includes('gitlab')) return `${baseUrl}/-/blob/${defaultBranch}/${filePath}`;
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
        provider: selectedProvider || 'google',
        model: selectedModel || 'gemini-2.0-flash'
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

  // Effect for initial load and refresh
  useEffect(() => {
    if (initialLoadRef.current) {
      initialLoadRef.current = false;
      setPageState('fetching_repo_structure');
    }
  }, []);

  // 1. Fetching Repository Structure
  useEffect(() => {
    if (pageState !== 'fetching_repo_structure') return;

    const fetchRepoData = async () => {
      setLoadingMessage(messages.loading?.fetchingStructure || 'Fetching repository structure...');
      let fileTreeData = '';
      let readmeContent = '';

      try {
        if (repoInfo.type === 'github') {
            const getGithubApiUrl = (repoUrl: string | null): string => {
                if (!repoUrl) return 'https://api.github.com';
                try {
                    const url = new URL(repoUrl);
                    return url.hostname === 'github.com' ? 'https://api.github.com' : `${url.protocol}//${url.hostname}/api/v3`;
                } catch { return 'https://api.github.com'; }
            };
            const githubApiBaseUrl = getGithubApiUrl(repoInfo.repoUrl);
            const repoInfoResponse = await fetch(`${githubApiBaseUrl}/repos/${owner}/${repo}`, { headers: createGithubHeaders(token) });
            const repoData = await repoInfoResponse.json();
            const branch = repoData.default_branch || 'main';
            setDefaultBranch(branch);

            // GitHub's get tree API is limited for large repos, but recursive fetch should get most things.
            // Note: For truly massive repos, this can still be truncated. A more robust solution would use the Git Database API.
            const treeResponse = await fetch(`${githubApiBaseUrl}/repos/${owner}/${repo}/git/trees/${branch}?recursive=1`, { headers: createGithubHeaders(token) });
            if (!treeResponse.ok) throw new Error(`GitHub API error: ${treeResponse.statusText}`);
            const treeData = await treeResponse.json();
            if (treeData.truncated) {
              console.warn("GitHub API response was truncated. The file list may be incomplete.");
            }
            fileTreeData = treeData.tree.filter((item: any) => item.type === 'blob').map((item: any) => item.path).join('\n');

            const readmeResponse = await fetch(`${githubApiBaseUrl}/repos/${owner}/${repo}/readme`, { headers: createGithubHeaders(token) });
            if (readmeResponse.ok) {
                const readmeData = await readmeResponse.json();
                readmeContent = atob(readmeData.content);
            }
        } else if (repoInfo.type === 'gitlab') {
            const projectDomain = extractUrlDomain(repoInfo.repoUrl ?? "https://gitlab.com");
            const projectPath = encodeURIComponent(extractUrlPath(repoInfo.repoUrl ?? '') ?? `${owner}/${repo}`);
            const projectInfoUrl = `${projectDomain}/api/v4/projects/${projectPath}`;
            const headers = createGitlabHeaders(token);

            const projectInfoRes = await fetch(projectInfoUrl, { headers });
            if (!projectInfoRes.ok) throw new Error(`GitLab project info error: ${projectInfoRes.statusText}`);
            const projectInfo = await projectInfoRes.json();
            const branch = projectInfo.default_branch || 'main';
            setDefaultBranch(branch);

            // Implement pagination for GitLab tree
            let allFiles: any[] = [];
            let page = 1;
            while (true) {
              const treeUrl = `${projectInfoUrl}/repository/tree?recursive=true&per_page=100&page=${page}`;
              const treeResponse = await fetch(treeUrl, { headers });
              if (!treeResponse.ok) throw new Error(`GitLab tree error: ${treeResponse.statusText}`);
              const pageData = await treeResponse.json();
              if (pageData.length === 0) break;
              allFiles = allFiles.concat(pageData);
              page++;
            }
            
            fileTreeData = allFiles.filter((item: any) => item.type === 'blob').map((item: any) => item.path).join('\n');

            const readmeUrl = `${projectInfoUrl}/repository/files/README.md/raw?ref=${branch}`;
            const readmeResponse = await fetch(readmeUrl, { headers });
            if (readmeResponse.ok) readmeContent = await readmeResponse.text();
        } else if (repoInfo.type === 'local' && repoInfo.localPath) {
            const response = await fetch(`/local_repo/structure?path=${encodeURIComponent(repoInfo.localPath)}`);
            if (!response.ok) {
                const errorData = await response.text();
                throw new Error(`Local repository API error (${response.status}): ${errorData}`);
            }
            const data = await response.json();
            fileTreeData = data.file_tree;
            readmeContent = data.readme;
            setDefaultBranch('main'); // Local repos don't have a concept of a remote branch
        }
        // Add Bitbucket logic here if needed...

        setFileTree(fileTreeData);
        setReadme(readmeContent);
        setPageState('determining_wiki_structure');
      } catch (err) {
        handleError(err instanceof Error ? err.message : 'Unknown error fetching repository data.');
      }
    };

    fetchRepoData();
  }, [pageState, repoInfo, token, owner, repo, messages.loading]);


  // 2. Determining Wiki Structure
  useEffect(() => {
    if (pageState !== 'determining_wiki_structure') return;

    setLoadingMessage(messages.loading?.determiningStructure || 'Determining wiki structure...');

    const prompt = `
      Analyze the file tree and README of the repository below to create a logical wiki structure.
      
      File Tree:
      <file_tree>
      ${fileTree}
      </file_tree>

      README:
      <readme>
      ${readme}
      </readme>

      CRITICAL INSTRUCTIONS:
      1. Your response MUST be ONLY the XML structure. Do not include any other text, explanations, or markdown formatting before or after the XML.
      2. The XML must be well-formed and valid.
      3. The root element MUST be <wiki_structure>.
      4. Follow the specified XML schema precisely.

      <wiki_structure>
        <title>[Overall title for the wiki]</title>
        <description>[Brief description of the repository]</description>
        <sections>
          <section id="section-1">
            <title>[Section title]</title>
            <pages>
              <page_ref>page-1</page_ref>
            </pages>
          </section>
        </sections>
        <pages>
          <page id="page-1">
            <title>[Page title]</title>
            <description>[Brief description of what this page will cover]</description>
            <importance>high|medium|low</importance>
            <relevant_files>
              <file_path>[Path to a relevant file]</file_path>
            </relevant_files>
          </page>
        </pages>
      </wiki_structure>
    `;
    const requestBody: ChatCompletionRequest = {
      repo_url: getRepoUrl(repoInfo),
      type: repoInfo.type,
      messages: [{ role: 'user', content: prompt }],
      provider: selectedProvider,
      model: isCustomModel ? customModel : selectedModel,
      language: language,
      token: token,
    };

    let responseBuffer = '';
    closeWebSocket(webSocketRef.current);
    webSocketRef.current = createChatWebSocket(
      requestBody,
      (chunk) => { responseBuffer += chunk; },
      (status, msg) => {
        console.log(`Structure status: ${status} - ${msg}`);
        if (status === 'error') {
          handleError(`WebSocket error while determining structure: ${msg}`);
        }
      },
      (err) => handleError(`WebSocket error while determining structure: ${err}`),
      () => { // onComplete
        try {
          console.log(`Wiki structure response completed. Length: ${responseBuffer.length} characters`);
          
          // Validate response completeness
          if (!responseBuffer.trim()) {
            throw new Error("Empty response received from server.");
          }
          
          const xmlMatch = responseBuffer.match(/<wiki_structure>[\s\S]*?<\/wiki_structure>/m);
          if (!xmlMatch) {
            console.error("Backend response did not contain valid XML structure.", {
              response: responseBuffer.length > 1000 ? responseBuffer.substring(0, 1000) + '...[truncated]' : responseBuffer,
              responseLength: responseBuffer.length
            });
            throw new Error(
              "The AI failed to generate a valid wiki structure. This can happen with complex repositories. Please try refreshing."
            );
          }
          
          console.log(`Found XML structure: ${xmlMatch[0].length} characters`);
          
          // Additional validation for wiki_structure completeness
          const xmlContent = xmlMatch[0];
          if (!xmlContent.includes('</wiki_structure>')) {
            console.error("Wiki structure XML appears to be incomplete");
            throw new Error("Incomplete wiki structure received. Please try refreshing.");
          }
          
          const parser = new DOMParser();
          const xmlDoc = parser.parseFromString(xmlMatch[0], "text/xml");
          const parseError = xmlDoc.querySelector('parsererror');
          if (parseError) {
            console.error("Failed to parse wiki structure XML.", {
              xml: xmlMatch[0],
              error: parseError.textContent,
            });
            throw new Error(
              "The AI returned a malformed XML structure. Please try refreshing."
            );
          }

          const pages: WikiPage[] = Array.from(xmlDoc.querySelectorAll('page')).map(p => ({
            id: p.getAttribute('id') || '',
            title: p.querySelector('title')?.textContent || '',
            filePaths: Array.from(p.querySelectorAll('file_path')).map(f => f.textContent || ''),
            importance: (p.querySelector('importance')?.textContent as 'high' | 'medium' | 'low') || 'medium',
            relatedPages: Array.from(p.querySelectorAll('related')).map(r => r.textContent || ''),
            content: '',
          }));

          const sections: WikiSection[] = Array.from(xmlDoc.querySelectorAll('section')).map(s => ({
            id: s.getAttribute('id') || '',
            title: s.querySelector('title')?.textContent || '',
            pages: Array.from(s.querySelectorAll('page_ref')).map(p => p.textContent || ''),
            subsections: Array.from(s.querySelectorAll('section_ref')).map(sub => sub.textContent || ''),
          }));
          
          const structure: WikiStructure = {
            id: 'wiki',
            title: xmlDoc.querySelector('title')?.textContent || 'Wiki',
            description: xmlDoc.querySelector('description')?.textContent || '',
            pages,
            sections,
            rootSections: sections.filter(s => !sections.some(parent => parent.subsections?.includes(s.id))).map(s => s.id),
          };

          setWikiStructure(structure);
          setGeneratedPages({});
          setPagesInProgress(new Set(pages.map(p => p.id)));
          pageQueueRef.current = [...structure.pages];
          setPageState('generating_page_content');
        } catch (err) {
          handleError(err instanceof Error ? err.message : 'Failed to process wiki structure.');
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


  // 3. Generating Page Content
  useEffect(() => {
    if (pageState !== 'generating_page_content') return;

    const MAX_CONCURRENT = 3;

    const processQueue = () => {
      if (pageQueueRef.current.length === 0 && activeRequestsRef.current === 0) {
        setPageState('ready');
        setLoadingMessage(undefined);
        return;
      }

      while (pageQueueRef.current.length > 0 && activeRequestsRef.current < MAX_CONCURRENT) {
        const page = pageQueueRef.current.shift();
        if (!page) continue;

        activeRequestsRef.current++;
        setLoadingMessage(`Generating: ${page.title}...`);

        const prompt = `Generate a markdown wiki page for "${page.title}" using these files: ${page.filePaths.join(', ')}.`;
        const requestBody: ChatCompletionRequest = {
          repo_url: getRepoUrl(repoInfo),
          type: repoInfo.type,
          messages: [{ role: 'user', content: prompt }],
          provider: selectedProvider,
          model: isCustomModel ? customModel : selectedModel,
          language: language,
          token: token,
        };

        let responseBuffer = '';
        // Use a new variable for the socket to avoid closure issues
        const pageSocket = createChatWebSocket(
          requestBody,
          (chunk) => { responseBuffer += chunk; },
          (status, msg) => {
            console.log(`Page gen status for ${page.id}: ${status} - ${msg}`);
            if (status === 'error') {
              console.error(`Error generating page ${page.id}: ${msg}`);
              setGeneratedPages(prev => ({ ...prev, [page.id]: { ...page, content: `Error: ${msg}` } }));
              activeRequestsRef.current--;
              setPagesInProgress(prev => {
                const newSet = new Set(prev);
                newSet.delete(page.id);
                return newSet;
              });
              processQueue(); // Continue with next
            }
          },
          (err) => {
            console.error(`Error generating page ${page.id}: ${err}`);
            setGeneratedPages(prev => ({ ...prev, [page.id]: { ...page, content: `Error: ${err}` } }));
            activeRequestsRef.current--;
            setPagesInProgress(prev => {
              const newSet = new Set(prev);
              newSet.delete(page.id);
              return newSet;
            });
            processQueue(); // Continue with next
          },
          () => { // onComplete
            console.log(`Page generation completed for ${page.id}. Length: ${responseBuffer.length} characters`);
            
            // Validate response completeness
            if (!responseBuffer.trim()) {
              console.error(`Empty response received for page ${page.id}`);
              setGeneratedPages(prev => ({ ...prev, [page.id]: { ...page, content: `Error: Empty response received for page ${page.title}` } }));
            } else {
              setGeneratedPages(prev => ({ ...prev, [page.id]: { ...page, content: responseBuffer } }));
            }
            
            activeRequestsRef.current--;
            setPagesInProgress(prev => {
              const newSet = new Set(prev);
              newSet.delete(page.id);
              return newSet;
            });
            processQueue(); // Continue with next
          }
        );
      }
    };

    processQueue();

  }, [pageState, repoInfo, language, selectedProvider, selectedModel, isCustomModel, customModel, token]);

  // --- UI Event Handlers ---
  const handleRefresh = () => {
    setError(null);
    setWikiStructure(undefined);
    setGeneratedPages({});
    setPagesInProgress(new Set());
    pageQueueRef.current = [];
    activeRequestsRef.current = 0;
    setPageState('fetching_repo_structure');
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
    <div className="h-screen paper-texture p-4 md:p-8 flex flex-col">
      <style>{wikiStyles}</style>
      <header className="max-w-[90%] xl:max-w-[1400px] mx-auto mb-8 h-fit w-full">
        <div className="flex items-center justify-between">
            <Link href="/" className="text-[var(--accent-primary)] hover:text-[var(--highlight)] flex items-center gap-1.5 transition-colors">
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
                  onClick={() => setIsAskModalOpen(true)}
                  className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors flex items-center gap-2"
              >
                  <FaComments />
                  {messages.repoPage?.askButton || 'Ask about this repo'}
              </button>
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