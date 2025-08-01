'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { FaGithub, FaCoffee, FaTwitter } from 'react-icons/fa';
import ThemeToggle from '@/components/theme-toggle';
import ConfigurationModal from '@/components/ConfigurationModal';
import { extractUrlPath, extractUrlDomain } from '@/utils/urlDecoder';

import { useLanguage } from '@/contexts/LanguageContext';

export default function Home() {
  const router = useRouter();
  const { messages } = useLanguage();

  // Create a simple translation function
  const t = (key: string, params: Record<string, string | number> = {}): string => {
    // Split the key by dots to access nested properties
    const keys = key.split('.');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let value: any = messages;

    // Navigate through the nested properties
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        // Return the key if the translation is not found
        return key;
      }
    }

    // If the value is a string, replace parameters
    if (typeof value === 'string') {
      return Object.entries(params).reduce((acc: string, [paramKey, paramValue]) => {
        return acc.replace(`{${paramKey}}`, String(paramValue));
      }, value);
    }

    // Return the key if the value is not a string
    return key;
  };

  const [repositoryInput, setRepositoryInput] = useState('https://github.com/AsyncFuncAI/deepwiki-open');

  const REPO_CONFIG_CACHE_KEY = 'deepwikiRepoConfigCache';

  const loadConfigFromCache = (repoUrl: string) => {
    if (!repoUrl) return;
    try {
      const cachedConfigs = localStorage.getItem(REPO_CONFIG_CACHE_KEY);
      if (cachedConfigs) {
        const configs = JSON.parse(cachedConfigs);
        const config = configs[repoUrl.trim()];
        if (config) {
          setIsComprehensiveView(config.isComprehensiveView === undefined ? true : config.isComprehensiveView);
          setProvider(config.provider || '');
          setModel(config.model || '');
          setIsCustomModel(config.isCustomModel || false);
          setCustomModel(config.customModel || '');
          setSelectedPlatform(config.selectedPlatform || 'github');
        }
      }
    } catch (error) {
      console.error('Error loading config from localStorage:', error);
    }
  };

  const handleRepositoryInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newRepoUrl = e.target.value;
    setRepositoryInput(newRepoUrl);
    if (newRepoUrl.trim() === "") {
      // Optionally reset fields if input is cleared
    } else {
        loadConfigFromCache(newRepoUrl);
    }
  };

  useEffect(() => {
    if (repositoryInput) {
      loadConfigFromCache(repositoryInput);
    }
  }, []);

  // Provider-based model selection state
  const [provider, setProvider] = useState<string>('');
  const [model, setModel] = useState<string>('');
  const [isCustomModel, setIsCustomModel] = useState<boolean>(false);
  const [customModel, setCustomModel] = useState<string>('');

  // Wiki type state - default to comprehensive view
  const [isComprehensiveView, setIsComprehensiveView] = useState<boolean>(true);

  const [selectedPlatform, setSelectedPlatform] = useState<'github' | 'gitlab' | 'bitbucket'>('github');
  const [accessToken, setAccessToken] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Authentication state
  const [authRequired, setAuthRequired] = useState<boolean>(false);
  const [authCode, setAuthCode] = useState<string>('');
  const [isAuthLoading, setIsAuthLoading] = useState<boolean>(true);


  // Fetch authentication status on component mount
  useEffect(() => {
    const fetchAuthStatus = async () => {
      try {
        setIsAuthLoading(true);
        const response = await fetch('/api/auth/status');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAuthRequired(data.auth_required);
      } catch (err) {
        console.error("Failed to fetch auth status:", err);
        // Assuming auth is required if fetch fails to avoid blocking UI for safety
        setAuthRequired(true);
      } finally {
        setIsAuthLoading(false);
      }
    };

    fetchAuthStatus();
  }, []);

  // Parse repository URL/input and extract owner and repo
  const parseRepositoryInput = (input: string): {
    owner: string,
    repo: string,
    type: string,
    fullPath?: string,
    localPath?: string
  } | null => {
    input = input.trim();

    let owner = '', repo = '', type = 'github', fullPath;
    let localPath: string | undefined;

    // Handle Windows absolute paths (e.g., C:\path\to\folder)
    const windowsPathRegex = /^[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$/;
    const customGitRegex = /^(?:https?:\/\/)?([^\/]+)\/(.+?)\/([^\/]+)(?:\.git)?\/?$/;

    if (windowsPathRegex.test(input)) {
      type = 'local';
      localPath = input;
      repo = input.split('\\').pop() || 'local-repo';
      owner = 'local';
    }
    // Handle Unix/Linux absolute paths (e.g., /path/to/folder)
    else if (input.startsWith('/')) {
      type = 'local';
      localPath = input;
      repo = input.split('/').filter(Boolean).pop() || 'local-repo';
      owner = 'local';
    }
    else if (customGitRegex.test(input)) {
      // Detect repository type based on domain
      const domain = extractUrlDomain(input);
      if (domain?.includes('github.com')) {
        type = 'github';
      } else if (domain?.includes('gitlab.com') || domain?.includes('gitlab.')) {
        type = 'gitlab';
      } else if (domain?.includes('bitbucket.org') || domain?.includes('bitbucket.')) {
        type = 'bitbucket';
      } else {
        type = 'web'; // fallback for other git hosting services
      }

      fullPath = extractUrlPath(input)?.replace(/\.git$/, '');
      const parts = fullPath?.split('/') ?? [];
      if (parts.length >= 2) {
        repo = parts[parts.length - 1] || '';
        owner = parts[parts.length - 2] || '';
      }
    }
    // Unsupported URL formats
    else {
      console.error('Unsupported URL format:', input);
      return null;
    }

    if (!owner || !repo) {
      return null;
    }

    // Clean values
    owner = owner.trim();
    repo = repo.trim();

    // Remove .git suffix if present
    if (repo.endsWith('.git')) {
      repo = repo.slice(0, -4);
    }

    return { owner, repo, type, fullPath, localPath };
  };

  // State for configuration modal
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Parse repository input to validate
    const parsedRepo = parseRepositoryInput(repositoryInput);

    if (!parsedRepo) {
      setError('Invalid repository format. Use "owner/repo", GitHub/GitLab/BitBucket URL, or a local folder path like "/path/to/folder" or "C:\\path\\to\\folder".');
      return;
    }

    // If valid, open the configuration modal
    setError(null);
    setIsConfigModalOpen(true);
  };

  const validateAuthCode = async () => {
    try {
      if(authRequired) {
        if(!authCode) {
          return false;
        }
        const response = await fetch('/api/auth/validate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({'code': authCode})
        });
        if (!response.ok) {
          return false;
        }
        const data = await response.json();
        return data.success || false;
      }
    } catch {
      return false;
    }
    return true;
  };

  const handleGenerateWiki = async () => {

    // Check authorization code
    const validation = await validateAuthCode();
    if(!validation) {
      setError(`Failed to validate the authorization code`);
      console.error(`Failed to validate the authorization code`);
      setIsConfigModalOpen(false);
      return;
    }

    // Prevent multiple submissions
    if (isSubmitting) {
      console.log('Form submission already in progress, ignoring duplicate click');
      return;
    }

    try {
      const currentRepoUrl = repositoryInput.trim();
      if (currentRepoUrl) {
        const existingConfigs = JSON.parse(localStorage.getItem(REPO_CONFIG_CACHE_KEY) || '{}');
        const configToSave = {
          isComprehensiveView,
          provider,
          model,
          isCustomModel,
          customModel,
          selectedPlatform,
        };
        existingConfigs[currentRepoUrl] = configToSave;
        localStorage.setItem(REPO_CONFIG_CACHE_KEY, JSON.stringify(existingConfigs));
      }
    } catch (error) {
      console.error('Error saving config to localStorage:', error);
    }

    setIsSubmitting(true);

    // Parse repository input
    const parsedRepo = parseRepositoryInput(repositoryInput);

    if (!parsedRepo) {
      setError('Invalid repository format. Use "owner/repo", GitHub/GitLab/BitBucket URL, or a local folder path like "/path/to/folder" or "C:\\path\\to\\folder".');
      setIsSubmitting(false);
      return;
    }

    const { owner, repo, type, localPath } = parsedRepo;

    // Store tokens in query params if they exist
    const params = new URLSearchParams();
    if (accessToken) {
      params.append('token', accessToken);
    }
    // Always include the type parameter
    params.append('type', (type == 'local' ? type : selectedPlatform) || 'github');
    // Add local path if it exists
    if (localPath) {
      params.append('local_path', encodeURIComponent(localPath));
    } else {
      params.append('repo_url', encodeURIComponent(repositoryInput));
    }
    // Add model parameters
    params.append('provider', provider);
    params.append('model', model);
    if (isCustomModel && customModel) {
      params.append('custom_model', customModel);
    }

    // Add language parameter (English only)
    params.append('language', 'en');

    // Add comprehensive parameter
    params.append('comprehensive', isComprehensiveView.toString());

    const queryString = params.toString() ? `?${params.toString()}` : '';

    // Navigate to the dynamic route
    router.push(`/${owner}/${repo}${queryString}`);

    // The isSubmitting state will be reset when the component unmounts during navigation
  };

  return (
    <div className="h-screen bg-[var(--background)] flex flex-col">
      {/* Top navigation bar */}
      <nav className="flex justify-between items-center p-4">
        <Link href="/wiki/projects"
          className="text-sm text-[var(--muted)] hover:text-[var(--accent-primary)] transition-colors hover:underline">
          {t('nav.wikiProjects')}
        </Link>
        <ThemeToggle />
      </nav>

      {/* Main content centered like Google */}
      <div className="flex-1 flex flex-col justify-center items-center px-4">
        {/* DeepWiki Logo */}
        <div className="text-center mb-8">
          <h1 className="text-6xl md:text-7xl font-light text-[var(--foreground)] mb-2">
            <span className="text-[var(--accent-primary)]">D</span>
            <span className="text-[var(--accent-secondary)]">e</span>
            <span className="text-[var(--highlight)]">e</span>
            <span className="text-[var(--accent-primary)]">p</span>
            <span className="text-[var(--accent-secondary)]">W</span>
            <span className="text-[var(--accent-primary)]">i</span>
            <span className="text-[var(--highlight)]">k</span>
            <span className="text-[var(--accent-primary)]">i</span>
          </h1>
          <p className="text-sm text-[var(--muted)]">{t('common.tagline')}</p>
        </div>

        {/* Search bar */}
        <div className="w-full max-w-xl mb-8">
          <form onSubmit={handleFormSubmit}>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <FaGithub className="h-5 w-5 text-[var(--muted)]" />
              </div>
              <input
                type="text"
                value={repositoryInput}
                onChange={handleRepositoryInputChange}
                placeholder="Enter repository URL or owner/repo..."
                className="input-confluence block w-full pl-12 pr-16 py-4 text-lg rounded-full shadow-lg hover:shadow-xl focus:shadow-xl transition-all duration-200"
              />
              <div className="absolute inset-y-0 right-0 flex items-center">
                <button
                  type="submit"
                  className="mr-4 p-2 text-[var(--muted)] hover:text-[var(--accent-primary)] transition-colors"
                  disabled={isSubmitting}
                  title="Generate Wiki"
                >
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </button>
              </div>
            </div>
            {error && (
              <div className="text-[var(--error)] text-sm mt-2 text-center">
                {error}
              </div>
            )}
          </form>
        </div>

        {/* Confluence-style buttons */}
        <div className="flex gap-3 mb-8">
          <button
            onClick={handleFormSubmit}
            disabled={isSubmitting}
            className="btn-confluence"
          >
            {isSubmitting ? 'Processing...' : 'Generate Wiki'}
          </button>
          <Link 
            href="/wiki/projects"
            className="btn-confluence-secondary no-underline"
          >
            Browse Projects
          </Link>
        </div>

        {/* Quick examples */}
        <div className="text-center max-w-2xl">
          <p className="text-sm text-[var(--muted)] mb-3">Try these examples:</p>
          <div className="flex flex-wrap justify-center gap-2 text-xs">
            <button 
              onClick={() => setRepositoryInput('AsyncFuncAI/deepwiki-open')}
              className="px-3 py-1 text-[var(--accent-primary)] hover:underline cursor-pointer transition-colors"
            >
              AsyncFuncAI/deepwiki-open
            </button>
            <button 
              onClick={() => setRepositoryInput('https://github.com/vercel/next.js')}
              className="px-3 py-1 text-[var(--accent-primary)] hover:underline cursor-pointer transition-colors"
            >
              vercel/next.js
            </button>
            <button 
              onClick={() => setRepositoryInput('facebook/react')}
              className="px-3 py-1 text-[var(--accent-primary)] hover:underline cursor-pointer transition-colors"
            >
              facebook/react
            </button>
          </div>
        </div>
      </div>

      {/* Configuration Modal */}
      <ConfigurationModal
        isOpen={isConfigModalOpen}
        onClose={() => setIsConfigModalOpen(false)}
        repositoryInput={repositoryInput}
        isComprehensiveView={isComprehensiveView}
        setIsComprehensiveView={setIsComprehensiveView}
        provider={provider}
        setProvider={setProvider}
        model={model}
        setModel={setModel}
        isCustomModel={isCustomModel}
        setIsCustomModel={setIsCustomModel}
        customModel={customModel}
        setCustomModel={setCustomModel}
        selectedPlatform={selectedPlatform}
        setSelectedPlatform={setSelectedPlatform}
        accessToken={accessToken}
        setAccessToken={setAccessToken}
        onSubmit={handleGenerateWiki}
        isSubmitting={isSubmitting}
        authRequired={authRequired}
        authCode={authCode}
        setAuthCode={setAuthCode}
        isAuthLoading={isAuthLoading}
      />

      {/* Footer */}
      <footer className="p-4 text-center">
        <div className="flex flex-col sm:flex-row justify-center items-center gap-4 text-sm text-[var(--muted)]">
          <div className="flex items-center space-x-4">
            <a href="https://github.com/AsyncFuncAI/deepwiki-open" target="_blank" rel="noopener noreferrer"
              className="hover:text-[var(--accent-primary)] transition-colors">
              <FaGithub className="text-lg" />
            </a>
            <a href="https://buymeacoffee.com/sheing" target="_blank" rel="noopener noreferrer"
              className="hover:text-[var(--accent-primary)] transition-colors">
              <FaCoffee className="text-lg" />
            </a>
            <a href="https://x.com/sashimikun_void" target="_blank" rel="noopener noreferrer"
              className="hover:text-[var(--accent-primary)] transition-colors">
              <FaTwitter className="text-lg" />
            </a>
          </div>
          <p>{t('footer.copyright')}</p>
        </div>
      </footer>
    </div>
  );
}