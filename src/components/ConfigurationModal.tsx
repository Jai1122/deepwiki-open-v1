'use client';

import React, { useState } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import UserSelector from './UserSelector';
import TokenInput from './TokenInput';

interface ConfigurationModalProps {
  isOpen: boolean;
  onClose: () => void;

  // Repository input
  repositoryInput: string;


  // Wiki type options
  isComprehensiveView: boolean;
  setIsComprehensiveView: (value: boolean) => void;

  // Model selection
  provider: string;
  setProvider: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  isCustomModel: boolean;
  setIsCustomModel: (value: boolean) => void;
  customModel: string;
  setCustomModel: (value: string) => void;

  // Platform selection
  selectedPlatform: 'github' | 'gitlab' | 'bitbucket';
  setSelectedPlatform: (value: 'github' | 'gitlab' | 'bitbucket') => void;

  // Access token
  accessToken: string;
  setAccessToken: (value: string) => void;


  // Form submission
  onSubmit: () => void;
  isSubmitting: boolean;

  // Authentication
  authRequired?: boolean;
  authCode?: string;
  setAuthCode?: (code: string) => void;
  isAuthLoading?: boolean;
}

export default function ConfigurationModal({
  isOpen,
  onClose,
  repositoryInput,
  isComprehensiveView,
  setIsComprehensiveView,
  provider,
  setProvider,
  model,
  setModel,
  isCustomModel,
  setIsCustomModel,
  customModel,
  setCustomModel,
  selectedPlatform,
  setSelectedPlatform,
  accessToken,
  setAccessToken,
  onSubmit,
  isSubmitting,
  authRequired,
  authCode,
  setAuthCode,
  isAuthLoading
}: ConfigurationModalProps) {
  const { messages: t } = useLanguage();

  // Show token section state
  const [showTokenSection, setShowTokenSection] = useState(false);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4 text-center bg-black/50">
        <div className="modal-confluence relative transform overflow-hidden text-left transition-all sm:my-8 sm:max-w-2xl sm:w-full">
          {/* Modal header with close button */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-color)]">
            <h3 className="text-lg font-medium text-[var(--accent-primary)]">
              <span className="text-[var(--accent-primary)]">{t.form?.configureWiki || 'Configure Wiki'}</span>
            </h3>
            <button
              type="button"
              onClick={onClose}
              className="text-[var(--muted)] hover:text-[var(--foreground)] focus:outline-none transition-colors"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Modal body */}
          <div className="p-6 max-h-[70vh] overflow-y-auto">
            {/* Repository info */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-[var(--foreground)] mb-2">
                {t.form?.repository || 'Repository'}
              </label>
              <div className="card-confluence p-3 text-sm text-[var(--foreground)]">
                {repositoryInput}
              </div>
            </div>


            {/* Wiki Type Selector - more compact version */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-[var(--foreground)] mb-2">
                {t.form?.wikiType || 'Wiki Type'}
              </label>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setIsComprehensiveView(true)}
                  className={`flex-1 flex items-center justify-between p-3 rounded border-2 transition-colors ${
                    isComprehensiveView
                      ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                      : 'bg-[var(--card-bg)] border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--hover-bg)]'
                  }`}
                >
                  <div className="flex items-center">
                    <div className="text-left">
                      <div className="font-medium text-sm">{t.form?.comprehensive || 'Comprehensive'}</div>
                      <div className="text-xs opacity-80">
                        {t.form?.comprehensiveDescription || 'Detailed wiki with structured sections'}
                      </div>
                    </div>
                  </div>
                  {isComprehensiveView && (
                    <div className="ml-2 h-4 w-4 rounded-full bg-[var(--accent-primary)]/20 flex items-center justify-center">
                      <div className="h-2 w-2 rounded-full bg-[var(--accent-primary)]"></div>
                    </div>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => setIsComprehensiveView(false)}
                  className={`flex-1 flex items-center justify-between p-3 rounded border-2 transition-colors ${
                    !isComprehensiveView
                      ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                      : 'bg-[var(--card-bg)] border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--hover-bg)]'
                  }`}
                >
                  <div className="flex items-center">
                    <div className="text-left">
                      <div className="font-medium text-sm">{t.form?.concise || 'Concise'}</div>
                      <div className="text-xs opacity-80">
                        {t.form?.conciseDescription || 'Simplified wiki with fewer pages'}
                      </div>
                    </div>
                  </div>
                  {!isComprehensiveView && (
                    <div className="ml-2 h-4 w-4 rounded-full bg-[var(--accent-primary)]/20 flex items-center justify-center">
                      <div className="h-2 w-2 rounded-full bg-[var(--accent-primary)]"></div>
                    </div>
                  )}
                </button>
              </div>
            </div>

            {/* Model Selector */}
            <div className="mb-4">
              <UserSelector
                provider={provider}
                setProvider={setProvider}
                model={model}
                setModel={setModel}
                isCustomModel={isCustomModel}
                setIsCustomModel={setIsCustomModel}
                customModel={customModel}
                setCustomModel={setCustomModel}
                showFileFilters={false}
              />
            </div>

            {/* Access token section using TokenInput component */}
            <TokenInput
              selectedPlatform={selectedPlatform}
              setSelectedPlatform={setSelectedPlatform}
              accessToken={accessToken}
              setAccessToken={setAccessToken}
              showTokenSection={showTokenSection}
              onToggleTokenSection={() => setShowTokenSection(!showTokenSection)}
              allowPlatformChange={true}
            />

            {/* Authorization Code Input */}
            {isAuthLoading && (
              <div className="mb-4 p-3 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)] text-sm text-[var(--muted)]">
                Loading authentication status...
              </div>
            )}
            {!isAuthLoading && authRequired && (
              <div className="mb-4 p-4 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
                <label htmlFor="authCode" className="block text-sm font-medium text-[var(--foreground)] mb-2">
                  {t.form?.authorizationCode || 'Authorization Code'}
                </label>
                <input
                  type="password"
                  id="authCode"
                  value={authCode || ''}
                  onChange={(e) => setAuthCode?.(e.target.value)}
                  className="input-japanese block w-full px-3 py-2 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
                  placeholder="Enter your authorization code"
                />
                 <div className="flex items-center mt-2 text-xs text-[var(--muted)]">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-[var(--muted)]"
                    fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                   {t.form?.authorizationRequired || 'Authentication is required to generate the wiki.'}
                </div>
              </div>
            )}
          </div>

          {/* Modal footer */}
          <div className="flex items-center justify-end gap-2 px-6 py-4 border-t border-[var(--border-color)]">
            <button
              type="button"
              onClick={onClose}
              className="btn-confluence-secondary"
            >
              {t.common?.cancel || 'Cancel'}
            </button>
            <button
              type="button"
              onClick={onSubmit}
              disabled={isSubmitting}
              className="btn-confluence disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (t.common?.processing || 'Processing...') : (t.common?.generateWiki || 'Generate Wiki')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
