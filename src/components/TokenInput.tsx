'use client';

import React from 'react';
import { useLanguage } from '@/contexts/LanguageContext';

interface TokenInputProps {
  selectedPlatform: 'bitbucket';
  setSelectedPlatform: (value: 'bitbucket') => void;
  accessToken: string;
  setAccessToken: (value: string) => void;
  showTokenSection?: boolean;
  onToggleTokenSection?: () => void;
  allowPlatformChange?: boolean;
}

export default function TokenInput({
  selectedPlatform,
  setSelectedPlatform,
  accessToken,
  setAccessToken,
  showTokenSection = true,
  onToggleTokenSection,
  allowPlatformChange = true
}: TokenInputProps) {
  const { messages: t } = useLanguage();

  const platformName = selectedPlatform.charAt(0).toUpperCase() + selectedPlatform.slice(1);

  return (
    <div className="mb-4">
      {onToggleTokenSection && (
        <button
          type="button"
          onClick={onToggleTokenSection}
          className="text-sm text-[var(--accent-primary)] hover:text-[var(--highlight)] flex items-center transition-colors border-b border-[var(--border-color)] hover:border-[var(--accent-primary)] pb-0.5 mb-2"
        >
          {showTokenSection ? t.form?.hideTokens || 'Hide Access Tokens' : t.form?.addTokens || 'Add Access Tokens for Private Repositories'}
        </button>
      )}

      {showTokenSection && (
        <div className="mt-2 p-4 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
          {allowPlatformChange && (
            <div className="mb-3">
              <label className="block text-xs font-medium text-[var(--foreground)] mb-2">
                Platform: Bitbucket
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded border-2 bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]"
                  disabled
                >
                  <span className="text-sm">Bitbucket</span>
                </button>
              </div>
            </div>
          )}

          <div>
            <label htmlFor="access-token" className="block text-xs font-medium text-[var(--foreground)] mb-2">
              App Password
            </label>
            <input
              id="access-token"
              type="password"
              value={accessToken}
              onChange={(e) => setAccessToken(e.target.value)}
              placeholder="Enter your Bitbucket App Password"
              className="input-confluence block w-full"
            />
            <div className="flex items-center mt-2 text-xs text-[var(--muted)]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-[var(--muted)]"
                fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {t.form?.tokenSecurityNote || 'Your token is stored locally and never sent to our servers.'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 