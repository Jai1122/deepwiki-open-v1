'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Define the context type
interface LanguageContextType {
  language: string;
  setLanguage: (lang: string) => void;
  messages: any;
  supportedLanguages: Record<string, string>;
}

// Create the context
const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider = ({ children }: { children: ReactNode }) => {
  // Fixed to English only
  const [language] = useState<string>('en');
  const [messages, setMessages] = useState<any>({});
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [supportedLanguages] = useState<Record<string, string>>({ "en": "English" });

  useEffect(() => {
    // Load English messages only
    const initLanguage = async () => {
      try {
        const enMessages = await import('@/messages/en.json');
        setMessages(enMessages);
      } catch (err) {
        console.error('Error loading English messages:', err);
        setMessages({});
      } finally {
        setIsLoading(false);
      }
    };

    initLanguage();
  }, []);

  // No-op setLanguage function for compatibility
  const setLanguage = () => {
    // English only - no language switching
  };

  // Show loading state while initializing
  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage, messages, supportedLanguages }}>
      {children}
    </LanguageContext.Provider>
  );
};

// Hook to use the LanguageContext
export const useLanguage = (): LanguageContextType => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};