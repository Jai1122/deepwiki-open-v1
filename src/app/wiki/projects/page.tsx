'use client';

import React from 'react';
import Link from 'next/link';
import ProcessedProjects from '@/components/ProcessedProjects';
import ThemeToggle from '@/components/theme-toggle';
import { useLanguage } from '@/contexts/LanguageContext';
import { FaHome } from 'react-icons/fa';

export default function WikiProjectsPage() {
  const { messages } = useLanguage();

  return (
    <div className="h-screen bg-[var(--background)] flex flex-col">
      {/* Top navigation bar - Google style */}
      <nav className="flex justify-between items-center p-4">
        <Link href="/"
          className="text-sm text-[var(--muted)] hover:text-[var(--accent-primary)] transition-colors hover:underline flex items-center gap-2">
          <FaHome className="h-4 w-4" />
          Home
        </Link>
        <ThemeToggle />
      </nav>

      {/* Main content */}
      <div className="flex-1 px-4 pb-4">
        <ProcessedProjects
          showHeader={true}
          messages={messages}
          className=""
        />
      </div>
    </div>
  );
}