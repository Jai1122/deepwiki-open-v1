'use client';

import React, {useState, useRef, useEffect, useCallback} from 'react';
import {FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import Markdown from './Markdown';
import { useLanguage } from '@/contexts/LanguageContext';
import RepoInfo from '@/types/repoinfo';
import getRepoUrl from '@/utils/getRepoUrl';
import ModelSelectionModal from './ModelSelectionModal';
import { createChatWebSocket, closeWebSocket, ChatCompletionRequest } from '@/utils/websocketClient';

// Interfaces remain the same
interface Model { id: string; name: string; }
interface Provider { id: string; name: string; models: Model[]; supportsCustomModel?: boolean; }
interface Message { role: 'user' | 'assistant' | 'system'; content: string; }
interface ResearchStage { title: string; content: string; iteration: number; type: 'plan' | 'update' | 'conclusion'; }
interface AskProps {
  repoInfo: RepoInfo;
  provider?: string;
  model?: string;
  isCustomModel?: boolean;
  customModel?: string;
  language?: string;
  onRef?: (ref: { clearConversation: () => void }) => void;
}

const Ask: React.FC<AskProps> = ({
  repoInfo,
  provider = '',
  model = '',
  isCustomModel = false,
  customModel = '',
  language = 'en',
  onRef
}) => {
  // --- Core State ---
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [researchIteration, setResearchIteration] = useState(0);
  const [researchComplete, setResearchComplete] = useState(false);
  
  // --- UI & Modal State ---
  const [selectedProvider, setSelectedProvider] = useState(provider);
  const [selectedModel, setSelectedModel] = useState(model);
  const [isCustomSelectedModel, setIsCustomSelectedModel] = useState(isCustomModel);
  const [customSelectedModel, setCustomSelectedModel] = useState(customModel);
  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);
  const [isComprehensiveView, setIsComprehensiveView] = useState(true);
  const [researchStages, setResearchStages] = useState<ResearchStage[]>([]);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);

  // --- Refs for UI and stable callbacks ---
  const inputRef = useRef<HTMLInputElement>(null);
  const responseRef = useRef<HTMLDivElement>(null);
  const webSocketRef = useRef<WebSocket | null>(null);
  const { messages } = useLanguage();

  // This ref is the key to solving the stale state issue.
  // It holds a reference to the latest state and props needed by the WebSocket callbacks.
  const stateRef = useRef({
    conversationHistory,
    researchIteration,
    deepResearch,
    repoInfo,
    selectedProvider,
    selectedModel,
    isCustomSelectedModel,
    customSelectedModel,
    language,
  });

  // Keep the state ref updated on every render.
  useEffect(() => {
    stateRef.current = {
      conversationHistory,
      researchIteration,
      deepResearch,
      repoInfo,
      selectedProvider,
      selectedModel,
      isCustomSelectedModel,
      customSelectedModel,
      language,
    };
  });

  // --- Lifecycle & UI Effects ---
  useEffect(() => {
    inputRef.current?.focus();
    return () => {
      closeWebSocket(webSocketRef.current);
    };
  }, []);

  useEffect(() => {
    if (onRef) onRef({ clearConversation });
  }, [onRef]);

  useEffect(() => {
    if (responseRef.current) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [response]);

  // --- Core Logic ---

  const clearConversation = () => {
    setQuestion('');
    setResponse('');
    setConversationHistory([]);
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setCurrentStageIndex(0);
    inputRef.current?.focus();
  };

  // This is the new, robust function for handling all requests.
  const startRequest = useCallback((history: Message[], iteration: number) => {
    setIsLoading(true);
    setResearchIteration(iteration);
    
    // For deep research, we clear the response for the new iteration.
    if (iteration > 1) {
      setResponse('');
    }

    const currentState = stateRef.current;
    const requestBody: ChatCompletionRequest = {
      repo_url: getRepoUrl(currentState.repoInfo),
      type: currentState.repoInfo.type,
      messages: history,
      provider: currentState.selectedProvider,
      model: currentState.isCustomSelectedModel ? currentState.customSelectedModel : currentState.selectedModel,
      language: currentState.language,
      token: currentState.repoInfo.token,
    };

    let responseBuffer = '';
    closeWebSocket(webSocketRef.current);

    webSocketRef.current = createChatWebSocket(
      requestBody,
      // onContent
      (contentChunk) => {
        responseBuffer += contentChunk;
        setResponse(responseBuffer);
      },
      // onStatus
      (status, message) => console.log(`WebSocket status: ${status} - ${message}`),
      // onError
      (error) => {
        console.error('WebSocket error:', error);
        setResponse(prev => prev + '\n\nError: Connection failed.');
        setIsLoading(false);
      },
      // onComplete - This is where the magic happens
      () => {
        const { deepResearch, conversationHistory: latestHistory } = stateRef.current;
        const isCompleteByMaxIterations = iteration >= 5;

        // Check if we should continue the deep research loop.
        if (deepResearch && !isCompleteByMaxIterations) {
          // Use a timeout to let the user read the last response
          setTimeout(() => {
            const newHistory: Message[] = [
              ...latestHistory,
              { role: 'assistant', content: responseBuffer },
              { role: 'user', content: '[DEEP RESEARCH] Continue the research' }
            ];
            setConversationHistory(newHistory);
            startRequest(newHistory, iteration + 1);
          }, 2000);
        } else {
          // If we are not continuing, the process is finished.
          if (deepResearch && isCompleteByMaxIterations) {
            const completionNote = "\n\n## Final Conclusion\nAfter multiple iterations, the deep research process has concluded. The findings presented across all iterations form the comprehensive answer.";
            setResponse(prev => prev + completionNote);
          }
          setResearchComplete(true);
          setIsLoading(false); // This is the single, reliable point where loading is stopped.
        }
      }
    );
  }, []); // This function is stable and does not re-create on every render.

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    // Reset all state for a new request.
    setResponse('');
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setCurrentStageIndex(0);

    const initialHistory: Message[] = [{
      role: 'user',
      content: deepResearch ? `[DEEP RESEARCH] ${question}` : question
    }];
    setConversationHistory(initialHistory);
    
    // Kick off the first request.
    startRequest(initialHistory, 1);
  };

  // --- The rest of the component is for rendering the UI ---
  // (This part is largely unchanged)
  const [buttonWidth, setButtonWidth] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (buttonRef.current) {
      const width = buttonRef.current.offsetWidth;
      setButtonWidth(width);
    }
  }, [messages.ask?.askButton, isLoading]);

  return (
    <div>
      <div className="p-4">
        <div className="flex items-center justify-end mb-4">
          <button
            type="button"
            onClick={() => setIsModelSelectionModalOpen(true)}
            className="text-xs px-2.5 py-1 rounded border border-[var(--border-color)]/40 bg-[var(--background)]/10 text-[var(--foreground)]/80 hover:bg-[var(--background)]/30 hover:text-[var(--foreground)] transition-colors flex items-center gap-1.5"
          >
            <span>{selectedProvider}/{isCustomSelectedModel ? customSelectedModel : selectedModel}</span>
            <svg className="h-3.5 w-3.5 text-[var(--accent-primary)]/70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="mt-4">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={messages.ask?.placeholder || 'What would you like to know about this codebase?'}
              className="block w-full rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] text-[var(--foreground)] px-5 py-3.5 text-base shadow-sm focus:border-[var(--accent-primary)] focus:ring-2 focus:ring-[var(--accent-primary)]/30 focus:outline-none transition-all"
              style={{ paddingRight: `${buttonWidth + 24}px` }}
              disabled={isLoading}
            />
            <button
              ref={buttonRef}
              type="submit"
              disabled={isLoading || !question.trim()}
              className={`absolute right-3 top-1/2 transform -translate-y-1/2 px-4 py-2 rounded-md font-medium text-sm ${
                isLoading || !question.trim()
                  ? 'bg-[var(--button-disabled-bg)] text-[var(--button-disabled-text)] cursor-not-allowed'
                  : 'bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary)]/90 shadow-sm'
              } transition-all duration-200 flex items-center gap-1.5`}
            >
              {isLoading ? (
                <div className="w-4 h-4 rounded-full border-2 border-t-transparent border-white animate-spin" />
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                  <span>{messages.ask?.askButton || 'Ask'}</span>
                </>
              )}
            </button>
          </div>

          <div className="flex items-center mt-2 justify-between">
            <div className="group relative">
              <label className="flex items-center cursor-pointer">
                <span className="text-xs text-gray-600 dark:text-gray-400 mr-2">Deep Research</span>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={deepResearch}
                    onChange={() => setDeepResearch(!deepResearch)}
                    className="sr-only"
                  />
                  <div className={`w-10 h-5 rounded-full transition-colors ${deepResearch ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}`}></div>
                  <div className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform transform ${deepResearch ? 'translate-x-5' : ''}`}></div>
                </div>
              </label>
              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs rounded p-2 w-72 z-10">
                <div className="relative">
                  <div className="absolute -bottom-2 left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
                  <p className="mb-1">Deep Research conducts a multi-turn investigation process:</p>
                  <ul className="list-disc pl-4 text-xs">
                    <li><strong>Initial Research:</strong> Creates a research plan and initial findings</li>
                    <li><strong>Iteration 1-4:</strong> Explores specific aspects in depth</li>
                    <li><strong>Final Conclusion:</strong> Comprehensive answer based on all iterations</li>
                  </ul>
                </div>
              </div>
            </div>
            {deepResearch && (
              <div className="text-xs text-purple-600 dark:text-purple-400">
                Multi-turn research process enabled
                {researchIteration > 0 && !researchComplete && ` (iteration ${researchIteration})`}
                {researchComplete && ` (complete)`}
              </div>
            )}
          </div>
        </form>

        {response && (
          <div className="border-t border-gray-200 dark:border-gray-700 mt-4">
            <div
              ref={responseRef}
              className="p-4 max-h-[500px] overflow-y-auto"
            >
              <Markdown content={response} />
            </div>
            <div className="p-2 flex justify-end items-center border-t border-gray-200 dark:border-gray-700">
              <button
                id="ask-clear-conversation"
                onClick={clearConversation}
                className="text-xs text-gray-500 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 px-2 py-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                Clear conversation
              </button>
            </div>
          </div>
        )}

        {isLoading && !response && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <div className="animate-pulse flex space-x-1">
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {deepResearch
                  ? `Research iteration ${researchIteration} in progress...`
                  : "Thinking..."}
              </span>
            </div>
          </div>
        )}
      </div>

      <ModelSelectionModal
        isOpen={isModelSelectionModalOpen}
        onClose={() => setIsModelSelectionModalOpen(false)}
        provider={selectedProvider}
        setProvider={setSelectedProvider}
        model={selectedModel}
        setModel={setSelectedModel}
        isCustomModel={isCustomSelectedModel}
        setIsCustomModel={setIsCustomSelectedModel}
        customModel={customSelectedModel}
        setCustomModel={setCustomSelectedModel}
        isComprehensiveView={isComprehensiveView}
        setIsComprehensiveView={setIsComprehensiveView}
        showFileFilters={false}
        onApply={() => {}}
        showWikiType={false}
        authRequired={false}
        isAuthLoading={false}
      />
    </div>
  );
};

export default Ask;
