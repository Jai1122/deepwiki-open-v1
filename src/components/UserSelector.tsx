'use client';

import React, { useState, useEffect } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';

// Define the interfaces for our model configuration
interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  models: Model[];
  supportsCustomModel?: boolean;
}

interface ModelConfig {
  providers: Provider[];
  defaultProvider: string;
}

interface ModelSelectorProps {
  provider: string;
  setProvider: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  isCustomModel: boolean;
  setIsCustomModel: (value: boolean) => void;
  customModel: string;
  setCustomModel: (value: string) => void;

  // File filter configuration (deprecated - no longer used)
  showFileFilters?: boolean;
  excludedDirs?: string;
  setExcludedDirs?: (value: string) => void;
  excludedFiles?: string;
  setExcludedFiles?: (value: string) => void;
  includedDirs?: string;
  setIncludedDirs?: (value: string) => void;
  includedFiles?: string;
  setIncludedFiles?: (value: string) => void;
}

export default function UserSelector({
  provider,
  setProvider,
  model,
  setModel,
  isCustomModel,
  setIsCustomModel,
  customModel,
  setCustomModel,
  
  // File filter options are deprecated but kept for backward compatibility
  showFileFilters = false,
}: ModelSelectorProps) {
  const { messages: t } = useLanguage();

  // State for model configurations from backend
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch model configurations from the backend
  useEffect(() => {
    const fetchModelConfig = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const response = await fetch('/api/models/config');

        if (!response.ok) {
          throw new Error(`Error fetching model configurations: ${response.status}`);
        }

        const data = await response.json();
        setModelConfig(data);

        // Initialize provider and model with defaults from API if not already set
        if (!provider && data.defaultProvider) {
          setProvider(data.defaultProvider);
        }

        // Initialize model with the default model for the provider
        if (!model || !data.providers.find((p: Provider) => p.id === provider)?.models.find((m: Model) => m.id === model)) {
          const selectedProvider = data.providers.find((p: Provider) => p.id === (provider || data.defaultProvider));
          if (selectedProvider && selectedProvider.models.length > 0) {
            const defaultModel = selectedProvider.models.find((m: Model) => m.id === selectedProvider.models[0].id);
            if (defaultModel) {
              setModel(defaultModel.id);
            }
          }
        }
      } catch (err) {
        console.error('Error fetching model configurations:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };

    fetchModelConfig();
  }, [provider, model, setProvider, setModel]);

  // Get available models for the selected provider
  const getModelsForProvider = (providerId: string): Model[] => {
    const selectedProvider = modelConfig?.providers.find(p => p.id === providerId);
    return selectedProvider?.models || [];
  };

  // Check if the selected provider supports custom models
  const getSupportsCustomModel = (providerId: string): boolean => {
    const selectedProvider = modelConfig?.providers.find(p => p.id === providerId);
    return selectedProvider?.supportsCustomModel || false;
  };

  // Handle provider change
  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    const models = getModelsForProvider(newProvider);
    if (models.length > 0) {
      setModel(models[0].id);
    }
    setIsCustomModel(false);
  };

  // Handle model change
  const handleModelChange = (newModel: string) => {
    setModel(newModel);
    setIsCustomModel(false);
  };

  // Handle custom model toggle
  const handleCustomModelToggle = (useCustom: boolean) => {
    setIsCustomModel(useCustom);
    if (useCustom) {
      setCustomModel(customModel || '');
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse bg-[var(--muted)]/20 h-20 rounded-md"></div>
        <div className="animate-pulse bg-[var(--muted)]/20 h-12 rounded-md"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-md">
        <p className="text-red-800 text-sm">Error loading model configurations: {error}</p>
        <button 
          onClick={() => window.location.reload()} 
          className="mt-2 text-red-600 hover:text-red-800 text-sm underline"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!modelConfig) {
    return (
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-md">
        <p className="text-yellow-800 text-sm">No model configurations available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="card-confluence p-4">
        {/* Provider Selection */}
        <div className="space-y-4">
          <div>
            <label htmlFor="provider-select" className="block text-sm font-medium text-[var(--foreground)] mb-2">
              {t.form?.provider || 'AI Provider'}
            </label>
            <select
              id="provider-select"
              value={provider}
              onChange={(e) => handleProviderChange(e.target.value)}
              className="select-confluence block w-full"
            >
              {modelConfig.providers.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* Model Selection */}
          <div>
            <label htmlFor="model-select" className="block text-sm font-medium text-[var(--foreground)] mb-2">
              {t.form?.model || 'Model'}
            </label>
            <div className="space-y-2">
              {/* Predefined Models */}
              <select
                id="model-select"
                value={isCustomModel ? '' : model}
                onChange={(e) => handleModelChange(e.target.value)}
                disabled={isCustomModel}
                className={`select-confluence block w-full ${
                  isCustomModel ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {getModelsForProvider(provider).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>

              {/* Custom Model Option */}
              {getSupportsCustomModel(provider) && (
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={isCustomModel}
                      onChange={(e) => handleCustomModelToggle(e.target.checked)}
                      className="mr-2 accent-[var(--accent-primary)] focus:ring-[var(--accent-primary)] rounded"
                    />
                    <span className="text-sm text-[var(--foreground)]">
                      {t.form?.useCustomModel || 'Use Custom Model'}
                    </span>
                  </label>
                  
                  {isCustomModel && (
                    <input
                      type="text"
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                      placeholder={t.form?.enterCustomModel || 'Enter custom model name...'}
                      className="input-confluence block w-full"
                    />
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Advanced file filtering options removed */}
        {showFileFilters && (
          <div className="mt-4 p-3 bg-[var(--hover-bg)] rounded border border-[var(--border-color)]">
            <p className="text-sm text-[var(--muted)] text-center">
              Advanced file filtering options have been removed. Default filtering is applied automatically.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}