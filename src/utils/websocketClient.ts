/**
 * WebSocket client for chat completions
 * This replaces the HTTP streaming endpoint with a WebSocket connection
 */

// Get the server base URL from environment or use default
const SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

// Convert HTTP URL to WebSocket URL
const getWebSocketUrl = () => {
  const baseUrl = SERVER_BASE_URL;
  // Replace http:// with ws:// or https:// with wss://
  const wsBaseUrl = baseUrl.replace(/^http/, 'ws');
  return `${wsBaseUrl}/ws/chat`;
};

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatCompletionRequest {
  repo_url: string;
  messages: ChatMessage[];
  filePath?: string;
  token?: string;
  type?: string;
  provider?: string;
  model?: string;
  language?: string;
  excluded_dirs?: string;
  excluded_files?: string;
}

/**
 * Creates a WebSocket connection for chat completions.
 * This version parses server messages and emits typed events.
 *
 * @param request The chat completion request
 * @param onContent Callback for content chunks
 * @param onStatus Callback for status updates
 * @param onError Callback for errors
 * @param onComplete Callback for when the connection closes cleanly
 * @returns The WebSocket connection
 */
export const createChatWebSocket = (
  request: ChatCompletionRequest,
  onContent: (content: string) => void,
  onStatus: (status: string, message?: string) => void,
  onError: (error: Event) => void,
  onComplete: () => void
): WebSocket => {
  const ws = new WebSocket(getWebSocketUrl());

  ws.onopen = () => {
    console.log('WebSocket connection established');
    onStatus('connected', 'Connection established. Sending request...');
    ws.send(JSON.stringify(request));
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.content) {
        onContent(data.content);
      } else if (data.status) {
        onStatus(data.status, data.message);
        if (data.status === 'done') {
          // The server has signaled completion. We can close the socket.
          // The `onclose` event will fire, triggering the `onComplete` callback.
          ws.close();
        }
      } else if (data.error) {
        console.error('Received error from server:', data.error);
        onStatus('error', data.error);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', event.data, error);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    onError(error);
  };

  ws.onclose = (event) => {
    if (event.wasClean) {
      console.log('WebSocket connection closed cleanly');
    } else {
      console.warn('WebSocket connection died');
    }
    onComplete();
  };

  return ws;
};

/**
 * Closes a WebSocket connection
 * @param ws The WebSocket connection to close
 */
export const closeWebSocket = (ws: WebSocket | null): void => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
};
