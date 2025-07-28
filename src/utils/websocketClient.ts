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
  
  // Timeout configuration
  const CONNECTION_TIMEOUT = 30000; // 30 seconds for connection
  const RESPONSE_TIMEOUT = 180000;  // 3 minutes for response
  const CHUNK_TIMEOUT = 45000;     // 45 seconds between chunks
  
  let connectionTimeout: NodeJS.Timeout;
  let responseTimeout: NodeJS.Timeout;
  let chunkTimeout: NodeJS.Timeout;
  let lastActivity = Date.now();
  let isCompleted = false;
  
  // Helper function to clear all timeouts
  const clearAllTimeouts = () => {
    if (connectionTimeout) clearTimeout(connectionTimeout);
    if (responseTimeout) clearTimeout(responseTimeout);
    if (chunkTimeout) clearTimeout(chunkTimeout);
  };
  
  // Helper function to handle timeout scenarios
  const handleTimeout = (type: string) => {
    if (isCompleted) return;
    
    console.error(`WebSocket ${type} timeout`);
    isCompleted = true;
    clearAllTimeouts();
    
    const timeoutMessages = {
      connection: 'Connection timeout - failed to establish connection within 30 seconds',
      response: 'Response timeout - no response received within 3 minutes',
      chunk: 'Stream timeout - no data received within 45 seconds'
    };
    
    onStatus('timeout', timeoutMessages[type as keyof typeof timeoutMessages] || 'Timeout occurred');
    
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };
  
  // Set connection timeout
  connectionTimeout = setTimeout(() => handleTimeout('connection'), CONNECTION_TIMEOUT);

  ws.onopen = () => {
    console.log('WebSocket connection established');
    clearTimeout(connectionTimeout);
    onStatus('connected', 'Connection established. Sending request...');
    ws.send(JSON.stringify(request));
    
    // Set response timeout after sending request
    responseTimeout = setTimeout(() => handleTimeout('response'), RESPONSE_TIMEOUT);
    
    // Start chunk timeout monitoring
    chunkTimeout = setTimeout(() => handleTimeout('chunk'), CHUNK_TIMEOUT);
  };

  ws.onmessage = (event) => {
    if (isCompleted) return;
    
    // Reset chunk timeout on any message received
    clearTimeout(chunkTimeout);
    chunkTimeout = setTimeout(() => handleTimeout('chunk'), CHUNK_TIMEOUT);
    lastActivity = Date.now();
    
    try {
      const data = JSON.parse(event.data);

      if (data.content) {
        onContent(data.content);
        // Clear response timeout once we start receiving content
        if (responseTimeout) {
          clearTimeout(responseTimeout);
          responseTimeout = null;
        }
      } else if (data.status) {
        onStatus(data.status, data.message);
        
        // Handle heartbeat messages
        if (data.status === 'heartbeat') {
          console.log('Received heartbeat from server');
          return;
        }
        
        if (data.status === 'done') {
          // The server has signaled completion
          isCompleted = true;
          clearAllTimeouts();
          ws.close();
        }
      } else if (data.error) {
        console.error('Received error from server:', data.error);
        onStatus('error', data.error);
        isCompleted = true;
        clearAllTimeouts();
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', event.data, error);
      onStatus('error', 'Failed to parse server message');
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    if (!isCompleted) {
      isCompleted = true;
      clearAllTimeouts();
      onError(error);
    }
  };

  ws.onclose = (event) => {
    if (!isCompleted) {
      isCompleted = true;
      clearAllTimeouts();
    }
    
    if (event.wasClean) {
      console.log('WebSocket connection closed cleanly');
    } else {
      console.warn('WebSocket connection died unexpectedly', {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean
      });
      
      // If connection died unexpectedly and we haven't completed, treat as error
      if (!event.wasClean && !isCompleted) {
        onStatus('error', 'Connection lost unexpectedly');
      }
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
  if (ws) {
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      console.log('Closing WebSocket connection');
      ws.close(1000, 'Client requested close');
    }
  }
};
