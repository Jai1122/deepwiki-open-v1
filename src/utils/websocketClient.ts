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
  
  // Timeout configuration - increased for complex responses
  const CONNECTION_TIMEOUT = 30000; // 30 seconds for connection
  const RESPONSE_TIMEOUT = 300000;  // 5 minutes for response
  const CHUNK_TIMEOUT = 120000;    // 2 minutes between chunks (increased from 90s)
  const HEARTBEAT_INTERVAL = 120000; // Send heartbeat every 2 minutes (reduced frequency)
  
  let connectionTimeout: NodeJS.Timeout;
  let responseTimeout: NodeJS.Timeout;
  let chunkTimeout: NodeJS.Timeout;
  let heartbeatInterval: NodeJS.Timeout;
  let lastActivity = Date.now();
  let isCompleted = false;
  let requestSent = false;
  
  // Helper function to clear all timeouts
  const clearAllTimeouts = () => {
    if (connectionTimeout) clearTimeout(connectionTimeout);
    if (responseTimeout) clearTimeout(responseTimeout);
    if (chunkTimeout) clearTimeout(chunkTimeout);
    if (heartbeatInterval) clearInterval(heartbeatInterval);
  };
  
  // Helper function to handle timeout scenarios
  const handleTimeout = (type: string) => {
    if (isCompleted) return;
    
    console.error(`WebSocket ${type} timeout`);
    isCompleted = true;
    clearAllTimeouts();
    
    const timeoutMessages = {
      connection: 'Connection timeout - failed to establish connection within 30 seconds',
      response: 'Response timeout - no response received within 5 minutes',
      chunk: 'Stream timeout - no data received within 2 minutes'
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
    requestSent = true;
    
    // Set response timeout after sending request
    responseTimeout = setTimeout(() => handleTimeout('response'), RESPONSE_TIMEOUT);
    
    // Start chunk timeout monitoring
    chunkTimeout = setTimeout(() => handleTimeout('chunk'), CHUNK_TIMEOUT);
    
    // Start heartbeat to keep connection alive during long operations (only after request is sent)
    heartbeatInterval = setInterval(() => {
      if (!isCompleted && requestSent && ws.readyState === WebSocket.OPEN) {
        try {
          console.log('Sending heartbeat ping to server');
          ws.send(JSON.stringify({ type: 'ping' }));
        } catch (error) {
          console.warn('Failed to send heartbeat ping:', error);
          // If we can't send a ping, the connection might be dead
          if (!isCompleted) {
            handleTimeout('chunk');
          }
        }
      }
    }, HEARTBEAT_INTERVAL);
  };

  ws.onmessage = (event) => {
    if (isCompleted) return;
    
    // Reset chunk timeout on any message received
    clearTimeout(chunkTimeout);
    chunkTimeout = setTimeout(() => handleTimeout('chunk'), CHUNK_TIMEOUT);
    lastActivity = Date.now();
    
    try {
      const data = JSON.parse(event.data);

      // Handle pong responses first
      if (data.type === 'pong') {
        console.log('Received pong from server');
        return;
      }
      
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
    clearAllTimeouts();
    
    // Handle specific close codes for better error reporting
    const isNormalClosure = event.code === 1000 || event.code === 1001;
    const isAbnormalClosure = event.code === 1005 || event.code === 1006;
    const isServerShutdown = event.code === 1012 || event.code === 1013;
    const isGoingAway = event.code === 1001;
    
    console.log('WebSocket connection closed', { 
      code: event.code, 
      reason: event.reason, 
      wasClean: event.wasClean,
      isCompleted 
    });
    
    if (event.wasClean || isNormalClosure) {
      console.log('WebSocket connection closed cleanly');
      if (isCompleted) {
        onComplete();
      } else {
        console.warn('WebSocket closed cleanly but completion was not signaled');
        onStatus('error', 'Response was incomplete - please retry the operation');
      }
    } else if (isAbnormalClosure) {
      console.warn('WebSocket connection had abnormal closure (network issue or server restart)');
      
      if (!isCompleted) {
        onStatus('error', 'Connection lost - please check your network and retry');
      } else {
        onComplete();
      }
    } else if (isServerShutdown) {
      console.warn('WebSocket connection closed due to server maintenance');
      onStatus('error', 'Server is temporarily unavailable - please retry in a few moments');
    } else if (isGoingAway) {
      console.warn('WebSocket connection closed because server is going away');
      if (!isCompleted) {
        onStatus('error', 'Server connection lost - please retry');
      } else {
        onComplete();
      }
    } else {
      console.warn('WebSocket connection closed unexpectedly', {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean
      });
      
      if (!isCompleted) {
        const errorMsg = event.code ? `Connection error (${event.code}) - please retry` : 'Connection lost - please retry';
        onStatus('error', errorMsg);
      } else {
        onComplete();
      }
    }
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
