import { NextResponse } from 'next/server';

// Ensure this matches your Python backend configuration
const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_HOST || 'http://localhost:8001';
const EXPORT_API_ENDPOINT = `${PYTHON_BACKEND_URL}/export/wiki`;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    console.log('Forwarding export request to Python backend:', body);

    const response = await fetch(EXPORT_API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      let errorBody = { error: `Export failed: ${response.statusText}` };
      try {
        errorBody = await response.json();
      } catch {
        // Keep default error if parsing fails
      }
      console.error(`Error from Python backend (${EXPORT_API_ENDPOINT}): ${response.status} - ${JSON.stringify(errorBody)}`);
      return NextResponse.json(errorBody, { status: response.status });
    }

    // For file downloads, we need to stream the response
    const contentType = response.headers.get('content-type') || 'application/octet-stream';
    const contentDisposition = response.headers.get('content-disposition');
    
    const responseBody = await response.arrayBuffer();
    
    const headers: Record<string, string> = {
      'Content-Type': contentType,
    };
    
    if (contentDisposition) {
      headers['Content-Disposition'] = contentDisposition;
    }

    return new NextResponse(responseBody, {
      status: 200,
      headers: headers,
    });

  } catch (error: unknown) {
    console.error(`Network or other error when calling ${EXPORT_API_ENDPOINT}:`, error);
    const message = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json(
      { error: `Failed to export wiki: ${message}` },
      { status: 503 }
    );
  }
}