import { NextRequest, NextResponse } from 'next/server';

// The target backend server base URL, derived from environment variable or defaulted.
const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const targetUrl = `${TARGET_SERVER_BASE_URL}/models/update`;

    // Make the actual request to the backend service
    const backendResponse = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(body)
    });

    // If the backend service responds with an error
    if (!backendResponse.ok) {
      return NextResponse.json(
        { error: `Backend service responded with status: ${backendResponse.status}` },
        { status: backendResponse.status }
      );
    }

    // Forward the response from the backend
    const result = await backendResponse.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error updating model selection:', error);    
    return new NextResponse(JSON.stringify({ error: error }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
  }
}

// Handle OPTIONS requests for CORS if needed
export function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}