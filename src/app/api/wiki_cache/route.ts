import { NextResponse } from 'next/server';

// Ensure this matches your Python backend configuration
const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_HOST || 'http://localhost:8001';
const CACHE_API_ENDPOINT = `${PYTHON_BACKEND_URL}/api/wiki_cache`;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    console.log('Forwarding wiki cache save request to Python backend:', {
      ...body,
      generated_pages: Object.keys(body.generated_pages || {}).length // Don't log full content
    });

    const response = await fetch(CACHE_API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      let errorBody = { error: `Cache save failed: ${response.statusText}` };
      try {
        errorBody = await response.json();
      } catch {
        // Keep default error if parsing fails
      }
      console.error(`Error from Python backend (${CACHE_API_ENDPOINT}): ${response.status} - ${JSON.stringify(errorBody)}`);
      return NextResponse.json(errorBody, { status: response.status });
    }

    const result = await response.json();
    return NextResponse.json(result);

  } catch (error: unknown) {
    console.error(`Network or other error when calling ${CACHE_API_ENDPOINT}:`, error);
    const message = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json(
      { error: `Failed to save wiki cache: ${message}` },
      { status: 503 }
    );
  }
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const owner = searchParams.get('owner');
    const repo = searchParams.get('repo');
    const repo_type = searchParams.get('repo_type');
    const language = searchParams.get('language');

    if (!owner || !repo || !repo_type || !language) {
      return NextResponse.json(
        { error: 'Missing required parameters: owner, repo, repo_type, language' },
        { status: 400 }
      );
    }

    const params = new URLSearchParams({ owner, repo, repo_type, language });
    const response = await fetch(`${CACHE_API_ENDPOINT}?${params}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      let errorBody = { error: `Cache fetch failed: ${response.statusText}` };
      try {
        errorBody = await response.json();
      } catch {
        // Keep default error if parsing fails
      }
      console.error(`Error from Python backend (${CACHE_API_ENDPOINT}): ${response.status} - ${JSON.stringify(errorBody)}`);
      return NextResponse.json(errorBody, { status: response.status });
    }

    const result = await response.json();
    return NextResponse.json(result);

  } catch (error: unknown) {
    console.error(`Network or other error when calling ${CACHE_API_ENDPOINT}:`, error);
    const message = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json(
      { error: `Failed to fetch wiki cache: ${message}` },
      { status: 503 }
    );
  }
}