const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function createSession(): Promise<string> {
  const res = await fetch(`${BASE_URL}/api/session`, { method: 'POST' });
  const data = await res.json();
  return data.session_id;
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${BASE_URL}/api/session/${sessionId}`, { method: 'DELETE' });
}

export interface LogEvent {
  type: 'log';
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  module: string;
  message: string;
  ts: number;
}

export async function streamChat(
  message: string,
  sessionId: string,
  callbacks: {
    onLog?: (log: LogEvent) => void;
    onMeta: (data: any) => void;
    onToken: (token: string) => void;
    onDone: (debug: any) => void;
    onError: (msg: string) => void;
  }
) {
  try {
    const res = await fetch(`${BASE_URL}/api/chat/rag/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (res.status === 429) {
      const retryAfter = res.headers.get('Retry-After');
      const seconds = retryAfter ? parseInt(retryAfter) : 60;
      callbacks.onError(`Rate limit exceeded. Please wait ${seconds}s before retrying.`);
      return;
    }

    if (!res.ok) {
      callbacks.onError(`Server error: ${res.status}`);
      return;
    }

    if (!res.body) throw new Error('No response body');

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6));

        if (event.type === 'log')   callbacks.onLog?.(event as LogEvent);
        if (event.type === 'meta')  callbacks.onMeta(event);
        if (event.type === 'token') callbacks.onToken(event.content);
        if (event.type === 'done')  callbacks.onDone(event.debug);
        if (event.type === 'error') callbacks.onError(event.message);
      }
    }
  } catch (error: any) {
    callbacks.onError(error.message || 'Connection failed');
  }
}