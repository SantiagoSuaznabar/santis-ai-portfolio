import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, deleteSession, streamChat, type LogEvent } from './api';
import styles from './Rag.module.css';

// ─── Types ───────────────────────────────────────────────────────────────────

type Message = {
  role: 'user' | 'assistant';
  content: string;
  results?: { content: string; relevance_rank: number; relevance_score: number }[];
  fromCache?: boolean;
  isStreaming?: boolean;
  isError?: boolean;
};

type LogEntryWithId = LogEvent & { id: number; relativeTs?: string };

interface StackItem {
  name: string;
  role: string;
  color?: string;
}

interface Props {
  description?: string;
  stack?: StackItem[];
}

// Token drip speed — raise to slow down visible streaming
const TOKEN_DRIP_MS = 35;

// ─── Sub-components ───────────────────────────────────────────────────────────

function LogLine({ log }: { log: LogEntryWithId }) {
  return (
    <div className={`${styles.logEntry} ${styles[`level-${log.level}`]}`}>
      <div className={styles.logMeta}>
        <span className={`${styles.logBadge} ${styles[`level-${log.level}`]}`}>{log.level}</span>
        <span className={styles.logModule}>{log.module}</span>
        {log.relativeTs && <span className={styles.logTs}>{log.relativeTs}</span>}
      </div>
      <div className={styles.logMessage}>{log.message}</div>
    </div>
  );
}

function SourceCards({ results }: { results: Message['results'] }) {
  if (!results || results.length === 0) return null;
  return (
    <div className={styles.sourceCards}>
      <div className={styles.sourceCardsLabel}>
        {results.length} source{results.length !== 1 ? 's' : ''} retrieved
      </div>
      {results.map((r, i) => (
        <div key={i} className={styles.sourceCard}>
          <span className={styles.sourceRank}>#{r.relevance_rank}</span>
          <span className={styles.sourceContent}>{r.content}</span>
          <span className={styles.sourceScore}>{(r.relevance_score * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  if (msg.role === 'user') {
    return (
      <div className={`${styles.messageRow} ${styles.user}`}>
        <div className={styles.roleLabel}>You</div>
        <div className={styles.messageBubble}>{msg.content}</div>
      </div>
    );
  }
  if (msg.isError) {
    return (
      <div className={`${styles.messageRow} ${styles.assistant}`}>
        <div className={styles.roleLabel}>System</div>
        <div className={styles.errorBubble}>⚠ {msg.content}</div>
      </div>
    );
  }
  return (
    <div className={`${styles.messageRow} ${styles.assistant}`}>
      <div className={styles.roleLabel}>RAG</div>
      <div className={styles.messageBubble}>
        {msg.fromCache && <div className={styles.cacheBadge}>⚡ cached response</div>}
        <span>
          {msg.content}
          {msg.isStreaming && <span className={styles.streamCursor} />}
        </span>
        <SourceCards results={msg.results} />
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Rag({ description, stack = [] }: Props) {
  const [sessionId, setSessionId]   = useState<string | null>(null);
  const [messages, setMessages]     = useState<Message[]>([]);
  const [logs, setLogs]             = useState<LogEntryWithId[]>([]);
  const [input, setInput]           = useState('');
  const [isLoading, setIsLoading]   = useState(false);
  const [isConnecting, setIsConnecting] = useState(true);

  const sessionIdRef  = useRef<string | null>(null);
  const chatEndRef    = useRef<HTMLDivElement>(null);
  const logsEndRef    = useRef<HTMLDivElement>(null);
  const logIdRef      = useRef(0);
  const firstTsRef    = useRef<number | null>(null);
  const textareaRef   = useRef<HTMLTextAreaElement>(null);

  // Token drip queue — decouples SSE cadence from React renders
  const tokenQueueRef  = useRef<string[]>([]);
  const dripTimerRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamDoneRef  = useRef(false);

  function startDrip() {
    if (dripTimerRef.current) return;
    dripTimerRef.current = setInterval(() => {
      const token = tokenQueueRef.current.shift();
      if (token !== undefined) {
        setMessages(prev =>
          prev.map((m, i) =>
            i === prev.length - 1 && m.role === 'assistant'
              ? { ...m, content: m.content + token }
              : m
          )
        );
      } else if (streamDoneRef.current) {
        clearInterval(dripTimerRef.current!);
        dripTimerRef.current = null;
        setIsLoading(false);
        setMessages(prev =>
          prev.map((m, i) =>
            i === prev.length - 1 && m.role === 'assistant'
              ? { ...m, isStreaming: false }
              : m
          )
        );
      }
    }, TOKEN_DRIP_MS);
  }

  function stopDrip() {
    if (dripTimerRef.current) { clearInterval(dripTimerRef.current); dripTimerRef.current = null; }
    tokenQueueRef.current = [];
    streamDoneRef.current = false;
  }

  // Session lifecycle
  useEffect(() => {
    async function init() {
      try {
        const id = await createSession();
        setSessionId(id);
        sessionIdRef.current = id;
      } finally { setIsConnecting(false); }
    }
    init();
    return () => { stopDrip(); if (sessionIdRef.current) deleteSession(sessionIdRef.current); };
  }, []);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
  };

  const handleSubmit = useCallback(async () => {
    const userMsg = input.trim();
    if (!userMsg || !sessionId || isLoading) return;

    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    stopDrip();
    firstTsRef.current = null;

    setMessages(prev => [
      ...prev,
      { role: 'user', content: userMsg },
      { role: 'assistant', content: '', isStreaming: true },
    ]);
    setIsLoading(true);

    await streamChat(userMsg, sessionId, {
      onLog: (log) => {
        const id = ++logIdRef.current;
        if (firstTsRef.current === null) firstTsRef.current = log.ts;
        const ms = Math.round((log.ts - firstTsRef.current) * 1000);
        const relativeTs = ms < 1000 ? `+${ms}ms` : `+${(ms / 1000).toFixed(1)}s`;
        setLogs(prev => [...prev, { ...log, id, relativeTs }]);
      },
      onMeta: (meta) => {
        setMessages(prev =>
          prev.map((m, i) =>
            i === prev.length - 1 && m.role === 'assistant'
              ? { ...m, results: meta.results, fromCache: meta.from_cache }
              : m
          )
        );
      },
      onToken: (token) => { tokenQueueRef.current.push(token); startDrip(); },
      onDone: (debug) => {
        streamDoneRef.current = true;
        if (debug) console.debug('[RAG done]', debug);
      },
      onError: (err) => {
        stopDrip();
        setIsLoading(false);
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last?.role === 'assistant' && last.isStreaming) {
            return [...prev.slice(0, -1), { role: 'assistant', content: err, isError: true }];
          }
          return [...prev, { role: 'assistant', content: err, isError: true }];
        });
      },
    });
  }, [input, sessionId, isLoading]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={styles.shell}>

      {/* ── Left sidebar: stack info + logs ── */}
      <aside className={styles.sidebar}>

        {/* Stack info */}
        <div className={styles.sideSection}>
          <div className={styles.sideSectionHeader}>
            <span className={styles.termDot} />
            <span className={styles.termDot} />
            <span className={styles.termDot} />
            <span className={styles.sideSectionTitle}>Stack</span>
          </div>

          {description && (
            <p className={styles.moduleDesc}>{description}</p>
          )}

          {stack.length > 0 && (
            <div className={styles.stackList}>
              {stack.map(s => (
                <div key={s.name} className={styles.stackItem}>
                  <span
                    className={styles.stackDot}
                    style={{ background: s.color || '#3b82f6' }}
                  />
                  <span className={styles.stackName}>{s.name}</span>
                  <span className={styles.stackRole}>{s.role}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Logs */}
        <div className={styles.sideSection} style={{ flex: 1, minHeight: 0 }}>
          <div className={styles.sideSectionHeader}>
            <span className={styles.sideSectionTitle}>Pipeline Logs</span>
          </div>

          <div className={`${styles.logsScroll} scroll-thin`}>
            {logs.length === 0 ? (
              <div className={styles.logsEmpty}>
                <span className={styles.logsEmptyIcon}>⬡</span>
                <span>Awaiting events…</span>
              </div>
            ) : (
              logs.map(log => <LogLine key={log.id} log={log} />)
            )}
            <div ref={logsEndRef} />
          </div>
        </div>

        {/* Status bar */}
        <div className={styles.statusBar}>
          <span className={`${styles.statusDot} ${isConnecting ? styles.connecting : ''}`} />
          {isConnecting ? 'connecting…' : sessionId ? `session ${sessionId.slice(0, 8)}…` : 'no session'}
        </div>
      </aside>

      {/* ── Right: Chat ── */}
      <section className={styles.chatPanel}>
        <div className={`${styles.chatWindow} scroll-thin`}>
          {messages.length === 0 ? (
            <div className={styles.welcomeScreen}>
              <div className={styles.welcomeOrb}>⬡</div>
              <div className={styles.welcomeTitle}>Ask anything</div>
              <div className={styles.welcomeSub}>
                Queries are grounded in retrieved passages. Watch the pipeline logs on the left as each stage runs.
              </div>
            </div>
          ) : (
            messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)
          )}

          {isLoading && messages[messages.length - 1]?.content === '' && (
            <div className={styles.thinking}>
              <div className={styles.thinkingDots}><span /><span /><span /></div>
              Retrieving context…
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className={styles.inputArea}>
          <div className={styles.inputWrapper}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask something…"
              disabled={isLoading || isConnecting}
              rows={1}
            />
            <button
              className={styles.sendBtn}
              onClick={handleSubmit}
              disabled={isLoading || isConnecting || !input.trim()}
              aria-label="Send"
            >
              <svg className={styles.sendIcon} viewBox="0 0 24 24">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
          <div className={styles.inputHint}>↵ send · ⇧↵ newline</div>
        </div>
      </section>
    </div>
  );
}