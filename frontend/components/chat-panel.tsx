"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { Bot, Loader2, Mic, Send, Sparkles, Square, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

type Role = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
};

const ORCHESTRATOR_URL =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ?? "http://localhost:8000";

const SESSION_STORAGE_KEY = "pa-orchestrator-session";

/** Speech-to-Text v2 sync Recognize: keep clips under ~1 minute. */
const MAX_RECORDING_MS = 55_000;

function pickRecorderMimeType(): string | undefined {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];
  for (const t of types) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return undefined;
}

function extensionForMime(mime: string): string {
  if (mime.includes("webm")) return "webm";
  if (mime.includes("mp4")) return "mp4";
  if (mime.includes("ogg")) return "ogg";
  return "webm";
}

const SUGGESTIONS = [
  "What were our top cloud costs last week?",
  "Compare prod vs dev spend by service",
  "Show BigQuery-related costs for this month",
];

function friendlyApiError(message: string): string {
  return message.replace(/^\d+\s+[A-Z_]+:\s*/i, "").trim() || message;
}

function extractA2AText(payload: Record<string, unknown>): string {
  const status = payload.status as Record<string, unknown> | undefined;
  const message = status?.message as Record<string, unknown> | undefined;
  const artifact = payload.artifact as Record<string, unknown> | undefined;
  const fromMsg = message?.parts as { text?: string }[] | undefined;
  const fromArt = artifact?.parts as { text?: string }[] | undefined;
  const parts = fromMsg?.length ? fromMsg : fromArt ?? [];
  return parts.map((p) => p.text ?? "").join("");
}

function parseSseBlocks(buffer: string): { events: string[]; rest: string } {
  const events: string[] = [];
  let rest = buffer;
  let idx: number;
  while ((idx = rest.indexOf("\n\n")) !== -1) {
    const raw = rest.slice(0, idx);
    rest = rest.slice(idx + 2);
    const line = raw.split("\n").find((l) => l.startsWith("data:"));
    if (line) {
      events.push(line.replace(/^data:\s?/, "").trim());
    }
  }
  return { events, rest };
}

type HealthState = "checking" | "ok" | "error";

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthState>("checking");
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordChunksRef = useRef<BlobPart[]>([]);
  const recordMimeRef = useRef<string>("audio/webm");
  const recordLimitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    try {
      const existing = sessionStorage.getItem(SESSION_STORAGE_KEY);
      if (existing) setSessionId(existing);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      setHealth("checking");
      try {
        const r = await fetch(`${ORCHESTRATOR_URL}/health`, {
          method: "GET",
          cache: "no-store",
        });
        if (!cancelled) setHealth(r.ok ? "ok" : "error");
      } catch {
        if (!cancelled) setHealth("error");
      }
    };
    void check();
    const id = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const stopMediaTracks = useCallback(() => {
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
  }, []);

  const clearRecordLimitTimer = useCallback(() => {
    if (recordLimitTimerRef.current) {
      clearTimeout(recordLimitTimerRef.current);
      recordLimitTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      clearRecordLimitTimer();
      mediaRecorderRef.current?.stop();
      stopMediaTracks();
    };
  }, [clearRecordLimitTimer, stopMediaTracks]);

  const transcribeBlob = useCallback(async (blob: Blob, filename: string) => {
    setIsTranscribing(true);
    setVoiceError(null);
    try {
      const fd = new FormData();
      fd.append("file", blob, filename);
      const res = await fetch("/api/transcribe", {
        method: "POST",
        body: fd,
      });
      const data = (await res.json()) as { text?: string; error?: string };
      if (!res.ok) {
        setVoiceError(
          friendlyApiError(data.error || `Transcription failed (${res.status})`)
        );
        return;
      }
      const text = (data.text ?? "").trim();
      if (!text) {
        setVoiceError("No speech detected. Try again.");
        return;
      }
      setInput((prev) => {
        const p = prev.trim();
        return p ? `${p} ${text}` : text;
      });
      requestAnimationFrame(() => textareaRef.current?.focus());
    } catch (e) {
      setVoiceError(
        e instanceof Error ? e.message : "Could not reach transcription service."
      );
    } finally {
      setIsTranscribing(false);
    }
  }, []);

  const stopRecording = useCallback(() => {
    clearRecordLimitTimer();
    const rec = mediaRecorderRef.current;
    if (!rec || rec.state === "inactive") {
      setIsRecording(false);
      return;
    }
    try {
      if (rec.state === "recording" && "requestData" in rec) {
        (rec as MediaRecorder).requestData();
      }
    } catch {
      /* ignore */
    }
    rec.stop();
  }, [clearRecordLimitTimer]);

  const startRecording = useCallback(async () => {
    if (loading || isTranscribing || isRecording) return;
    setVoiceError(null);
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setVoiceError("Microphone is not available in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const mimeType = pickRecorderMimeType();
      recordMimeRef.current = mimeType ?? "audio/webm";
      const rec = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      recordChunksRef.current = [];
      rec.ondataavailable = (ev) => {
        if (ev.data.size > 0) recordChunksRef.current.push(ev.data);
      };
      rec.onerror = () => {
        setVoiceError("Recording error.");
        setIsRecording(false);
        stopMediaTracks();
        mediaRecorderRef.current = null;
        clearRecordLimitTimer();
      };
      rec.onstop = () => {
        stopMediaTracks();
        mediaRecorderRef.current = null;
        setIsRecording(false);
        const chunks = recordChunksRef.current;
        recordChunksRef.current = [];
        const mime = recordMimeRef.current || rec.mimeType || "audio/webm";
        const blob = new Blob(chunks, { type: mime });
        if (blob.size < 256) {
          setVoiceError("Recording too short.");
          return;
        }
        const ext = extensionForMime(mime);
        void transcribeBlob(blob, `recording.${ext}`);
      };
      mediaRecorderRef.current = rec;
      rec.start(250);
      setIsRecording(true);
      recordLimitTimerRef.current = setTimeout(() => {
        stopRecording();
      }, MAX_RECORDING_MS);
    } catch (e) {
      const name = e instanceof DOMException ? e.name : "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        setVoiceError("Microphone permission denied.");
      } else {
        setVoiceError(
          e instanceof Error ? e.message : "Could not start microphone."
        );
      }
    }
  }, [
    loading,
    isTranscribing,
    isRecording,
    stopMediaTracks,
    clearRecordLimitTimer,
    transcribeBlob,
    stopRecording,
  ]);

  const toggleVoice = useCallback(() => {
    if (isRecording) stopRecording();
    else void startRecording();
  }, [isRecording, stopRecording, startRecording]);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const sendMessage = async (override?: string) => {
    const text = (override ?? input).trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    const assistantId = crypto.randomUUID();
    setMessages((m) => [
      ...m,
      userMsg,
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setInput("");
    setLoading(true);
    requestAnimationFrame(() => textareaRef.current?.focus());

    try {
      const res = await fetch(`${ORCHESTRATOR_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          message: text,
          ...(sessionId ? { session_id: sessionId } : {}),
        }),
      });

      const sid = res.headers.get("X-Session-Id");
      if (sid) {
        setSessionId(sid);
        try {
          sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
        } catch {
          /* ignore */
        }
      }

      if (!res.ok || !res.body) {
        const errText = await res.text();
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: `Request failed (${res.status}): ${errText || res.statusText}`,
                }
              : msg
          )
        );
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let sseBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        sseBuffer += decoder.decode(value, { stream: true });
        const { events, rest } = parseSseBlocks(sseBuffer);
        sseBuffer = rest;

        for (const ev of events) {
          if (!ev) continue;
          let data: Record<string, unknown>;
          try {
            data = JSON.parse(ev) as Record<string, unknown>;
          } catch {
            continue;
          }
          if (data.error) {
            setMessages((m) =>
              m.map((msg) =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      content: String(
                        (data.detail as string) || "Orchestrator error"
                      ),
                    }
                  : msg
              )
            );
            continue;
          }
          const delta = extractA2AText(data);
          if (!delta) continue;
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: msg.content + delta }
                : msg
            )
          );
          scrollToBottom();
        }
      }
    } catch (e) {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: `Network error: ${e instanceof Error ? e.message : String(e)}`,
              }
            : msg
        )
      );
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
        <span className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
          Try asking
        </span>
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              disabled={loading}
              onClick={() => void sendMessage(s)}
              className={cn(
                "rounded-full border border-primary/20 bg-primary/[0.06] px-3.5 py-1.5 text-left text-xs font-medium text-foreground",
                "shadow-sm transition hover:border-primary/35 hover:bg-primary/[0.11] hover:shadow",
                "disabled:pointer-events-none disabled:opacity-50"
              )}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="relative flex min-h-[min(520px,70vh)] flex-col overflow-hidden rounded-2xl border border-border/70 bg-card shadow-xl shadow-primary/5 ring-1 ring-black/[0.04] dark:ring-white/[0.06]">
        <div className="flex items-center justify-between border-b border-border/70 bg-gradient-to-r from-muted/50 via-muted/30 to-transparent px-4 py-3.5">
          <div className="flex items-center gap-3 text-sm font-medium">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/12 text-primary shadow-inner ring-1 ring-primary/10">
              <Sparkles className="h-4 w-4" />
            </span>
            <div>
              <p className="leading-tight text-foreground">Orchestrator</p>
              <p className="text-xs font-normal text-muted-foreground">
                Routes cost questions to the specialist agent
              </p>
            </div>
          </div>
          <div
            className={cn(
              "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
              health === "ok" && "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
              health === "error" && "bg-destructive/15 text-destructive",
              health === "checking" && "bg-muted text-muted-foreground",

            )}
            title="Orchestrator /health"
          >
            {health === "checking" && (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            )}
            {health === "ok" && (
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
            )}
            {health === "error" && (
              <span className="h-2 w-2 rounded-full bg-destructive" />
            )}
            {health === "ok"
              ? "Connected"
              : health === "error"
                ? "Offline"
                : "…"}
          </div>
        </div>

        <ScrollArea className="flex-1 bg-gradient-to-b from-background/40 to-muted/10 p-4">
          <div className="flex flex-col gap-4 pr-3">
            {messages.length === 0 && (
              <div className="mx-auto max-w-md rounded-2xl border border-dashed border-primary/25 bg-primary/[0.04] px-6 py-9 text-center">
                <p className="text-sm font-semibold text-foreground">
                  Cloud cost &amp; usage only
                </p>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  The orchestrator answers greetings, then sends real cost
                  questions to the specialist via A2A. Unrelated requests get a
                  safe decline — try a suggestion above or type your own.
                </p>
              </div>
            )}
            {messages.map((m) => (
              <div
                key={m.id}
                className={cn(
                  "flex gap-3",
                  m.role === "user" ? "flex-row-reverse" : "flex-row"
                )}
              >
                <div
                  className={cn(
                    "mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border shadow-sm",
                    m.role === "user"
                      ? "border-primary/30 bg-primary text-primary-foreground"
                      : "border-border bg-background text-muted-foreground"
                  )}
                >
                  {m.role === "user" ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4" />
                  )}
                </div>
                <div
                  className={cn(
                    "max-w-[min(100%,42rem)] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm",
                    m.role === "user"
                      ? "bg-gradient-to-br from-primary to-primary/90 text-primary-foreground shadow-primary/20"
                      : "border border-border/70 bg-card text-foreground shadow-sm"
                  )}
                >
                  <p className="whitespace-pre-wrap break-words">
                    {m.content ||
                      (m.role === "assistant" && loading ? "" : m.content)}
                  </p>
                  {m.role === "assistant" && loading && !m.content && (
                    <span className="inline-flex gap-1 pt-0.5">
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground/70 [animation-delay:-0.3s]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground/70 [animation-delay:-0.15s]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground/70" />
                    </span>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>

        <form
          className="border-t border-border/70 bg-muted/20 p-3 backdrop-blur-md sm:p-4"
          onSubmit={(e) => {
            e.preventDefault();
            void sendMessage();
          }}
        >
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about cloud spend, services, environments, trends…"
              disabled={loading || isTranscribing}
              rows={2}
              autoComplete="off"
              className={cn(
                "min-h-[48px] flex-1 resize-y rounded-xl border border-input/90 bg-background px-3.5 py-3 text-sm",
                "shadow-sm outline-none ring-offset-background placeholder:text-muted-foreground",
                "focus-visible:border-primary/40 focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:ring-offset-2",
                "disabled:cursor-not-allowed disabled:opacity-50"
              )}
            />
            <div className="flex shrink-0 items-center justify-end gap-2 sm:h-[48px] sm:justify-start">
              <Button
                type="button"
                variant="outline"
                size="icon"
                disabled={loading || isTranscribing}
                onClick={() => toggleVoice()}
                className={cn(
                  "h-11 w-11 shrink-0 rounded-xl border-input/90 shadow-sm",
                  isRecording &&
                    "animate-pulse border-destructive/60 bg-destructive/10 text-destructive hover:bg-destructive/15 hover:text-destructive"
                )}
                aria-pressed={isRecording}
                aria-label={isRecording ? "Stop recording" : "Voice input"}
                title={
                  isRecording
                    ? "Stop and transcribe"
                    : "Voice input (Google Speech-to-Text)"
                }
              >
                {isTranscribing ? (
                  <Loader2 className="h-5 w-5 animate-spin" aria-hidden />
                ) : isRecording ? (
                  <Square className="h-5 w-5 fill-current" aria-hidden />
                ) : (
                  <Mic className="h-5 w-5" aria-hidden />
                )}
              </Button>
              <Button
                type="submit"
                disabled={loading || !input.trim() || isTranscribing}
                className="h-11 shrink-0 gap-2 rounded-xl px-6 font-semibold shadow-sm sm:h-[44px]"
              >
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
                Send
              </Button>
            </div>
          </div>
          {voiceError && (
            <div
              className="mt-3 rounded-lg border border-destructive/25 bg-destructive/[0.06] px-3 py-2.5 text-center text-sm text-destructive"
              role="alert"
            >
              {friendlyApiError(voiceError)}
            </div>
          )}
          <p className="mt-3 text-center text-[11px] leading-snug text-muted-foreground">
            Enter to send · Shift+Enter for newline · Mic → Google Speech-to-Text
            · Orchestrator SSE → cost agent
          </p>
        </form>
      </div>
    </div>
  );
}
