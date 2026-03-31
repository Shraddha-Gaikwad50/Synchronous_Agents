import { ChatPanel } from "@/components/chat-panel";

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-background">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_90%_60%_at_50%_-25%,hsl(var(--primary)/0.18),transparent)]" aria-hidden />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_40%_at_100%_50%,hsl(173_40%_50%/0.06),transparent)]" aria-hidden />
      <div className="relative mx-auto flex min-h-screen max-w-3xl flex-col gap-8 px-4 py-10 md:max-w-4xl md:px-8 md:py-14">
        <header className="space-y-4 text-center md:text-left">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-primary/80">
            Local · Hybrid agent mesh
          </p>
          <h1 className="text-balance bg-gradient-to-br from-foreground via-foreground to-foreground/65 bg-clip-text text-3xl font-semibold tracking-tight text-transparent md:text-4xl">
            Cost intelligence chat
          </h1>
          <p className="mx-auto max-w-2xl text-pretty text-sm leading-relaxed text-muted-foreground md:mx-0 md:text-base">
            Talk to the conversational orchestrator on your machine. It routes
            cost and usage questions to the specialist agent over A2A; other
            topics get a clear, safe response.
          </p>
        </header>

        <ChatPanel />

        <footer className="pb-6 text-center text-xs leading-relaxed text-muted-foreground md:text-left">
          Ensure the orchestrator and cost agent are running (e.g.{" "}
          <code className="rounded-md border border-border/80 bg-muted/80 px-2 py-0.5 font-mono text-[0.75rem] text-foreground/90">
            scripts/start-all.sh
          </code>{" "}
          or{" "}
          <code className="rounded-md border border-border/80 bg-muted/80 px-2 py-0.5 font-mono text-[0.75rem] text-foreground/90">
            scripts/start-all.ps1
          </code>
          ).
        </footer>
      </div>
    </main>
  );
}
