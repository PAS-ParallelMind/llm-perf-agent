import { useState } from "react";
import Chat from "./pages/Chat";
import Trace from "./pages/Trace";
import ThemeToggle from "./components/ThemeToggle";

type Tab = "chat" | "trace";

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");

  return (
    <div className="flex flex-col h-full">
      <header className="border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center gap-6">
          <h1 className="text-lg font-semibold">
            llm-perf-agent <span className="text-slate-400">/ webui</span>
          </h1>
          <nav className="flex items-center gap-1">
            <TabButton active={tab === "chat"}  onClick={() => setTab("chat")}>Chat</TabButton>
            <TabButton active={tab === "trace"} onClick={() => setTab("trace")}>Trace</TabButton>
          </nav>
          <span className="text-sm text-slate-500 hidden md:inline">
            {tab === "chat"
              ? "live agent — markdown-rendered"
              : "raw conversation log"}
          </span>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-hidden">
        {/* Chat manages its own scroll (sticky composer at the bottom);
            Trace is a long scrollable list. */}
        {tab === "chat"
          ? <Chat />
          : <div className="h-full overflow-auto"><Trace /></div>}
      </main>
    </div>
  );
}

function TabButton(
  { active, children, onClick }:
  { active: boolean; children: React.ReactNode; onClick: () => void },
) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "px-3 py-1 text-sm rounded transition " +
        (active
          ? "bg-indigo-600 text-white"
          : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800")
      }
    >
      {children}
    </button>
  );
}
