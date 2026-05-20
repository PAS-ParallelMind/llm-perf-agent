import Trace from "./pages/Trace";
import ThemeToggle from "./components/ThemeToggle";

export default function App() {
  return (
    <div className="flex flex-col h-full">
      <header className="border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center gap-6">
          <h1 className="text-lg font-semibold">
            llm-perf-agent <span className="text-slate-400">/ webui</span>
          </h1>
          <span className="text-sm text-slate-500">chat-session trace viewer</span>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-auto">
        <Trace />
      </main>
    </div>
  );
}
