import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import Analyzer from "./pages/Analyzer";
import Benchmarks from "./pages/Benchmarks";
import Trace from "./pages/Trace";
import ThemeToggle from "./components/ThemeToggle";

function NavTab({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
          isActive
            ? "border-indigo-500 text-indigo-700 dark:text-indigo-300"
            : "border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
        }`
      }
    >
      {label}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="flex flex-col h-full">
      <header className="border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center gap-6">
          <h1 className="text-lg font-semibold">
            llm-perf-agent <span className="text-slate-400">/ webui</span>
          </h1>
          <nav className="flex items-center">
            <NavTab to="/analyzer"   label="Analyzer" />
            <NavTab to="/benchmarks" label="Benchmarks" />
            <NavTab to="/trace"      label="Trace" />
          </nav>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/analyzer" replace />} />
          <Route path="/analyzer"   element={<Analyzer />} />
          <Route path="/benchmarks" element={<Benchmarks />} />
          <Route path="/trace"      element={<Trace />} />
        </Routes>
      </main>
    </div>
  );
}
