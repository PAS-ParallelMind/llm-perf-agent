import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// Shared markdown renderer for assistant / user message content.
// GFM (tables, task lists, strikethrough) + KaTeX ($...$ and $$...$$).
// Code blocks are styled via .md pre in index.css — no syntax highlighting
// for now (keeps the bundle small).
export default function Markdown({ children }: { children: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
