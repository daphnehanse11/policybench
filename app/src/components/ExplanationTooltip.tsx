"use client";

import {
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

type TooltipPosition = {
  top: number;
  left: number;
};

export default function ExplanationTooltip({
  explanation,
  children = "why",
}: {
  explanation: string;
  children?: ReactNode;
}) {
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<TooltipPosition | null>(null);
  const tooltipId = useId();

  useEffect(() => {
    if (!open) return;

    const updatePosition = () => {
      const trigger = triggerRef.current;
      if (!trigger) return;

      const rect = trigger.getBoundingClientRect();
      setPosition({
        top: rect.bottom + 10,
        left: rect.left + rect.width / 2,
      });
    };

    updatePosition();
    window.addEventListener("scroll", updatePosition, true);
    window.addEventListener("resize", updatePosition);

    return () => {
      window.removeEventListener("scroll", updatePosition, true);
      window.removeEventListener("resize", updatePosition);
    };
  }, [open]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-describedby={open ? tooltipId : undefined}
        aria-label={explanation}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="inline-flex rounded-full border border-border px-1.5 py-0.5 text-[10px] leading-none text-text-muted transition-colors hover:border-primary/40 hover:text-text focus:border-primary/50 focus:text-text focus:outline-none cursor-help"
      >
        {children}
      </button>
      {open &&
        position &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            id={tooltipId}
            role="tooltip"
            className="pointer-events-none fixed z-[100] max-w-xs -translate-x-1/2 rounded-xl border border-border bg-surface px-3 py-2 text-left text-xs leading-relaxed text-text shadow-[0_12px_48px_rgba(0,0,0,0.35)]"
            style={{
              top: position.top,
              left: position.left,
            }}
          >
            {explanation}
          </div>,
          document.body,
        )}
    </>
  );
}
