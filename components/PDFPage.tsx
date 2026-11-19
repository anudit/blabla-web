import { useEffect, useRef, useState } from 'preact/hooks';
import type { CSSProperties } from 'preact';
import { staticStyles } from '../theme';

interface PDFPageProps {
  data: any;
  pdfDoc: any;
  onLineClick: (lineId: number) => void;
  pageContainerStyle: CSSProperties;
}

const PDFPage = ({ data, pdfDoc, onLineClick, pageContainerStyle }: PDFPageProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const renderTaskRef = useRef<any>(null);
  const pageRef = useRef<any>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !pdfDoc) return;
    let renderedWidth = 0; // backing-store width currently on the canvas (0 = released)
    let disposed = false;

    // Backing store sized to the displayed width × dpr (never above the
    // layout viewport), instead of the full scale-2 page bitmap.
    const targetWidth = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const css = container.clientWidth || data.viewport.width / dpr;
      return Math.max(1, Math.min(Math.round(css * dpr), Math.round(data.viewport.width)));
    };

    const renderPage = async () => {
      if (disposed || !canvasRef.current) return;
      const width = targetWidth();
      if (width === renderedWidth) return;
      if (renderTaskRef.current) { try { renderTaskRef.current.cancel(); } catch (e) {} renderTaskRef.current = null; }
      try {
        const page = pageRef.current ?? await pdfDoc.getPage(data.pageNumber);
        pageRef.current = page;
        const canvas = canvasRef.current;
        if (!canvas || disposed) return;
        const viewport = page.getViewport({ scale: data.viewport.scale * (width / data.viewport.width) });
        canvas.width = Math.round(viewport.width);
        canvas.height = Math.round(viewport.height);
        renderedWidth = width;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const renderTask = page.render({ canvasContext: ctx, viewport });
        renderTaskRef.current = renderTask;
        await renderTask.promise;
      } catch (e: any) {
        if (e.name !== 'RenderingCancelledException' && e.name !== 'RenderCancelledError') console.error("Render error:", e);
      }
    };

    // Free the bitmap (GPU/IOSurface memory) but keep the line overlay so
    // sentence highlighting and scrollIntoView keep working.
    const releaseCanvas = () => {
      if (renderTaskRef.current) { try { renderTaskRef.current.cancel(); } catch (e) {} renderTaskRef.current = null; }
      if (pageRef.current) { pageRef.current.cleanup(); pageRef.current = null; }
      if (canvasRef.current) { canvasRef.current.width = 0; canvasRef.current.height = 0; }
      renderedWidth = 0;
    };

    // Hysteresis: render within 1500px of the viewport, release beyond 4000px,
    // so fast scrolling back and forth doesn't thrash or flash blank pages.
    const near = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting) { setIsVisible(true); setTimeout(renderPage, 0); }
    }, { rootMargin: '1500px 0px' });
    const far = new IntersectionObserver((entries) => {
      if (entries[0] && !entries[0].isIntersecting) releaseCanvas();
    }, { rootMargin: '4000px 0px' });
    near.observe(container);
    far.observe(container);

    // Re-render at the new resolution when the layout width changes noticeably.
    const resize = new ResizeObserver(() => {
      if (renderedWidth && Math.abs(targetWidth() - renderedWidth) > renderedWidth * 0.1) renderPage();
    });
    resize.observe(container);

    return () => { disposed = true; near.disconnect(); far.disconnect(); resize.disconnect(); releaseCanvas(); };
  }, [data, pdfDoc]);

  const aspectRatio = data.viewport.height / data.viewport.width;

  return (
    <div ref={containerRef} style={{ ...pageContainerStyle, maxWidth: data.viewport.width }}>
      <div style={{ position: 'relative', width: '100%', paddingBottom: `${aspectRatio * 100}%` }}>
        {isVisible && (
          <>
            <canvas ref={canvasRef} style={{ ...staticStyles.canvas, position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
            <div style={{ ...staticStyles.overlay, position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
              {data.lines.map((line: any) => (
                <PDFLine key={line.id} line={line} onLineClick={onLineClick} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const PDFLine = ({ line, onLineClick }: any) => {
  return (
    <div
      id={`line-${line.id}`}
      data-line-id={line.id}
      onClick={(e) => { e.stopPropagation(); onLineClick(line.id); }}
      style={{ ...staticStyles.lineBase, left: line.left, top: line.top, width: line.width, height: line.height }}
    />
  );
};

export default PDFPage;
