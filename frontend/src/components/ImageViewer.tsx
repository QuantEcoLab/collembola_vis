import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'
import { ZoomIn, ZoomOut, Maximize } from 'lucide-react'

interface Props {
  src: string
  alt?: string
  className?: string
  onImageClick?: (x: number, y: number) => void
  overlay?: React.ReactNode
  originalWidth?: number
  originalHeight?: number
}

export default function ImageViewer({
  src,
  alt = 'Image',
  className = '',
  onImageClick,
  overlay,
}: Props) {
  return (
    <div className={`relative bg-gray-100 rounded-lg overflow-hidden ${className}`}>
      <TransformWrapper
        initialScale={1}
        minScale={0.1}
        maxScale={10}
        doubleClick={{ disabled: true }}
      >
        {({ zoomIn, zoomOut, resetTransform }) => (
          <>
            <div className="absolute top-2 right-2 z-10 flex gap-1">
              <button
                onClick={() => zoomIn()}
                className="bg-white/90 hover:bg-white p-1.5 rounded shadow-sm"
              >
                <ZoomIn size={16} />
              </button>
              <button
                onClick={() => zoomOut()}
                className="bg-white/90 hover:bg-white p-1.5 rounded shadow-sm"
              >
                <ZoomOut size={16} />
              </button>
              <button
                onClick={() => resetTransform()}
                className="bg-white/90 hover:bg-white p-1.5 rounded shadow-sm"
              >
                <Maximize size={16} />
              </button>
            </div>
            <TransformComponent
              wrapperStyle={{ width: '100%', height: '100%' }}
              contentStyle={{ width: '100%', height: '100%' }}
            >
              <div className="relative inline-block">
                <img
                  src={src}
                  alt={alt}
                  className="max-w-full max-h-full object-contain"
                  onClick={(e) => {
                    if (!onImageClick) return
                    const rect = e.currentTarget.getBoundingClientRect()
                    const x = ((e.clientX - rect.left) / rect.width) * e.currentTarget.naturalWidth
                    const y = ((e.clientY - rect.top) / rect.height) * e.currentTarget.naturalHeight
                    onImageClick(x, y)
                  }}
                />
                {overlay}
              </div>
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  )
}
