import { useEffect } from 'react'
import { X, Download } from 'lucide-react'
import MeasurementTable from './MeasurementTable'

interface Props {
  isOpen: boolean
  onClose: () => void
  data: Record<string, any>[]
  selectedIndex: number | null
  onRowClick: (rowIndex: number) => void
  onExport: (format: 'csv' | 'excel') => void
  measurementDone: boolean
}

export function MeasurementModal({
  isOpen,
  onClose,
  data,
  selectedIndex,
  onRowClick,
  onExport,
  measurementDone,
}: Props) {
  useEffect(() => {
    if (!isOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm">
      <div className="w-[95vw] h-[95vh] bg-white rounded-xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="shrink-0 flex items-center justify-between px-5 py-3 border-b">
          <h2 className="font-semibold text-gray-900">Measurement Results</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onExport('csv')}
              disabled={!measurementDone}
              className="flex items-center gap-1.5 text-sm px-3 py-1.5 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Download size={13} />
              CSV
            </button>
            <button
              onClick={() => onExport('excel')}
              disabled={!measurementDone}
              className="flex items-center gap-1.5 text-sm px-3 py-1.5 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Download size={13} />
              Excel
            </button>
            <button
              onClick={onClose}
              className="ml-2 p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
              title="Close (Esc)"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Table body */}
        <div className="flex-1 min-h-0">
          <MeasurementTable
            data={data}
            selectedIndex={selectedIndex}
            onRowClick={onRowClick}
          />
        </div>
      </div>
    </div>
  )
}
