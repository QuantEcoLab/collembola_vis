import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
  type VisibilityState,
} from '@tanstack/react-table'
import { useState, useMemo, useEffect, useRef } from 'react'
import { ArrowUpDown, Columns3 } from 'lucide-react'

// Columns hidden by default — raw pixel coords and metadata noise
const DEFAULT_HIDDEN = new Set([
  'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
  'bbox_width_px', 'bbox_height_px',
  'centroid_x_px', 'centroid_y_px',
  'area_px', 'perimeter_px',
  'major_axis_px', 'minor_axis_px',
  'eccentricity', 'solidity',
  'confidence', 'method',
])

interface MeasurementRow {
  [key: string]: number | string
}

interface Props {
  data: MeasurementRow[]
  selectedIndex?: number | null
  onRowClick?: (originalIndex: number) => void
}

export default function MeasurementTable({ data, selectedIndex, onRowClick }: Props) {
  const [sorting, setSorting] = useState<SortingState>([])
  const [showColMenu, setShowColMenu] = useState(false)
  const colMenuRef = useRef<HTMLDivElement>(null)
  const tbodyRef = useRef<HTMLTableSectionElement>(null)

  // Column visibility — initialised once from data keys
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(() => {
    if (!data.length) return {}
    return Object.fromEntries(Object.keys(data[0]).map((k) => [k, !DEFAULT_HIDDEN.has(k)]))
  })

  // Close column menu on outside click
  useEffect(() => {
    if (!showColMenu) return
    const handler = (e: MouseEvent) => {
      if (colMenuRef.current && !colMenuRef.current.contains(e.target as Node)) {
        setShowColMenu(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showColMenu])

  const columns = useMemo(() => {
    if (!data.length) return []
    const helper = createColumnHelper<MeasurementRow>()
    return Object.keys(data[0]).map((key) =>
      helper.accessor(key, {
        id: key,
        header: key,
        cell: (info) => {
          const v = info.getValue()
          return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v
        },
      }),
    )
  }, [data])

  const table = useReactTable({
    data,
    columns,
    state: { sorting, columnVisibility },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  // Scroll selected row into view
  useEffect(() => {
    if (selectedIndex == null || !tbodyRef.current) return
    const renderedRows = table.getRowModel().rows
    const renderedIdx = renderedRows.findIndex((r) => r.index === selectedIndex)
    if (renderedIdx >= 0) {
      tbodyRef.current.querySelectorAll('tr')[renderedIdx]?.scrollIntoView({
        block: 'nearest',
        behavior: 'smooth',
      })
    }
  }, [selectedIndex, table])

  if (!data.length) return <p className="text-sm text-gray-500 p-4">No data</p>

  const visibleCount = table.getVisibleFlatColumns().length
  const totalCount = table.getAllColumns().length

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b bg-gray-50 shrink-0">
        <span className="text-xs text-gray-500">
          {data.length} organisms · {visibleCount}/{totalCount} columns
        </span>
        <div className="relative" ref={colMenuRef}>
          <button
            onClick={() => setShowColMenu((v) => !v)}
            className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${
              showColMenu ? 'bg-blue-50 text-blue-700' : 'text-gray-500 hover:bg-gray-100'
            }`}
          >
            <Columns3 size={12} />
            Columns
          </button>
          {showColMenu && (
            <div className="absolute right-0 top-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 p-2 w-48 max-h-72 overflow-y-auto">
              <div className="flex gap-1 mb-2 pb-1.5 border-b">
                <button
                  onClick={() => setColumnVisibility(Object.fromEntries(table.getAllColumns().map((c) => [c.id, true])))}
                  className="flex-1 text-[11px] text-blue-600 hover:underline"
                >
                  All
                </button>
                <button
                  onClick={() =>
                    setColumnVisibility(
                      Object.fromEntries(table.getAllColumns().map((c) => [c.id, !DEFAULT_HIDDEN.has(c.id)])),
                    )
                  }
                  className="flex-1 text-[11px] text-gray-500 hover:underline"
                >
                  Default
                </button>
              </div>
              {table.getAllColumns().map((col) => (
                <label
                  key={col.id}
                  className="flex items-center gap-2 px-1 py-0.5 hover:bg-gray-50 rounded cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={col.getIsVisible()}
                    onChange={col.getToggleVisibilityHandler()}
                    className="rounded"
                  />
                  <span className="text-xs text-gray-700 truncate">{col.id}</span>
                </label>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto flex-1 min-h-0">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 sticky top-0 z-10">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((h) => (
                  <th
                    key={h.id}
                    className="px-3 py-2 text-left font-medium text-gray-600 cursor-pointer select-none whitespace-nowrap border-b"
                    onClick={h.column.getToggleSortingHandler()}
                  >
                    <span className="flex items-center gap-1">
                      {flexRender(h.column.columnDef.header, h.getContext())}
                      <ArrowUpDown size={12} className="text-gray-400 shrink-0" />
                    </span>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody ref={tbodyRef} className="divide-y">
            {table.getRowModel().rows.map((row) => {
              const isSelected = row.index === selectedIndex
              return (
                <tr
                  key={row.id}
                  onClick={() => onRowClick?.(row.index)}
                  className={`${
                    isSelected ? 'bg-amber-50' : 'hover:bg-gray-50'
                  } ${onRowClick ? 'cursor-pointer' : ''}`}
                >
                  {row.getVisibleCells().map((cell, ci) => (
                    <td
                      key={cell.id}
                      className={`px-3 py-1.5 whitespace-nowrap ${
                        isSelected
                          ? `text-amber-900 font-medium${ci === 0 ? ' border-l-2 border-amber-400' : ''}`
                          : ''
                      }`}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
