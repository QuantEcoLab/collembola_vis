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
import { formatVolume } from './VolumeSummary'

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
  [key: string]: number | string | boolean | null
}

const EXP_HEADERS: Record<string, string> = {
  detection_id: 'Detection ID', segmentation_confidence: 'Seg. confidence', volume_mm3: 'Procijenjeni volumen (mm³)',
  volume_unit: 'Jedinica volumena', length_mm: 'Duljina (mm)', width_mm: 'Najveća širina elipse (mm)',
  length_px: 'Duljina (px)', width_px: 'Najveća širina elipse (px)', area_px: 'Površina (px²)', area_mm2: 'Površina (mm²)',
  perimeter_px: 'Opseg (px)', perimeter_mm: 'Opseg (mm)', calibration_um_per_px: 'Kalibracija (µm/px)',
  dedup_status: 'Dedup', touches_crop_edge: 'Dodir ruba cropa', warning: 'Upozorenje',
}

function defaultVisibility(data: MeasurementRow[]): VisibilityState {
  if (!data.length) return {}
  const experimental = data[0].model_name === 'exp03_no_os'
  if (!experimental) return Object.fromEntries(Object.keys(data[0]).map(k => [k, !DEFAULT_HIDDEN.has(k)]))
  const calibrated = typeof data[0].calibration_um_per_px === 'number'
  const metrics = calibrated ? ['length_mm', 'width_mm', 'area_mm2', 'perimeter_mm'] : ['length_px', 'width_px', 'area_px', 'perimeter_px']
  const shown = new Set(['detection_id', 'segmentation_confidence', 'volume_mm3', ...metrics,
    'calibration_um_per_px', 'dedup_status', 'touches_crop_edge', 'warning'])
  return Object.fromEntries(Object.keys(data[0]).map(k => [k, shown.has(k)]))
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
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(() => defaultVisibility(data))
  const experimental = data[0]?.model_name === 'exp03_no_os'
  const schemaKey = Object.keys(data[0] ?? {}).join('|') + String(experimental) + String(typeof data[0]?.calibration_um_per_px)
  const previousSchema = useRef(schemaKey)
  useEffect(() => {
    if (previousSchema.current !== schemaKey) {
      previousSchema.current = schemaKey
      setColumnVisibility(defaultVisibility(data))
    }
  }, [schemaKey, data])

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
        header: experimental ? EXP_HEADERS[key] ?? key : key,
        cell: (info) => {
          const v = info.getValue()
          if (experimental && key === 'volume_mm3') {
            const reason = info.row.original.volume_unit
            return typeof v === 'number' ? formatVolume(v) : typeof reason === 'string' && reason.startsWith('nije') ? reason : 'nije dostupan bez kalibracije'
          }
          if (v == null) return '—'
          if (typeof v === 'boolean') return v ? 'DA' : 'NE'
          return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v
        },
      }),
    )
  }, [data, experimental])

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
                      defaultVisibility(data),
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
                    className={`px-3 py-2 text-left font-medium cursor-pointer select-none whitespace-nowrap border-b ${experimental && h.column.id === 'volume_mm3' ? 'bg-violet-100 text-violet-950 font-bold' : 'text-gray-600'}`}
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
                      className={`px-3 py-1.5 whitespace-nowrap ${experimental && cell.column.id === 'volume_mm3' ? 'bg-violet-50 text-violet-950 font-bold text-base ' : ''}${
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
