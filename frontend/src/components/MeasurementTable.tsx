import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from '@tanstack/react-table'
import { useState, useMemo, useEffect, useRef } from 'react'
import { ArrowUpDown } from 'lucide-react'

interface MeasurementRow {
  [key: string]: number | string
}

interface Props {
  data: MeasurementRow[]
  selectedIndex?: number | null
}

export default function MeasurementTable({ data, selectedIndex }: Props) {
  const [sorting, setSorting] = useState<SortingState>([])
  const tbodyRef = useRef<HTMLTableSectionElement>(null)

  const columns = useMemo(() => {
    if (!data.length) return []
    const helper = createColumnHelper<MeasurementRow>()
    return Object.keys(data[0]).map((key) =>
      helper.accessor(key, {
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
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  // Scroll selected row into view whenever selectedIndex changes
  useEffect(() => {
    if (selectedIndex == null || !tbodyRef.current) return
    const rows = tbodyRef.current.querySelectorAll('tr')
    // Find the rendered row whose original data index matches selectedIndex
    const renderedRows = table.getRowModel().rows
    const renderedIdx = renderedRows.findIndex((r) => r.index === selectedIndex)
    if (renderedIdx >= 0 && rows[renderedIdx]) {
      rows[renderedIdx].scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [selectedIndex, table])

  if (!data.length) return <p className="text-sm text-gray-500">No data</p>

  return (
    <div className="overflow-auto max-h-[500px] border rounded-lg">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50 sticky top-0">
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => (
                <th
                  key={h.id}
                  className="px-3 py-2 text-left font-medium text-gray-600 cursor-pointer select-none whitespace-nowrap"
                  onClick={h.column.getToggleSortingHandler()}
                >
                  <span className="flex items-center gap-1">
                    {flexRender(h.column.columnDef.header, h.getContext())}
                    <ArrowUpDown size={12} className="text-gray-400" />
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
              <tr key={row.id} className={isSelected ? '' : 'hover:bg-gray-50'}>
                {row.getVisibleCells().map((cell, ci) => (
                  <td
                    key={cell.id}
                    className={`px-3 py-1.5 whitespace-nowrap ${
                      isSelected
                        ? `bg-amber-100 text-amber-900 font-medium${ci === 0 ? ' border-l-2 border-amber-400' : ''}`
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
  )
}
