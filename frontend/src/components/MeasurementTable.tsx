import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from '@tanstack/react-table'
import { useState, useMemo } from 'react'
import { ArrowUpDown } from 'lucide-react'

interface MeasurementRow {
  [key: string]: number | string
}

interface Props {
  data: MeasurementRow[]
}

export default function MeasurementTable({ data }: Props) {
  const [sorting, setSorting] = useState<SortingState>([])

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
        <tbody className="divide-y">
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="hover:bg-gray-50">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-3 py-1.5 whitespace-nowrap">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
