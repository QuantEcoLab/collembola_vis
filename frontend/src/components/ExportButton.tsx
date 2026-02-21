import { Download } from 'lucide-react'

interface Props {
  url: string
  filename?: string
  label?: string
}

export default function ExportButton({ url, filename, label = 'Export CSV' }: Props) {
  return (
    <a
      href={url}
      download={filename}
      className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
    >
      <Download size={16} />
      {label}
    </a>
  )
}
