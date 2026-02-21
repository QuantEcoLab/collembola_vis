import { useCallback, useState } from 'react'
import { Upload, FolderOpen } from 'lucide-react'
import { uploadImage, registerFromPath } from '../api/client'
import type { ImageInfo } from '../api/types'

interface Props {
  onUploaded: (info: ImageInfo) => void
}

export default function ImageUploader({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [error, setError] = useState<string | null>(null)

  // Server-path tab
  const [serverPath, setServerPath] = useState('')
  const [registeringPath, setRegisteringPath] = useState(false)
  const [tab, setTab] = useState<'upload' | 'server'>('upload')

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return
      setError(null)
      setUploading(true)
      setUploadPct(0)
      try {
        const info = await uploadImage(files[0], setUploadPct)
        onUploaded(info)
      } catch (e: any) {
        setError(e.message || 'Upload failed')
      } finally {
        setUploading(false)
        setUploadPct(0)
      }
    },
    [onUploaded],
  )

  const handleServerPath = async () => {
    if (!serverPath.trim()) return
    setError(null)
    setRegisteringPath(true)
    try {
      const info = await registerFromPath(serverPath.trim())
      onUploaded(info)
    } catch (e: any) {
      setError(e.message || 'Failed to load image')
    } finally {
      setRegisteringPath(false)
    }
  }

  return (
    <div className="space-y-2">
      {/* Tab switcher */}
      <div className="flex gap-1 text-sm">
        <button
          onClick={() => { setTab('upload'); setError(null) }}
          className={`px-3 py-1.5 rounded-lg border ${
            tab === 'upload'
              ? 'bg-blue-50 border-blue-300 text-blue-700'
              : 'bg-white border-gray-200 text-gray-500 hover:text-gray-700'
          }`}
        >
          Upload file
        </button>
        <button
          onClick={() => { setTab('server'); setError(null) }}
          className={`px-3 py-1.5 rounded-lg border ${
            tab === 'server'
              ? 'bg-blue-50 border-blue-300 text-blue-700'
              : 'bg-white border-gray-200 text-gray-500 hover:text-gray-700'
          }`}
        >
          Server path
        </button>
      </div>

      {tab === 'upload' ? (
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
            dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
          } ${uploading ? 'pointer-events-none' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            handleFiles(e.dataTransfer.files)
          }}
          onClick={() => {
            if (uploading) return
            const input = document.createElement('input')
            input.type = 'file'
            input.accept = 'image/*'
            input.onchange = () => handleFiles(input.files)
            input.click()
          }}
        >
          <Upload className="mx-auto mb-3 text-gray-400" size={32} />

          {uploading ? (
            <div className="space-y-3">
              <p className="text-sm text-gray-600">
                {uploadPct < 100 ? `Uploading… ${uploadPct}%` : 'Processing image…'}
              </p>
              <div className="w-full max-w-xs mx-auto bg-gray-200 rounded-full h-2.5 overflow-hidden">
                {uploadPct === 0 ? (
                  <div className="h-2.5 w-full bg-gradient-to-r from-blue-300 via-blue-500 to-blue-300 animate-pulse rounded-full" />
                ) : (
                  <div
                    className="bg-blue-500 h-2.5 rounded-full transition-all duration-200"
                    style={{ width: `${uploadPct}%` }}
                  />
                )}
              </div>
              {uploadPct === 100 && (
                <p className="text-xs text-gray-400">Generating thumbnail…</p>
              )}
            </div>
          ) : (
            <>
              <p className="text-sm text-gray-600">
                Drag & drop an image here, or <span className="text-blue-600 underline">browse</span>
              </p>
              <p className="text-xs text-gray-400 mt-1">Supports JPG, PNG, TIFF</p>
            </>
          )}
        </div>
      ) : (
        <div className="border rounded-lg p-6 space-y-3">
          <div className="flex items-center gap-2 text-gray-500">
            <FolderOpen size={20} />
            <span className="text-sm">Enter the absolute path to an image already on the server</span>
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={serverPath}
              onChange={(e) => setServerPath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleServerPath()}
              placeholder="/home/adeb/dev/collembola_vis/data/slike/image.jpg"
              className="flex-1 border rounded-lg px-3 py-2 text-sm font-mono"
              disabled={registeringPath}
            />
            <button
              onClick={handleServerPath}
              disabled={!serverPath.trim() || registeringPath}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
            >
              {registeringPath ? 'Loading…' : 'Load'}
            </button>
          </div>
          <p className="text-xs text-gray-400">
            The file stays in place — only a thumbnail is generated, no copy is made.
          </p>
        </div>
      )}

      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  )
}
