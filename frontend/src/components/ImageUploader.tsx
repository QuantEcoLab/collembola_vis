import { useCallback, useState } from 'react'
import { Upload } from 'lucide-react'
import { uploadImage } from '../api/client'
import type { ImageInfo } from '../api/types'

interface Props {
  onUploaded: (info: ImageInfo) => void
}

export default function ImageUploader({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return
      setError(null)
      setUploading(true)
      try {
        const info = await uploadImage(files[0])
        onUploaded(info)
      } catch (e: any) {
        setError(e.message || 'Upload failed')
      } finally {
        setUploading(false)
      }
    },
    [onUploaded],
  )

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
        dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
      }`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        handleFiles(e.dataTransfer.files)
      }}
      onClick={() => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = 'image/*'
        input.onchange = () => handleFiles(input.files)
        input.click()
      }}
    >
      <Upload className="mx-auto mb-3 text-gray-400" size={32} />
      {uploading ? (
        <p className="text-sm text-gray-500">Uploading...</p>
      ) : (
        <>
          <p className="text-sm text-gray-600">
            Drag & drop an image here, or <span className="text-blue-600 underline">browse</span>
          </p>
          <p className="text-xs text-gray-400 mt-1">Supports JPG, PNG, TIFF</p>
        </>
      )}
      {error && <p className="text-sm text-red-600 mt-2">{error}</p>}
    </div>
  )
}
