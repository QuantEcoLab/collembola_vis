import { useState } from 'react'
import ImageUploader from '../components/ImageUploader'
import ImageViewer from '../components/ImageViewer'
import DetectionParams, { type DetectionParamsType } from '../components/DetectionParams'
import JobProgress from '../components/JobProgress'
import { runDetection, thumbnailUrl, outputFileUrl } from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import type { ImageInfo } from '../api/types'
import { ScanSearch } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export default function DetectPage() {
  const [image, setImage] = useState<ImageInfo | null>(null)
  const [params, setParams] = useState<DetectionParamsType>({
    conf: 0.6,
    iou: 0.5,
    tileSize: 1280,
    overlap: 256,
  })
  const [jobId, setJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const job = useJobProgress(jobId)
  const navigate = useNavigate()

  const handleRun = async () => {
    if (!image) return
    setError(null)
    try {
      const res = await runDetection({
        image_id: image.image_id,
        conf: params.conf,
        iou: params.iou,
        tile_size: params.tileSize,
        overlap: params.overlap,
      })
      setJobId(res.job_id)
    } catch (e: any) {
      setError(e.message)
    }
  }

  const overlayAvailable = job?.status === 'completed' && job.result?.image_stem

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Detection</h2>
        <p className="text-sm text-gray-500 mt-1">
          Run tiled YOLO inference to detect collembola organisms
        </p>
      </div>

      {!image && <ImageUploader onUploaded={setImage} />}

      {image && (
        <>
          <div className="h-[400px]">
            <ImageViewer
              src={
                overlayAvailable
                  ? outputFileUrl(job!.id, `${job!.result.image_stem}_overlay.jpg`)
                  : thumbnailUrl(image.image_id)
              }
              alt={image.filename}
              className="h-full"
            />
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="font-medium text-gray-900">Parameters</h3>
              <DetectionParams params={params} onChange={setParams} />
            </div>

            <div className="space-y-4">
              <h3 className="font-medium text-gray-900">Run</h3>
              <button
                onClick={handleRun}
                disabled={!!jobId && job?.status === 'running'}
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium"
              >
                <ScanSearch size={16} />
                Run Detection
              </button>

              {error && <p className="text-sm text-red-600">{error}</p>}

              <JobProgress job={job} />

              {job?.status === 'completed' && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 space-y-2">
                  <p className="font-medium text-green-800">
                    Detected {job.result.num_detections} organisms
                  </p>
                  <button
                    onClick={() =>
                      navigate('/measure', {
                        state: {
                          imageId: image.image_id,
                          imagePath: job.result.image_path,
                          detectionsCSV: job.result.csv_path,
                        },
                      })
                    }
                    className="text-sm text-blue-600 hover:underline"
                  >
                    Measure these detections →
                  </button>
                </div>
              )}

              <button
                onClick={() => {
                  setImage(null)
                  setJobId(null)
                }}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Change image
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
