import { Edit3, Zap, CheckCircle2 } from 'lucide-react';

interface PathDecisionCardProps {
  detectionCount: number;
  onSelectPath: (path: 'annotate' | 'measure') => void;
}

export function PathDecisionCard({
  detectionCount,
  onSelectPath,
}: PathDecisionCardProps) {
  return (
    <div className="border border-blue-300 rounded-lg bg-blue-50 mb-4">
      {/* Header */}
      <div className="p-4 border-b border-blue-200">
        <div className="flex items-center gap-3 mb-2">
          <CheckCircle2 className="w-5 h-5 text-green-600" />
          <h3 className="text-lg font-semibold text-gray-900">Detection Complete!</h3>
        </div>
        <p className="text-sm text-gray-700">
          Found <strong>{detectionCount} organisms</strong>. What would you like to do next?
        </p>
      </div>

      {/* Two path options */}
      <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Path A: Edit Annotations */}
        <button
          onClick={() => onSelectPath('annotate')}
          className="flex flex-col items-start p-4 border-2 border-blue-300 rounded-lg bg-white hover:border-blue-500 hover:shadow-md transition-all text-left group"
        >
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center group-hover:bg-blue-200 transition-colors">
              <Edit3 className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h4 className="font-semibold text-gray-900">Edit Annotations</h4>
              <p className="text-xs text-gray-600">Review and refine</p>
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-3">
            Manually review detections, add missing organisms, remove false positives, and
            refine bounding boxes before measuring.
          </p>
          <div className="text-xs text-blue-700 font-medium">
            Recommended for maximum accuracy →
          </div>
        </button>

        {/* Path B: Measure Directly */}
        <button
          onClick={() => onSelectPath('measure')}
          className="flex flex-col items-start p-4 border-2 border-green-300 rounded-lg bg-white hover:border-green-500 hover:shadow-md transition-all text-left group"
        >
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center group-hover:bg-green-200 transition-colors">
              <Zap className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <h4 className="font-semibold text-gray-900">Measure Directly</h4>
              <p className="text-xs text-gray-600">Skip to results</p>
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-3">
            Trust the automatic detections and proceed directly to measurement. You can
            always come back to edit annotations later.
          </p>
          <div className="text-xs text-green-700 font-medium">
            Fastest path to results →
          </div>
        </button>
      </div>

      {/* Info footer */}
      <div className="px-4 pb-4">
        <div className="bg-blue-100 border border-blue-200 rounded-lg p-3">
          <p className="text-xs text-blue-800">
            <strong>Tip:</strong> If the detections look accurate, measuring directly is
            perfectly fine. You can always return to edit annotations and re-measure if needed.
          </p>
        </div>
      </div>
    </div>
  );
}
