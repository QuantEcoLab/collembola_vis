import React, { useState } from 'react';
import { ModalDialog } from './ModalDialog';
import { Settings } from 'lucide-react';

interface DetectionConfig {
  tileSize: number;
  overlap: number;
  confidence: number;
  iouThreshold: number;
  device: string;
}

interface AdvancedDetectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (config: DetectionConfig) => void;
  initialConfig?: Partial<DetectionConfig>;
}

const DEFAULT_CONFIG: DetectionConfig = {
  tileSize: 1280,
  overlap: 256,
  confidence: 0.6,
  iouThreshold: 0.5,
  device: 'auto',
};

export function AdvancedDetectionModal({
  isOpen,
  onClose,
  onSubmit,
  initialConfig = {},
}: AdvancedDetectionModalProps) {
  const [config, setConfig] = useState<DetectionConfig>({
    ...DEFAULT_CONFIG,
    ...initialConfig,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    const newErrors: Record<string, string> = {};

    if (config.tileSize < 640 || config.tileSize > 2560) {
      newErrors.tileSize = 'Tile size must be between 640 and 2560';
    }

    if (config.overlap < 0 || config.overlap >= config.tileSize) {
      newErrors.overlap = `Overlap must be between 0 and ${config.tileSize - 1}`;
    }

    if (config.confidence < 0.1 || config.confidence > 1.0) {
      newErrors.confidence = 'Confidence must be between 0.1 and 1.0';
    }

    if (config.iouThreshold < 0.1 || config.iouThreshold > 1.0) {
      newErrors.iouThreshold = 'IoU threshold must be between 0.1 and 1.0';
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    onSubmit(config);
    onClose();
  };

  const handleReset = () => {
    setConfig(DEFAULT_CONFIG);
    setErrors({});
  };

  return (
    <ModalDialog
      isOpen={isOpen}
      onClose={onClose}
      title="Advanced Detection Settings"
      maxWidth="lg"
    >
      <form onSubmit={handleSubmit} className="p-6">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0">
              <Settings className="w-8 h-8 text-blue-600" />
            </div>
            <div className="flex-1">
              <p className="text-gray-700">
                Configure advanced parameters for the YOLO tiled detection pipeline.
                Default values work well for most images.
              </p>
            </div>
          </div>

          {/* Tile Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Tile Size (pixels)
            </label>
            <input
              type="number"
              value={config.tileSize}
              onChange={(e) =>
                setConfig({ ...config, tileSize: parseInt(e.target.value) })
              }
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                errors.tileSize ? 'border-red-500' : 'border-gray-300'
              }`}
              step="64"
              min="640"
              max="2560"
            />
            {errors.tileSize && (
              <p className="text-sm text-red-600 mt-1">{errors.tileSize}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Size of each tile for processing. Default: 1280px. Larger tiles = faster but uses
              more memory.
            </p>
          </div>

          {/* Overlap */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Overlap (pixels)
            </label>
            <input
              type="number"
              value={config.overlap}
              onChange={(e) =>
                setConfig({ ...config, overlap: parseInt(e.target.value) })
              }
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                errors.overlap ? 'border-red-500' : 'border-gray-300'
              }`}
              step="32"
              min="0"
              max={config.tileSize - 1}
            />
            {errors.overlap && (
              <p className="text-sm text-red-600 mt-1">{errors.overlap}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Overlap between adjacent tiles. Default: 256px. Helps detect organisms at tile
              boundaries.
            </p>
          </div>

          {/* Confidence Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Confidence Threshold
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                value={config.confidence}
                onChange={(e) =>
                  setConfig({ ...config, confidence: parseFloat(e.target.value) })
                }
                className="flex-1"
                step="0.05"
                min="0.1"
                max="1.0"
              />
              <span className="text-sm font-medium text-gray-700 w-12">
                {config.confidence.toFixed(2)}
              </span>
            </div>
            {errors.confidence && (
              <p className="text-sm text-red-600 mt-1">{errors.confidence}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Minimum confidence for detections. Default: 0.60. Lower = more detections but more
              false positives.
            </p>
          </div>

          {/* IoU Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              IoU Threshold (NMS)
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                value={config.iouThreshold}
                onChange={(e) =>
                  setConfig({ ...config, iouThreshold: parseFloat(e.target.value) })
                }
                className="flex-1"
                step="0.05"
                min="0.1"
                max="1.0"
              />
              <span className="text-sm font-medium text-gray-700 w-12">
                {config.iouThreshold.toFixed(2)}
              </span>
            </div>
            {errors.iouThreshold && (
              <p className="text-sm text-red-600 mt-1">{errors.iouThreshold}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Non-Maximum Suppression threshold. Default: 0.50. Lower = fewer duplicate
              detections.
            </p>
          </div>

          {/* Device */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Processing Device
            </label>
            <select
              value={config.device}
              onChange={(e) => setConfig({ ...config, device: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="auto">Auto (recommended)</option>
              <option value="cuda">CUDA (GPU)</option>
              <option value="cpu">CPU</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Hardware device for inference. 'Auto' will use GPU if available, otherwise CPU.
            </p>
          </div>

          {/* Info box */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2">Performance Tips:</h4>
            <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
              <li>Larger tile sizes process faster but use more memory</li>
              <li>More overlap catches edge cases but increases processing time</li>
              <li>Higher confidence reduces false positives but may miss organisms</li>
            </ul>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex justify-between mt-6">
          <button
            type="button"
            onClick={handleReset}
            className="px-4 py-2 text-gray-700 hover:text-gray-900 font-medium transition-colors"
          >
            Reset to Defaults
          </button>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium transition-colors"
            >
              Apply Settings
            </button>
          </div>
        </div>
      </form>
    </ModalDialog>
  );
}
