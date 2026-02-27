import React, { useState } from 'react';
import { ModalDialog } from './ModalDialog';
import { Zap, AlertCircle } from 'lucide-react';

interface FineTuneConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  patience: number;
  device: string;
}

interface FineTuneModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (config: FineTuneConfig) => void;
  detectionJobId?: string;
}

const DEFAULT_CONFIG: FineTuneConfig = {
  epochs: 50,
  batchSize: 16,
  learningRate: 0.001,
  patience: 10,
  device: 'auto',
};

export function FineTuneModal({
  isOpen,
  onClose,
  onSubmit,
  detectionJobId,
}: FineTuneModalProps) {
  const [config, setConfig] = useState<FineTuneConfig>(DEFAULT_CONFIG);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    const newErrors: Record<string, string> = {};

    if (config.epochs < 1 || config.epochs > 1000) {
      newErrors.epochs = 'Epochs must be between 1 and 1000';
    }

    if (config.batchSize < 1 || config.batchSize > 128) {
      newErrors.batchSize = 'Batch size must be between 1 and 128';
    }

    if (config.learningRate <= 0 || config.learningRate > 1) {
      newErrors.learningRate = 'Learning rate must be between 0 and 1';
    }

    if (config.patience < 1 || config.patience > 100) {
      newErrors.patience = 'Patience must be between 1 and 100';
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
      title="Fine-Tune Model"
      maxWidth="xl"
    >
      <form onSubmit={handleSubmit} className="p-6">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0">
              <Zap className="w-8 h-8 text-amber-600" />
            </div>
            <div className="flex-1">
              <p className="text-gray-700">
                Fine-tune the YOLO model using your manually edited annotations. This will
                create a new model checkpoint trained specifically on your corrections.
              </p>
            </div>
          </div>

          {/* Warning box */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-amber-800 space-y-2">
                <p>
                  <strong>Important:</strong> Fine-tuning requires manually edited annotations
                  with both accepted and rejected boxes.
                </p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>Training will use your accepted and added boxes as positive examples</li>
                  <li>Rejected boxes help the model learn what NOT to detect</li>
                  <li>The process may take 10-30 minutes depending on dataset size</li>
                  <li>You can monitor progress in real-time via the job queue</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Current job info */}
          {detectionJobId && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-sm text-blue-900">
                <strong>Training from:</strong> Detection job {detectionJobId}
              </p>
            </div>
          )}

          {/* Epochs */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Epochs
            </label>
            <input
              type="number"
              value={config.epochs}
              onChange={(e) =>
                setConfig({ ...config, epochs: parseInt(e.target.value) })
              }
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                errors.epochs ? 'border-red-500' : 'border-gray-300'
              }`}
              min="1"
              max="1000"
            />
            {errors.epochs && (
              <p className="text-sm text-red-600 mt-1">{errors.epochs}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Number of training epochs. Default: 50. More epochs = longer training but potentially better results.
            </p>
          </div>

          {/* Batch Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Batch Size
            </label>
            <input
              type="number"
              value={config.batchSize}
              onChange={(e) =>
                setConfig({ ...config, batchSize: parseInt(e.target.value) })
              }
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                errors.batchSize ? 'border-red-500' : 'border-gray-300'
              }`}
              min="1"
              max="128"
            />
            {errors.batchSize && (
              <p className="text-sm text-red-600 mt-1">{errors.batchSize}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Number of samples per batch. Default: 16. Larger batch = faster training but uses more memory.
            </p>
          </div>

          {/* Learning Rate */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Learning Rate
            </label>
            <input
              type="number"
              value={config.learningRate}
              onChange={(e) =>
                setConfig({ ...config, learningRate: parseFloat(e.target.value) })
              }
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                errors.learningRate ? 'border-red-500' : 'border-gray-300'
              }`}
              step="0.0001"
              min="0.0001"
              max="1"
            />
            {errors.learningRate && (
              <p className="text-sm text-red-600 mt-1">{errors.learningRate}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Initial learning rate for optimizer. Default: 0.001. Lower = slower but more stable training.
            </p>
          </div>

          {/* Patience */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Early Stopping Patience
            </label>
            <input
              type="number"
              value={config.patience}
              onChange={(e) =>
                setConfig({ ...config, patience: parseInt(e.target.value) })
              }
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                errors.patience ? 'border-red-500' : 'border-gray-300'
              }`}
              min="1"
              max="100"
            />
            {errors.patience && (
              <p className="text-sm text-red-600 mt-1">{errors.patience}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Stop training if no improvement after N epochs. Default: 10. Prevents overfitting.
            </p>
          </div>

          {/* Device */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Training Device
            </label>
            <select
              value={config.device}
              onChange={(e) => setConfig({ ...config, device: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="auto">Auto (recommended)</option>
              <option value="cuda">CUDA (GPU)</option>
              <option value="cpu">CPU (slow)</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Hardware device for training. GPU highly recommended for reasonable training times.
            </p>
          </div>

          {/* Info box */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 className="font-semibold text-green-900 mb-2">What happens next:</h4>
            <ol className="text-sm text-green-800 space-y-1 list-decimal list-inside">
              <li>Your annotations are converted to YOLO training format</li>
              <li>The model trains on your corrections for the specified epochs</li>
              <li>The best checkpoint is saved when validation loss improves</li>
              <li>You can download the fine-tuned model when training completes</li>
            </ol>
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
              className="px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 font-medium transition-colors"
            >
              Start Fine-Tuning
            </button>
          </div>
        </div>
      </form>
    </ModalDialog>
  );
}
