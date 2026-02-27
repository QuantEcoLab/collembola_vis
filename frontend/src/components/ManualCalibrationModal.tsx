import React, { useState } from 'react';
import { ModalDialog } from './ModalDialog';
import { Ruler, MousePointer2, Calculator } from 'lucide-react';

interface Point {
  x: number;
  y: number;
}

interface ManualCalibrationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: (umPerPixel: number) => void;
  imageWidth?: number;
  imageHeight?: number;
}

export function ManualCalibrationModal({
  isOpen,
  onClose,
  onComplete,
}: ManualCalibrationModalProps) {
  const [wizardStep, setWizardStep] = useState<1 | 2 | 3>(1);
  const [point1, setPoint1] = useState<Point | null>(null);
  const [point2, setPoint2] = useState<Point | null>(null);
  const [knownDistance, setKnownDistance] = useState<string>('');
  const [isListening, setIsListening] = useState(false);

  const resetState = () => {
    setWizardStep(1);
    setPoint1(null);
    setPoint2(null);
    setKnownDistance('');
    setIsListening(false);
  };

  const handleClose = () => {
    resetState();
    onClose();
  };

  const handleStartPointSelection = () => {
    setWizardStep(2);
    setIsListening(true);
    // Emit event to tell WorkspacePage to start listening for clicks
    window.dispatchEvent(new CustomEvent('start-calibration-point-selection'));
  };

  const handleCancelPointSelection = () => {
    setWizardStep(1);
    setPoint1(null);
    setPoint2(null);
    setIsListening(false);
    window.dispatchEvent(new CustomEvent('cancel-calibration-point-selection'));
  };

  const handlePointsComplete = () => {
    setWizardStep(3);
    setIsListening(false);
    window.dispatchEvent(new CustomEvent('complete-calibration-point-selection'));
  };

  const handleCalculate = () => {
    if (!point1 || !point2 || !knownDistance) return;

    const pixelDistance = Math.sqrt(
      Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2)
    );

    const umPerPixel = parseFloat(knownDistance) / pixelDistance;
    onComplete(umPerPixel);
    handleClose();
  };

  // Listen for point events from WorkspacePage
  React.useEffect(() => {
    if (!isOpen || !isListening) return;

    const handlePoint = (e: Event) => {
      const customEvent = e as CustomEvent<Point>;
      const point = customEvent.detail;

      if (!point1) {
        setPoint1(point);
      } else if (!point2) {
        setPoint2(point);
      }
    };

    window.addEventListener('calibration-point-selected', handlePoint);
    return () => window.removeEventListener('calibration-point-selected', handlePoint);
  }, [isOpen, isListening, point1, point2]);

  // Auto-advance to step 3 when both points are selected
  React.useEffect(() => {
    if (point1 && point2 && wizardStep === 2) {
      handlePointsComplete();
    }
  }, [point1, point2, wizardStep]);

  const pixelDistance =
    point1 && point2
      ? Math.sqrt(
          Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2)
        ).toFixed(1)
      : null;

  const calculatedUmPerPixel =
    pixelDistance && knownDistance
      ? (parseFloat(knownDistance) / parseFloat(pixelDistance)).toFixed(3)
      : null;

  return (
    <ModalDialog
      isOpen={isOpen}
      onClose={handleClose}
      title="Manual Calibration"
      maxWidth="lg"
      showCloseButton={wizardStep === 1}
    >
      <div className="p-6">
        {/* Step 1: Instructions */}
        {wizardStep === 1 && (
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                <Ruler className="w-10 h-10 text-blue-600" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Manual Scale Calibration
                </h3>
                <p className="text-gray-700 mb-4">
                  Use this method when you have a known reference distance in your image
                  (e.g., a scale bar).
                </p>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold text-blue-900 mb-2">How it works:</h4>
                  <ol className="list-decimal list-inside space-y-2 text-sm text-blue-800">
                    <li>Click two points on the image at a known distance apart</li>
                    <li>Enter the known distance between those points (in micrometers)</li>
                    <li>We'll calculate the scale (μm per pixel) automatically</li>
                  </ol>
                </div>

                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                  <p className="text-sm text-amber-800">
                    <strong>Tip:</strong> For best accuracy, use two points that are far apart
                    (e.g., endpoints of a scale bar).
                  </p>
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={handleClose}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleStartPointSelection}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium transition-colors flex items-center gap-2"
              >
                <MousePointer2 className="w-4 h-4" />
                Start Point Selection
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Point Selection */}
        {wizardStep === 2 && (
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                <MousePointer2 className="w-10 h-10 text-blue-600" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Click Two Points on the Image
                </h3>
                <p className="text-gray-700 mb-4">
                  Click on the image viewer to select two points at a known distance apart.
                </p>

                {/* Point status */}
                <div className="space-y-2 mb-4">
                  <div
                    className={`flex items-center gap-3 p-3 rounded-lg ${
                      point1
                        ? 'bg-green-50 border border-green-200'
                        : 'bg-gray-50 border border-gray-200'
                    }`}
                  >
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center ${
                        point1 ? 'bg-green-600' : 'bg-gray-400'
                      }`}
                    >
                      <span className="text-white text-sm font-bold">1</span>
                    </div>
                    <span className="text-sm font-medium">
                      {point1 ? `Point 1: (${point1.x}, ${point1.y})` : 'Click first point'}
                    </span>
                  </div>

                  <div
                    className={`flex items-center gap-3 p-3 rounded-lg ${
                      point2
                        ? 'bg-green-50 border border-green-200'
                        : 'bg-gray-50 border border-gray-200'
                    }`}
                  >
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center ${
                        point2 ? 'bg-green-600' : 'bg-gray-400'
                      }`}
                    >
                      <span className="text-white text-sm font-bold">2</span>
                    </div>
                    <span className="text-sm font-medium">
                      {point2 ? `Point 2: (${point2.x}, ${point2.y})` : 'Click second point'}
                    </span>
                  </div>
                </div>

                {pixelDistance && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <p className="text-sm text-blue-900">
                      <strong>Distance:</strong> {pixelDistance} pixels
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={handleCancelPointSelection}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Enter Known Distance */}
        {wizardStep === 3 && (
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                <Calculator className="w-10 h-10 text-blue-600" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Enter Known Distance
                </h3>
                <p className="text-gray-700 mb-4">
                  What is the known distance between the two points you selected?
                </p>

                {/* Summary of points */}
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Point 1:</span>{' '}
                      <span className="font-medium">
                        ({point1?.x}, {point1?.y})
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Point 2:</span>{' '}
                      <span className="font-medium">
                        ({point2?.x}, {point2?.y})
                      </span>
                    </div>
                    <div className="col-span-2">
                      <span className="text-gray-600">Pixel Distance:</span>{' '}
                      <span className="font-medium">{pixelDistance} px</span>
                    </div>
                  </div>
                </div>

                {/* Input for known distance */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Known Distance (μm)
                  </label>
                  <input
                    type="number"
                    value={knownDistance}
                    onChange={(e) => setKnownDistance(e.target.value)}
                    placeholder="e.g., 1000"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    step="0.01"
                    min="0"
                    autoFocus
                  />
                </div>

                {/* Calculated result */}
                {calculatedUmPerPixel && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <p className="text-sm text-green-900 mb-1">
                      <strong>Calculated Scale:</strong>
                    </p>
                    <p className="text-2xl font-bold text-green-700">
                      {calculatedUmPerPixel} μm/pixel
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => {
                  setWizardStep(2);
                  setPoint1(null);
                  setPoint2(null);
                  setIsListening(true);
                  window.dispatchEvent(new CustomEvent('start-calibration-point-selection'));
                }}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors"
              >
                Re-select Points
              </button>
              <button
                onClick={handleCalculate}
                disabled={!knownDistance || parseFloat(knownDistance) <= 0}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                Apply Calibration
              </button>
            </div>
          </div>
        )}
      </div>
    </ModalDialog>
  );
}
