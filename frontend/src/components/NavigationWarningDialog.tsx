import { AlertTriangle } from 'lucide-react';
import { ModalDialog } from './ModalDialog';

interface NavigationWarningDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  targetStep: number;
  hasDetections: boolean;
  hasMeasurements: boolean;
}

export function NavigationWarningDialog({
  isOpen,
  onClose,
  onConfirm,
  targetStep,
  hasDetections,
  hasMeasurements,
}: NavigationWarningDialogProps) {
  // Determine what will be cleared
  const clearingDetections = targetStep <= 2 && hasDetections;
  const clearingMeasurements = targetStep <= 3 && hasMeasurements;

  const dataLoss: string[] = [];
  if (clearingDetections) {
    dataLoss.push('All detection results');
    dataLoss.push('All annotation edits');
  }
  if (clearingMeasurements) {
    dataLoss.push('All measurements');
  }

  if (dataLoss.length === 0) {
    // No data loss, no need for warning
    onConfirm();
    return null;
  }

  return (
    <ModalDialog
      isOpen={isOpen}
      onClose={onClose}
      title="Warning: Data Will Be Cleared"
      maxWidth="md"
    >
      <div className="p-6">
        {/* Warning icon and message */}
        <div className="flex items-start gap-4 mb-6">
          <div className="flex-shrink-0">
            <AlertTriangle className="w-12 h-12 text-amber-500" />
          </div>
          <div className="flex-1">
            <p className="text-gray-700 mb-3">
              Going back to <strong>Step {targetStep}</strong> will clear the following data:
            </p>
            <ul className="list-disc list-inside space-y-1 text-gray-700">
              {dataLoss.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
            <p className="text-gray-700 mt-3">
              This action cannot be undone. Are you sure you want to continue?
            </p>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              onConfirm();
              onClose();
            }}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium transition-colors"
          >
            Continue & Clear Data
          </button>
        </div>
      </div>
    </ModalDialog>
  );
}
