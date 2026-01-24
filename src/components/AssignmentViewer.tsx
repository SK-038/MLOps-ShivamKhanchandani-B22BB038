import { BookOpen } from 'lucide-react';
import { Assignment } from '../types';
import ReadmeViewer from './ReadmeViewer';

interface AssignmentViewerProps {
  assignment: Assignment;
}

export default function AssignmentViewer({ assignment }: AssignmentViewerProps) {
  return (
    <div className="flex-1 overflow-y-auto bg-gray-50">
      <div className="max-w-5xl mx-auto p-8">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <BookOpen className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-900">{assignment.name}</h1>
          </div>
          <p className="text-gray-500">{assignment.folder}</p>
        </div>

        <ReadmeViewer assignmentFolder={assignment.folder} />
      </div>
    </div>
  );
}
