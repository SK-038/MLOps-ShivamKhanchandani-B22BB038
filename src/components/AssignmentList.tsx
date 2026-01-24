import { FolderOpen } from 'lucide-react';
import { Assignment } from '../types';

interface AssignmentListProps {
  assignments: Assignment[];
  onSelectAssignment: (assignment: Assignment) => void;
  selectedAssignment: Assignment | null;
}

export default function AssignmentList({
  assignments,
  onSelectAssignment,
  selectedAssignment
}: AssignmentListProps) {
  return (
    <div className="w-80 bg-white border-r border-gray-200 overflow-y-auto">
      <div className="p-6 border-b border-gray-200">
        <h1 className="text-2xl font-bold text-gray-900">Course Assignments</h1>
        <p className="text-sm text-gray-500 mt-1">{assignments.length} assignments</p>
      </div>
      <div className="p-4 space-y-2">
        {assignments.map((assignment) => (
          <button
            key={assignment.id}
            onClick={() => onSelectAssignment(assignment)}
            className={`w-full text-left p-4 rounded-lg transition-all ${
              selectedAssignment?.id === assignment.id
                ? 'bg-blue-50 border-2 border-blue-500 shadow-sm'
                : 'bg-gray-50 border-2 border-transparent hover:bg-gray-100 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-3">
              <FolderOpen
                className={`w-5 h-5 ${
                  selectedAssignment?.id === assignment.id ? 'text-blue-600' : 'text-gray-400'
                }`}
              />
              <div>
                <h3 className={`font-semibold ${
                  selectedAssignment?.id === assignment.id ? 'text-blue-900' : 'text-gray-900'
                }`}>
                  {assignment.name}
                </h3>
                <p className="text-xs text-gray-500">{assignment.folder}</p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
