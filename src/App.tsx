import { useState, useEffect } from 'react';
import AssignmentList from './components/AssignmentList';
import AssignmentViewer from './components/AssignmentViewer';
import { Assignment } from './types';
import { GraduationCap } from 'lucide-react';

function App() {
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [selectedAssignment, setSelectedAssignment] = useState<Assignment | null>(null);

  useEffect(() => {
    const loadAssignments = async () => {
      try {
        const response = await fetch('/assignments.json');
        if (response.ok) {
          const data = await response.json();
          setAssignments(data);
          if (data.length > 0) {
            setSelectedAssignment(data[0]);
          }
        }
      } catch (error) {
        console.error('Failed to load assignments:', error);
      }
    };

    loadAssignments();
  }, []);

  if (assignments.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-sky-100 flex items-center justify-center p-8">
        <div className="bg-white rounded-2xl shadow-xl p-12 max-w-2xl text-center">
          <GraduationCap className="w-16 h-16 text-blue-600 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-gray-900 mb-4">Course Assignments Viewer</h1>
          <p className="text-gray-600 mb-6">
            To get started, create an <code className="bg-gray-100 px-2 py-1 rounded text-sm">assignments.json</code> file in your public folder:
          </p>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-left text-sm overflow-x-auto">
{`[
  {
    "id": "assignment-1",
    "name": "Assignment 1",
    "folder": "assignment-1"
  },
  {
    "id": "assignment-2",
    "name": "Assignment 2",
    "folder": "assignment-2"
  }
]`}
          </pre>
          <p className="text-gray-600 mt-6 text-sm">
            Then add folders like <code className="bg-gray-100 px-2 py-1 rounded">assignment-1/</code> with a
            <code className="bg-gray-100 px-2 py-1 rounded ml-1">README.md</code> file inside.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <AssignmentList
        assignments={assignments}
        onSelectAssignment={setSelectedAssignment}
        selectedAssignment={selectedAssignment}
      />
      {selectedAssignment ? (
        <AssignmentViewer assignment={selectedAssignment} />
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-gray-500">Select an assignment to view</p>
        </div>
      )}
    </div>
  );
}

export default App;
