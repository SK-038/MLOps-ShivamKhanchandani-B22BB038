import { FileText } from 'lucide-react';
import { useEffect, useState } from 'react';

interface ReadmeViewerProps {
  assignmentFolder: string;
}

export default function ReadmeViewer({ assignmentFolder }: ReadmeViewerProps) {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReadme = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`/${assignmentFolder}/README.md`);
        if (!response.ok) {
          throw new Error('README.md not found');
        }
        const text = await response.text();
        setContent(text);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load README');
      } finally {
        setLoading(false);
      }
    };

    fetchReadme();
  }, [assignmentFolder]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4 pb-4 border-b border-gray-200">
        <FileText className="w-5 h-5 text-gray-600" />
        <h2 className="text-lg font-semibold text-gray-900">README.md</h2>
      </div>
      <div className="prose prose-sm max-w-none">
        <pre className="whitespace-pre-wrap text-gray-700 font-mono text-sm bg-gray-50 p-4 rounded-lg">
          {content}
        </pre>
      </div>
    </div>
  );
}
