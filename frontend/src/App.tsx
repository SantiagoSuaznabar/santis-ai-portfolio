import { Routes, Route } from 'react-router-dom';
import Rag from './features/rag/rag.tsx';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Rag />} />
    </Routes>
  );
}