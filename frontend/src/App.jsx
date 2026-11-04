import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Analysis from "./pages/Analysis";
import Explainability from "./pages/Explainability";
import ModelLab from "./pages/ModelLab";
import Experiments from "./pages/Experiments";
import Datasets from "./pages/Datasets";
import About from "./pages/About";
import { EmptyState } from "./components/ui";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/explainability" element={<Explainability />} />
        <Route path="/models" element={<ModelLab />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<EmptyState title="Page not found">Use the menu to choose a section.</EmptyState>} />
      </Routes>
    </Layout>
  );
}
