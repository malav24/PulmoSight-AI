import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "../services/api";

const AppContext = createContext(null);

function readStored(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}
function writeStored(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable - selection just isn't remembered */
  }
}

export function AppProvider({ children }) {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [models, setModels] = useState(null);
  const [modelsError, setModelsError] = useState(null);
  const [selectedModel, setSelectedModelState] = useState(readStored("rfrr.model"));
  const [lastFile, setLastFile] = useState(null); // shared between Analysis and Explainability
  const [lastAnalysis, setLastAnalysis] = useState(null);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (e) {
      setHealthError(e);
    }
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      const m = await api.models();
      setModels(m);
      setModelsError(null);
      return m;
    } catch (e) {
      setModelsError(e);
      return null;
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    refreshModels();
    const t = setInterval(refreshHealth, 30000);
    return () => clearInterval(t);
  }, [refreshHealth, refreshModels]);

  const trainedModels = useMemo(() => (models?.models || []).filter((m) => m.trained), [models]);

  // Selected model must be a trained one; otherwise fall back to the server's active / first trained model.
  const effectiveModel = useMemo(() => {
    const keys = trainedModels.map((m) => m.key);
    if (selectedModel && keys.includes(selectedModel)) return selectedModel;
    if (models?.active && keys.includes(models.active)) return models.active;
    if (keys.includes("model_b")) return "model_b";
    return keys[0] || null;
  }, [trainedModels, selectedModel, models]);

  const setSelectedModel = useCallback((key) => {
    setSelectedModelState(key);
    writeStored("rfrr.model", key);
  }, []);

  const value = {
    health, healthError, refreshHealth,
    models, modelsError, refreshModels, trainedModels,
    selectedModel: effectiveModel, setSelectedModel,
    lastFile, setLastFile, lastAnalysis, setLastAnalysis,
    apiReachable: !!health && !healthError,
  };
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export const useApp = () => useContext(AppContext);
