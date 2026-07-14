import { useEffect } from "react";

// Reusable model dropdown driven by the backend catalog, with a "Custom…" escape so any
// model id the provider serves can still be entered.
export default function ModelPicker({
  models,
  value,
  onChange,
  id = "model",
}: {
  models: string[];
  value: string;
  onChange: (v: string) => void;
  id?: string;
}) {
  const isCustom = value !== "" && !models.includes(value);

  // If the catalog changes (e.g. provider switched) and the value no longer fits, snap to
  // the first option so the field is never left in an invalid state.
  useEffect(() => {
    if (models.length && value === "") onChange(models[0]);
  }, [models]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="model-picker">
      <select
        id={id}
        value={isCustom ? "__custom__" : value}
        onChange={(e) => {
          const v = e.target.value;
          if (v === "__custom__") onChange(""); // switch to manual entry
          else onChange(v);
        }}
      >
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
        <option value="__custom__">Custom…</option>
      </select>
      {(isCustom || value === "") && (
        <input
          aria-label="Custom model id"
          placeholder="type a model id"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          style={{ marginTop: 8 }}
        />
      )}
    </div>
  );
}
