import { useId } from 'react';
import UiIcon from './UiIcon';

export default function CountInput({ label, value, onChange, min = 1, max = 50, disabled = false }) {
  const id = useId();
  const change = delta => {
    const current = value === '' || !Number.isFinite(Number(value)) ? min : Number(value);
    onChange(String(Math.max(min, Math.min(max, current + delta))));
  };
  return <div className="study-count-field">
    <label htmlFor={id}>{label}</label>
    <div className="study-count-control">
      <button type="button" aria-label={`Decrease ${label.toLowerCase()}`} disabled={disabled || Number(value) <= min} onClick={() => change(-1)}><span aria-hidden="true">−</span></button>
      <input id={id} type="number" aria-label={label} aria-describedby={`${id}-hint`} min={min} max={max} step="1" inputMode="numeric" value={value} disabled={disabled} onChange={event => onChange(event.target.value)} />
      <UiIcon name="edit" size={16} className="study-count-edit" />
      <button type="button" aria-label={`Increase ${label.toLowerCase()}`} disabled={disabled || Number(value) >= max} onClick={() => change(1)}><span aria-hidden="true">+</span></button>
    </div>
    <small id={`${id}-hint`}>Type a number or use − / + · {min}–{max}</small>
  </div>;
}
