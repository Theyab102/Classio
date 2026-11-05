export function checkBirthDate(value, today = new Date()) {
  const parts = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!parts) return null;

  const [, year, month, day] = parts.map(Number);
  const birth = new Date(year, month - 1, day);
  if (year < 1900 || birth.getFullYear() !== year || birth.getMonth() !== month - 1 ||
      birth.getDate() !== day || birth > today) return null;

  const birthdayPassed = today.getMonth() + 1 > month ||
    (today.getMonth() + 1 === month && today.getDate() >= day);
  const age = today.getFullYear() - year - (birthdayPassed ? 0 : 1);
  return age >= 13;
}
