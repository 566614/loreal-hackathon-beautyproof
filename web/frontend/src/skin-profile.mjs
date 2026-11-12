// Four observation groups use the same 0–3 frequency scale.
export const observations = [
  "额头在洁面后数小时明显出油", "鼻翼在白天容易泛油光", "T 区需要吸油或补妆", "T 区触摸时有油腻感",
  "两颊在白天明显泛油光", "两颊触摸时有油腻感", "两颊需要吸油或补妆", "两颊在洁面后数小时明显出油",
  "额头在洁面后感到紧绷", "鼻翼出现干燥或起皮", "T 区感到粗糙或缺水", "T 区需要额外保湿才能舒适",
  "两颊在洁面后感到紧绷", "两颊出现干燥或起皮", "两颊感到粗糙或缺水", "两颊需要额外保湿才能舒适",
];

export function assessSkin(answers) {
  const groups = [0, 4, 8, 12].map(start => {
    const values = answers.slice(start, start + 4).filter(value => Number.isInteger(value) && value >= 0 && value <= 3);
    return values.length >= 3 ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  });
  if (groups.some(value => value === null)) return { type: "暂难判断", reason: "每组至少回答 3 道观察题。" };
  const [tOil, cheekOil, tDry, cheekDry] = groups;
  if ((tOil >= 2 && tDry >= 2) || (cheekOil >= 2 && cheekDry >= 2)) {
    return { type: "暂难判断", reason: "同一区域的出油和干燥反应都较明显，建议记录状态后重测。" };
  }
  if (tOil >= 2 && cheekOil < 2 && cheekDry >= 2) return { type: "混合性", reason: "T 区易出油，两颊更偏干。" };
  if (tOil >= 2 && cheekOil >= 2) return { type: "油性", reason: "T 区和两颊都较容易出油。" };
  if (tDry >= 2 && cheekDry >= 2) return { type: "干性", reason: "T 区和两颊都较容易干燥。" };
  if (groups.every(value => value <= 1)) return { type: "中性", reason: "各区域的出油和干燥感都较少。" };
  return { type: "暂难判断", reason: "当前回答未形成清晰类型，建议记录状态后重测。" };
}
