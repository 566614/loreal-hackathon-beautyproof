// Published brand statements are maintained separately from efficacy evidence.
export const products = [
  { name: "适乐肤积雪草舒缓高保湿洁面乳", brand: "适乐肤", market: "中国大陆", category: "洁面",
    skin: ["干性"], goals: ["清洁", "保湿"], texture: [], sensitivity: true, ingredients: null,
    source: "https://www.prnasia.com/lightnews/lightnews-1-77-81120.shtml", sourceDate: "2024-11-10",
    basis: "品牌曾将其列为适合干敏肌的洁面产品" },
  { name: "适乐肤保湿修护润肤霜（C霜）", brand: "适乐肤", market: "中国大陆", category: "面霜",
    skin: ["干性"], goals: ["保湿"], texture: ["霜"], sensitivity: true, ingredients: null,
    source: "https://www.prnasia.com/lightnews/lightnews-1-77-81120.shtml", sourceDate: "2024-11-10",
    basis: "品牌曾将其列为适合干敏肌的润肤产品" },
  { name: "适乐肤水杨酸洁面", brand: "适乐肤", market: "中国大陆", category: "洁面",
    skin: ["油性"], goals: ["清洁", "控油"], texture: [], sensitivity: false, ingredients: null,
    source: "https://www.prnasia.com/story/450511-1.shtml", sourceDate: "2024-06-12",
    basis: "品牌活动曾向油性肌肤用户推荐此洁面" },
  { name: "欧莱雅黑精华第四代", brand: "巴黎欧莱雅", market: "中国大陆", category: "精华",
    skin: ["干性", "油性", "混合性"], goals: [], texture: ["清爽"], sensitivity: null, ingredients: null,
    source: "https://www.lorealparis.com.cn/product/SC567/", sourceDate: "2026-11-12",
    basis: "品牌页面称干性、油性、混合性及敏感肌可用，并称含少量酒精；页面未说明敏感肌测试人数等条件" },
];

export function recommend(profile, category = "全部", catalog = products) {
  if (!profile || profile.result.type === "暂难判断") return [];
  const avoid = (profile.avoid || "").split(/[,，、;；]/).map(value => value.trim().toLowerCase()).filter(Boolean);
  return catalog.filter(product => product.market === "中国大陆" && product.skin.includes(profile.result.type)
    && (category === "全部" || category === product.category)
    && (profile.sensitivity == null || profile.sensitivity < 2 || product.sensitivity === true)
    && (!avoid.length || (Array.isArray(product.ingredients) && !product.ingredients.some(ingredient =>
      avoid.some(term => ingredient.toLowerCase().includes(term))))))
    .map(product => ({ ...product, score: (profile.goals || []).filter(goal => product.goals.includes(goal)).length +
      (profile.preference || []).filter(value => (product.texture || []).includes(value)).length }))
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name, "zh-CN"))
    .slice(0, 3);
}
