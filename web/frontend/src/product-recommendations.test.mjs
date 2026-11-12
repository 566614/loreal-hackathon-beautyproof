import { test } from "node:test";
import assert from "node:assert/strict";
import { recommend } from "./product-recommendations.mjs";

test("recommendations respect skin, market, category and unknown ingredients", () => {
  const profile = { result: { type: "干性" }, goals: ["保湿"], avoid: "", sensitivity: 2 };
  const matches = recommend(profile, "面霜");
  assert.equal(matches.length, 1);
  assert.equal(matches[0].brand, "适乐肤");
  assert.ok(matches[0].source.startsWith("https://"));
  assert.deepEqual(recommend({ ...profile, avoid: "酒精" }), []);
  assert.deepEqual(recommend({ ...profile, result: { type: "暂难判断" } }), []);
  assert.deepEqual(recommend(profile, "面霜", [{ ...matches[0], market: "美国" }]), []);
  assert.deepEqual(recommend(profile, "精华"), []);
  assert.ok(recommend({ ...profile, sensitivity: 0 }, "精华").some(item => item.name === "欧莱雅黑精华第四代"));
});
