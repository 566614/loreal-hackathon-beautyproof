import { test } from "node:test";
import assert from "node:assert/strict";
import { assessSkin, observations } from "./skin-profile.mjs";

test("classifies regional observations and retains uncertainty", () => {
  const answers = values => values.flatMap(value => Array(4).fill(value));
  assert.equal(observations.length, 16);
  assert.equal(assessSkin(answers([0, 0, 2, 2])).type, "干性");
  assert.equal(assessSkin(answers([2, 2, 0, 0])).type, "油性");
  assert.equal(assessSkin(answers([2, 0, 0, 2])).type, "混合性");
  assert.equal(assessSkin(answers([0, 0, 0, 0])).type, "中性");
  assert.equal(assessSkin(answers([2, 0, 2, 0])).type, "暂难判断");
  assert.equal(assessSkin([]).type, "暂难判断");
});
