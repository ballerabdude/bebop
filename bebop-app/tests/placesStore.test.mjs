import assert from "node:assert/strict";
import test from "node:test";
import { PlacesStore } from "../src/robot/placesStore.ts";

function stubFetch(routes) {
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(url);
    const r = routes[url];
    if (!r) return new Response("{}", { status: 404 });
    return typeof r === "function" ? r() : new Response(JSON.stringify(r), { status: 200 });
  };
  return calls;
}
const tick = () => new Promise((r) => setTimeout(r, 10));

test("pending queue is fetched only while a consumer asks for it", async () => {
  const calls = stubFetch({ "/places": { places: [{ id: "a", name: "A" }] }, "/places/pending": { places: [{ id: "p", name: "" }] } });
  const store = new PlacesStore((p) => p);
  const release = store.acquire();
  await tick();
  assert.ok(!calls.includes("/places/pending"));
  const releasePending = store.acquire({ pending: true });
  await tick();
  assert.ok(calls.includes("/places/pending"));
  assert.equal(store.getSnapshot().pending.length, 1);
  releasePending();
  release();
  store.dispose();
});

test("a non-JSON pending reply keeps /places reachable and the last queue", async () => {
  stubFetch({ "/places": { places: [] }, "/places/pending": { places: [{ id: "p", name: "" }] } });
  const store = new PlacesStore((p) => p);
  const release = store.acquire({ pending: true });
  await tick();
  assert.equal(store.getSnapshot().pending.length, 1);
  stubFetch({ "/places": { places: [] }, "/places/pending": () => new Response("<html>", { status: 200 }) });
  await store.refresh();
  const s = store.getSnapshot();
  assert.equal(s.reachable, true);
  assert.equal(s.error, null);
  assert.equal(s.pending.length, 1);
  release();
  store.dispose();
});
