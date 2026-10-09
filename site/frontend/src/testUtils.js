// A fake backend for the component tests: routes fetch calls by path and records them.
export function makeResult(i, overrides = {}) {
	return {
		key: `google:Font ${i}`,
		name: `Font ${i}`,
		source: i % 2 ? "google" : "dafont",
		url: `https://example.com/font-${i}`,
		creator: i % 2 ? null : "Some Designer",
		specimen: `/api/font/specimen/${i}?v=abc`,
		rating: { average: null, count: 0, mine: null },
		vote: 0,
		...overrides,
	};
}

export function makePage({ query = "serif", page = 1, pageSize = 24, total = 100, tags, unmatched = [], inferred = [], suggested = [] } = {}) {
	const totalPages = Math.ceil(total / pageSize);
	const start = (page - 1) * pageSize;
	const count = Math.max(0, Math.min(pageSize, total - start));
	return {
		results: Array.from({ length: count }, (_, k) => makeResult(start + k)),
		page, pageSize, total, totalPages,
		tags: tags || [{ tag: query, weight: 1.0 }],
		unmatched, inferred, suggested,
	};
}

export function jsonResponse(body, status = 200) {
	return Promise.resolve({ ok: status >= 200 && status < 300, status, json: () => Promise.resolve(body) });
}

// handlers: {"/api/font/query": (url, init) => response promise}. Unhandled paths fail the test loudly.
export function installFetch(handlers) {
	// unless a test says otherwise, a query means one tag named after it
	const routes = {
		"/api/font/tags": ({ params }) => jsonResponse({ tags: params.query ? [{ tag: params.query, weight: 1 }] : [], suggested: [], unmatched: [] }),
		// and nothing to narrow it down with
		"/api/font/refine": () => jsonResponse({ refinements: [] }),
		...handlers,
	};
	const calls = [];
	global.fetch = jest.fn((input, init = {}) => {
		const url = new URL(input, "http://localhost");
		const call = { path: url.pathname, params: Object.fromEntries(url.searchParams), init,
			body: init.body && typeof init.body === "string" ? JSON.parse(init.body) : init.body };
		calls.push(call);
		const handler = routes[url.pathname];
		if (!handler) throw new Error("unexpected fetch " + url.pathname);
		return new Promise((resolve, reject) => {
			Promise.resolve(handler(call)).then(response => {
				if (init.signal && init.signal.aborted) reject(new DOMException("aborted", "AbortError"));
				else resolve(response);
			}, reject);
		});
	});
	return calls;
}
