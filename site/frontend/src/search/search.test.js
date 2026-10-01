import { render, screen, waitFor, within, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchPage, { pageWindow } from './index';
import { installFetch, jsonResponse, makePage, makeResult } from '../testUtils';

beforeEach(() => {
	window.history.replaceState({}, "", "/");
	window.HTMLElement.prototype.scrollIntoView = jest.fn();
});

const queryCalls = (calls) => calls.filter(c => c.path === "/api/font/query");

function queryHandler(total = 100, extra = {}) {
	return ({ params }) => jsonResponse(makePage({ query: params.query, page: Number(params.page), total, ...extra }));
}

async function search(text) {
	userEvent.type(screen.getByLabelText("Describe a font"), text + "{enter}");
}

describe("pageWindow", () => {
	test("small counts show every page", () => {
		expect(pageWindow(1, 3)).toEqual([1, 2, 3]);
		expect(pageWindow(1, 1)).toEqual([1]);
	});
	test("gaps are null and first/last are always present", () => {
		expect(pageWindow(1, 50)).toEqual([1, 2, 3, null, 50]);
		expect(pageWindow(25, 50)).toEqual([1, null, 23, 24, 25, 26, 27, null, 50]);
		expect(pageWindow(50, 50)).toEqual([1, null, 48, 49, 50]);
		expect(pageWindow(4, 50)).toEqual([1, 2, 3, 4, 5, 6, null, 50]);
	});
});

describe("searching", () => {
	test("nothing is fetched until a search is submitted", () => {
		const calls = installFetch({});
		render(<SearchPage username={null} />);
		expect(calls).toHaveLength(0);
		expect(screen.queryByRole("article")).toBeNull();
	});

	test("empty or blank submissions do nothing", () => {
		const calls = installFetch({});
		render(<SearchPage username={null} />);
		userEvent.type(screen.getByLabelText("Describe a font"), "{enter}");
		userEvent.type(screen.getByLabelText("Describe a font"), "   {enter}");
		expect(calls).toHaveLength(0);
	});

	test("keeps the original prompt above the search box", () => {
		installFetch({});
		render(<SearchPage username={null} />);
		expect(screen.getByText("Please enter a description to search for a font")).toBeInTheDocument();
	});

	test("shows specimen images linking out, with the matched tags", async () => {
		const calls = installFetch({
			"/api/font/tags": () => jsonResponse({ tags: [{ tag: "serif", weight: 1 }, { tag: "bold", weight: 1 }], suggested: [], unmatched: [] }),
			"/api/font/query": queryHandler(30),
		});
		render(<SearchPage username={null} />);
		await search("serif bold");

		const images = await screen.findAllByRole("img");
		expect(images).toHaveLength(24);
		expect(images[0]).toHaveAttribute("src", "/api/font/specimen/0?v=abc");
		expect(images[0]).toHaveAttribute("alt", "Font 0 specimen");
		expect(images[0].closest("a")).toHaveAttribute("href", "https://example.com/font-0");
		expect(images[0].closest("a")).toHaveAttribute("target", "_blank");
		expect(images[0].closest("a").getAttribute("rel")).toContain("noopener");
		const chips = within(await screen.findByRole("list", { name: "Tags in your search" })).getAllByRole("listitem");
		expect(chips.map(c => c.textContent)).toEqual(["serif", "bold"]);
		expect(within(chips[1]).getByRole("button")).toHaveAttribute("aria-label", "bold: included");
		expect(calls[0]).toMatchObject({ path: "/api/font/tags", params: { query: "serif bold" } });
		expect(calls[1].params).toEqual({ query: "serif bold", tags: "serif:1,bold:1", page: "1", pageSize: "24" });
		expect(screen.getAllByText("Google Fonts")).toHaveLength(12);
		expect(screen.getAllByText("DaFont · Some Designer")).toHaveLength(12);
	});

	test("unrecognised words are reported; nothing recognised says so", async () => {
		installFetch({
			"/api/font/tags": () => jsonResponse({ tags: [], suggested: [], unmatched: ["qwerty"] }),
			"/api/font/query": () => jsonResponse(makePage({ total: 0, tags: [] })),
		});
		render(<SearchPage username={null} />);
		await search("qwerty");
		expect(await screen.findByText("No tags recognised in that description.")).toBeInTheDocument();
		expect(screen.getByText("Not recognised: qwerty.")).toBeInTheDocument();
		expect(screen.queryByRole("img")).toBeNull();
		expect(screen.queryByRole("navigation")).toBeNull();
	});

	test("a failed search shows the server's message", async () => {
		installFetch({ "/api/font/query": () => jsonResponse({ message: "Query is longer than 200 characters" }, 400) });
		render(<SearchPage username={null} />);
		await search("x");
		expect(await screen.findByRole("alert")).toHaveTextContent("Query is longer than 200 characters");
	});

	test("a network failure is shown, not thrown", async () => {
		installFetch({ "/api/font/query": () => Promise.reject(new Error("offline")) });
		render(<SearchPage username={null} />);
		await search("x");
		expect(await screen.findByRole("alert")).toHaveTextContent("offline");
	});

	test("a slow earlier response never overwrites a newer search", async () => {
		let releaseFirst;
		installFetch({
			"/api/font/query": ({ params }) => params.query === "first"
				? new Promise(resolve => { releaseFirst = () => resolve(makePageResponse("first")); })
				: jsonResponse(makePage({ query: "second", total: 5, tags: [{ tag: "second", weight: 1 }] })),
		});
		const makePageResponse = q => ({ ok: true, status: 200, json: () => Promise.resolve(makePage({ query: q, total: 5, tags: [{ tag: q, weight: 1 }] })) });
		render(<SearchPage username={null} />);
		const box = screen.getByLabelText("Describe a font");
		userEvent.type(box, "first{enter}");
		await waitFor(() => expect(releaseFirst).toBeDefined());       // the first search's fonts are now in flight
		userEvent.clear(box);
		userEvent.type(box, "second{enter}");
		const tags = () => screen.getByRole("list", { name: "Tags in your search" });
		expect(await within(await screen.findByRole("list", { name: "Tags in your search" })).findByText("second")).toBeInTheDocument();
		await act(async () => { releaseFirst(); });
		expect(within(tags()).getByText("second")).toBeInTheDocument();
		expect(within(tags()).queryByText("first")).toBeNull();
	});
});

describe("tag line", () => {
	const slimy = [{ tag: "grunge", via: "dirty", similarity: 0.68 }, { tag: "distressed", via: "dirty", similarity: 0.68 },
		{ tag: "horror", via: "creepy", similarity: 0.62 }];

	// The server only lists the tags for a query and ranks the list it is given, like the real one
	function tagsFor(query) {
		const words = query.split(" ");
		const tags = [];
		words.forEach((w, i) => {
			if (["serif", "bold", "thin", "script", "feminine"].includes(w) && words[i - 1] !== "not") tags.push({ tag: w, weight: 1 });
			if (w === "rounded") tags.push({ tag: "rounded", weight: 0.6 });
			if (w === "airy") ["thin", "feminine"].forEach(t => { if (!tags.some(x => x.tag === t)) tags.push({ tag: t, weight: 0.5 }); });
		});
		const unmatched = words.filter(w => ["qwerty"].includes(w));
		const suggested = words.includes("slimy") ? slimy : [];
		return { tags, suggested, unmatched };
	}
	function backend(calls) {
		return {
			"/api/font/tags": ({ params }) => jsonResponse(tagsFor(params.query)),
			"/api/font/query": ({ params }) => jsonResponse(makePage({ query: params.query, page: Number(params.page), total: params.tags ? 30 : 0,
				tags: params.tags.split(",").filter(Boolean).map(t => ({ tag: t.split(":")[0], weight: 1 })) })),
		};
	}
	const line = () => screen.getByRole("list", { name: "Tags in your search" });
	const box = (name, state) => within(line()).getByRole("button", { name: `${name}: ${state}` });
	const lastQuery = (calls) => [...calls].reverse().find(c => c.path === "/api/font/query").params;
	const tagCalls = (calls) => calls.filter(c => c.path === "/api/font/tags");
	async function searched(text, calls = installFetch(backend())) {
		render(<SearchPage username={null} />);
		await search(text);
		await within(await screen.findByRole("list", { name: "Tags in your search" })).findAllByRole("listitem");
		return calls;
	}

	test("each tag is a word with a ticked box; there is no negation", async () => {
		await searched("serif bold not thin");
		expect(within(line()).getAllByRole("listitem").map(i => i.textContent)).toEqual(["serif", "bold"]);
		expect(box("serif", "included")).toHaveClass("TagBox-on");
		expect(box("bold", "included")).toBeInTheDocument();
		expect(screen.queryByRole("button", { name: /excluded/ })).toBeNull();
		expect(screen.queryByText("Searching for:")).toBeNull();          // the words speak for themselves
	});

	test("guessed tags and suggestions are plain tags in the same list: guesses ticked, suggestions not", async () => {
		await searched("airy slimy");
		expect(within(line()).getAllByRole("listitem").map(i => i.textContent))
			.toEqual(["thin", "feminine", "grunge", "distressed", "horror"]);
		expect(box("thin", "included")).toBeInTheDocument();
		expect(box("grunge", "off")).toHaveClass("TagBox-off");
		expect(screen.queryByText(/similar to/i)).toBeNull();
		expect(screen.queryByText(/→/)).toBeNull();
	});

	test("clicking flips a box between ticked and empty, all in the page", async () => {
		const calls = await searched("serif bold");
		expect(tagCalls(calls)).toHaveLength(1);
		expect(lastQuery(calls).tags).toBe("serif:1,bold:1");

		userEvent.click(box("serif", "included"));                     // tick -> empty
		await waitFor(() => expect(box("serif", "off")).toBeInTheDocument());
		await waitFor(() => expect(lastQuery(calls).tags).toBe("bold:1"));

		userEvent.click(box("serif", "off"));                          // empty -> tick
		await waitFor(() => expect(box("serif", "included")).toBeInTheDocument());
		await waitFor(() => expect(lastQuery(calls).tags).toBe("serif:1,bold:1"));
		expect(box("bold", "included")).toBeInTheDocument();           // the other tag never moved
		expect(tagCalls(calls)).toHaveLength(1);                       // the server was never asked about the ticks
		expect(window.location.search).toBe("?q=serif+bold");          // and they are not in the URL
	});

	test("a tag's weight rides along unchanged", async () => {
		const calls = await searched("rounded serif");
		expect(lastQuery(calls).tags).toBe("rounded:0.6,serif:1");
		userEvent.click(box("rounded", "included"));
		await waitFor(() => expect(box("rounded", "off")).toBeInTheDocument());
		userEvent.click(box("rounded", "off"));
		await waitFor(() => expect(lastQuery(calls).tags).toBe("rounded:0.6,serif:1"));
	});

	test("a box says what a click will do", async () => {
		await searched("serif");
		expect(box("serif", "included")).toHaveAttribute("title", "Included. Click to turn off.");
		userEvent.click(box("serif", "included"));
		await waitFor(() => expect(box("serif", "off")).toHaveAttribute("title", "Off. Click to include."));
	});

	test("a vote is filed under the tags the results were ranked on", async () => {
		const calls = installFetch({ ...backend(), "/api/font/approve": ({ body }) => jsonResponse({ message: "Successful", vote: body.vote }) });
		render(<SearchPage username="alice" />);
		await search("serif bold");
		await screen.findAllByRole("img");
		const vote = async (name) => {
			userEvent.click(within(screen.getAllByRole("article")[0]).getByRole("button", { name }));
			await waitFor(() => expect(calls.filter(c => c.path === "/api/font/approve").length).toBeGreaterThan(0));
			return calls.filter(c => c.path === "/api/font/approve").pop().body;
		};
		expect(await vote("This font matched my query")).toEqual({ fontKey: "google:Font 0", query: "serif bold", tags: "serif:1,bold:1", vote: 1 });

		userEvent.click(box("bold", "included"));
		await waitFor(() => expect(lastQuery(calls).tags).toBe("serif:1"));
		await waitFor(() => expect(screen.getAllByRole("article")[0].querySelector("[aria-pressed=true]")).toBeNull());
		expect(await vote("This font did not match my query")).toEqual({ fontKey: "google:Font 0", query: "serif bold", tags: "serif:1", vote: -1 });
	});

	test("a suggestion starts empty and its first click ticks it, in place", async () => {
		const calls = await searched("serif slimy");
		expect(lastQuery(calls).tags).toBe("serif:1");
		userEvent.click(box("grunge", "off"));
		await waitFor(() => expect(box("grunge", "included")).toBeInTheDocument());
		await waitFor(() => expect(lastQuery(calls).tags).toBe("serif:1,grunge:1"));
		expect(within(line()).getAllByRole("listitem").map(i => i.textContent)).toEqual(["serif", "grunge", "distressed", "horror"]);
	});

	test("with nothing ticked there are no fonts and the page says why", async () => {
		const calls = await searched("slimy");
		expect(await screen.findByText("Tick a tag to see fonts.")).toBeInTheDocument();
		expect(lastQuery(calls).tags).toBe("");
		userEvent.click(box("grunge", "off"));
		await waitFor(() => expect(screen.queryByText("Tick a tag to see fonts.")).toBeNull());
		expect((await screen.findAllByRole("img"))).toHaveLength(24);
	});

	test("changing a tag goes back to the first page; paging keeps the choices without asking again", async () => {
		const calls = await searched("serif bold");
		userEvent.click(await screen.findByRole("button", { name: "Next" }));
		await screen.findByText("Page 2 of 2 · 30 fonts");
		expect(lastQuery(calls)).toMatchObject({ page: "2", tags: "serif:1,bold:1" });
		userEvent.click(box("bold", "included"));
		await screen.findByText("Page 1 of 2 · 30 fonts");
		expect(lastQuery(calls)).toMatchObject({ page: "1", tags: "serif:1" });
		userEvent.click(await screen.findByRole("button", { name: "Next" }));
		await screen.findByText("Page 2 of 2 · 30 fonts");
		expect(lastQuery(calls)).toMatchObject({ page: "2", tags: "serif:1" });   // still without bold
		expect(box("bold", "off")).toBeInTheDocument();
		expect(tagCalls(calls)).toHaveLength(1);
		expect(window.location.search).toBe("?q=serif+bold&page=2");
	});

	test("searching again, even the same words, starts clean", async () => {
		const calls = await searched("serif bold");
		userEvent.click(box("serif", "included"));
		await waitFor(() => expect(box("serif", "off")).toBeInTheDocument());
		userEvent.type(screen.getByLabelText("Describe a font"), "{enter}");   // the same text again
		await waitFor(() => expect(box("serif", "included")).toBeInTheDocument());
		expect(tagCalls(calls)).toHaveLength(2);
		expect(lastQuery(calls).tags).toBe("serif:1,bold:1");
	});

	test("a different query loads its own tags; back/forward does too", async () => {
		const calls = await searched("serif");
		userEvent.clear(screen.getByLabelText("Describe a font"));
		userEvent.type(screen.getByLabelText("Describe a font"), "bold{enter}");
		await waitFor(() => expect(box("bold", "included")).toBeInTheDocument());
		expect(within(line()).queryByText("serif")).toBeNull();

		window.history.pushState({}, "", "/?q=serif");
		act(() => { window.dispatchEvent(new PopStateEvent("popstate")); });
		await waitFor(() => expect(box("serif", "included")).toBeInTheDocument());
		expect(tagCalls(calls).map(c => c.params.query)).toEqual(["serif", "bold", "serif"]);
	});

	test("opening a shared link loads the tags for its query and page", async () => {
		window.history.replaceState({}, "", "/?q=serif+bold&page=2");
		const calls = installFetch(backend());
		render(<SearchPage username={null} />);
		await screen.findByText("Page 2 of 2 · 30 fonts");
		expect(lastQuery(calls)).toMatchObject({ query: "serif bold", tags: "serif:1,bold:1", page: "2" });
	});

	test("words that match nothing are listed as not recognised", async () => {
		await searched("serif qwerty");
		expect(await screen.findByText("Not recognised: qwerty.")).toBeInTheDocument();
	});

	test("nothing recognised says so", async () => {
		installFetch({
			"/api/font/tags": () => jsonResponse({ tags: [], suggested: [], unmatched: ["qwerty"] }),
			"/api/font/query": () => jsonResponse(makePage({ total: 0, tags: [] })),
		});
		render(<SearchPage username={null} />);
		await search("qwerty");
		expect(await screen.findByText("No tags recognised in that description.")).toBeInTheDocument();
		expect(screen.getByText("Not recognised: qwerty.")).toBeInTheDocument();
	});

	test("a failure listing the tags is shown", async () => {
		installFetch({ "/api/font/tags": () => jsonResponse({ message: "Query is longer than 200 characters" }, 400) });
		render(<SearchPage username={null} />);
		await search("x");
		expect(await screen.findByRole("alert")).toHaveTextContent("Query is longer than 200 characters");
		expect(screen.queryByText("Searching…")).toBeNull();
	});
});

describe("pagination", () => {
	test("pages through the results, as many as the user likes", async () => {
		const calls = installFetch({ "/api/font/query": queryHandler(288) });
		render(<SearchPage username={null} />);
		await search("serif");
		const nav = await screen.findByRole("navigation", { name: "Pages of results" });
		expect(within(nav).getByText("Page 1 of 12 · 288 fonts")).toBeInTheDocument();
		expect(within(nav).getByRole("button", { name: "Previous" })).toBeDisabled();

		userEvent.click(within(nav).getByRole("button", { name: "Next" }));
		await screen.findByText("Page 2 of 12 · 288 fonts");
		expect(calls[calls.length - 1].params.page).toBe("2");
		expect(screen.getByAltText("Font 24 specimen")).toBeInTheDocument();
		expect(screen.queryByAltText("Font 0 specimen")).toBeNull();
		expect(window.HTMLElement.prototype.scrollIntoView).toHaveBeenCalled();

		userEvent.click(within(screen.getByRole("navigation")).getByRole("button", { name: "12" }));
		await screen.findByText("Page 12 of 12 · 288 fonts");
		expect(within(screen.getByRole("navigation")).getByRole("button", { name: "Next" })).toBeDisabled();
		expect(screen.getAllByRole("img")).toHaveLength(24);

		userEvent.click(within(screen.getByRole("navigation")).getByRole("button", { name: "Previous" }));
		await screen.findByText("Page 11 of 12 · 288 fonts");
	});

	test("the current page is marked and a single page has no pager", async () => {
		installFetch({ "/api/font/query": queryHandler(10) });
		render(<SearchPage username={null} />);
		await search("serif");
		await screen.findAllByRole("img");
		expect(screen.queryByRole("navigation")).toBeNull();
	});

	test("a new search goes back to page 1", async () => {
		const calls = installFetch({ "/api/font/query": queryHandler(288) });
		render(<SearchPage username={null} />);
		await search("serif");
		userEvent.click(await screen.findByRole("button", { name: "Next" }));
		await screen.findByText("Page 2 of 12 · 288 fonts");
		userEvent.clear(screen.getByLabelText("Describe a font"));
		userEvent.type(screen.getByLabelText("Describe a font"), "bold{enter}");
		await screen.findByText("Page 1 of 12 · 288 fonts");
		expect(calls[calls.length - 1].params).toMatchObject({ query: "bold", page: "1" });
	});

	test("the query and page live in the URL, and back goes back", async () => {
		const calls = installFetch({ "/api/font/query": queryHandler(288) });
		render(<SearchPage username={null} />);
		await search("serif bold");
		userEvent.click(await screen.findByRole("button", { name: "Next" }));
		await screen.findByText("Page 2 of 12 · 288 fonts");
		expect(window.location.search).toBe("?q=serif+bold&page=2");

		window.history.pushState({}, "", "/?q=serif+bold");
		act(() => { window.dispatchEvent(new PopStateEvent("popstate")); });
		await screen.findByText("Page 1 of 12 · 288 fonts");
		expect(calls[calls.length - 1].params.page).toBe("1");
	});

	test("opens straight onto a page from a shared link", async () => {
		window.history.replaceState({}, "", "/?q=script&page=3");
		const calls = installFetch({ "/api/font/query": queryHandler(288) });
		render(<SearchPage username={null} />);
		await screen.findByText("Page 3 of 12 · 288 fonts");
		expect(queryCalls(calls)[0].params).toMatchObject({ query: "script", page: "3" });
		expect(screen.getByLabelText("Describe a font")).toHaveValue("script");
	});

	test("a link past the last page lands on the last page", async () => {
		window.history.replaceState({}, "", "/?q=script&page=99");
		const calls = installFetch({ "/api/font/query": queryHandler(50) });
		render(<SearchPage username={null} />);
		await screen.findByText("Page 3 of 3 · 50 fonts");
		expect(queryCalls(calls).map(c => c.params.page)).toEqual(["99", "3"]);
	});

	test("garbage in the URL is ignored", async () => {
		window.history.replaceState({}, "", "/?q=script&page=abc");
		const calls = installFetch({ "/api/font/query": queryHandler(50) });
		render(<SearchPage username={null} />);
		await screen.findAllByRole("img");
		expect(queryCalls(calls)[0].params.page).toBe("1");
	});
});

describe("feedback", () => {
	async function loaded(username, handlers = {}, props = {}) {
		const calls = installFetch({ "/api/font/query": queryHandler(30), ...handlers });
		const onNeedLogin = jest.fn();
		const view = render(<SearchPage username={username} onNeedLogin={onNeedLogin} {...props} />);
		await search("serif");
		await screen.findAllByRole("img");
		return { calls, onNeedLogin, view };
	}
	const firstCard = () => screen.getAllByRole("article")[0];

	test("logged out: asks to log in and sends nothing", async () => {
		const { calls, onNeedLogin } = await loaded(null);
		userEvent.click(within(firstCard()).getByRole("button", { name: "This font matched my query" }));
		expect(onNeedLogin).toHaveBeenCalled();
		expect(within(firstCard()).getByRole("status")).toHaveTextContent("Log in to give feedback");
		userEvent.click(within(firstCard()).getByRole("button", { name: "This font did not match my query" }));
		expect(calls.filter(c => c.init.method === "POST")).toHaveLength(0);
	});

	test("approve, flip, and clear", async () => {
		const answers = [];
		const { calls } = await loaded("alice", { "/api/font/approve": ({ body }) => { answers.push(body.vote); return jsonResponse({ message: "Successful", vote: body.vote }); } });
		const yes = () => within(firstCard()).getByRole("button", { name: "This font matched my query" });
		const no = () => within(firstCard()).getByRole("button", { name: "This font did not match my query" });

		userEvent.click(yes());
		await waitFor(() => expect(yes()).toHaveAttribute("aria-pressed", "true"));
		expect(calls[calls.length - 1].body).toEqual({ fontKey: "google:Font 0", query: "serif", tags: "serif:1", vote: 1 });

		userEvent.click(no());
		await waitFor(() => expect(no()).toHaveAttribute("aria-pressed", "true"));
		expect(yes()).toHaveAttribute("aria-pressed", "false");

		userEvent.click(no());
		await waitFor(() => expect(no()).toHaveAttribute("aria-pressed", "false"));
		expect(answers).toEqual([1, -1, 0]);
	});

	test("an existing vote from the server is shown pressed", async () => {
		installFetch({ "/api/font/query": () => jsonResponse({ ...makePage({ total: 1 }), results: [makeResult(0, { vote: -1 })] }) });
		render(<SearchPage username="alice" />);
		await search("serif");
		const no = await screen.findByRole("button", { name: "This font did not match my query" });
		expect(no).toHaveAttribute("aria-pressed", "true");
	});

	test("an expired session asks to log in again", async () => {
		const { onNeedLogin } = await loaded("alice", { "/api/font/approve": () => jsonResponse({ message: "Session expired" }, 401) });
		userEvent.click(within(firstCard()).getByRole("button", { name: "This font did not match my query" }));
		await within(firstCard()).findByText("Your session ended, please log in again");
		expect(onNeedLogin).toHaveBeenCalled();
	});

	test("server and network errors are shown on the card", async () => {
		let call = 0;
		await loaded("alice", {
			"/api/font/approve": () => ++call === 1 ? jsonResponse({ message: "Font not found" }, 404) : Promise.reject(new Error("offline")),
		});
		userEvent.click(within(firstCard()).getByRole("button", { name: "This font matched my query" }));
		await within(firstCard()).findByText("Font not found");
		userEvent.click(within(firstCard()).getByRole("button", { name: "This font did not match my query" }));
		await within(firstCard()).findByText("Could not reach the server");
	});

	test("the description box is hidden unless asked for", async () => {
		await loaded("alice");
		expect(screen.queryByRole("button", { name: "Describe" })).toBeNull();
	});

	test("thumbs say what they mean on hover", async () => {
		await loaded("alice");
		const up = within(firstCard()).getByRole("button", { name: "This font matched my query" });
		const down = within(firstCard()).getByRole("button", { name: "This font did not match my query" });
		expect(up).toHaveAttribute("title", "This font matched my query");
		expect(down).toHaveAttribute("title", "This font did not match my query");
	});

	test("the source sits under the font name, there are no stars, and the thumbs are the only feedback", async () => {
		await loaded("alice");
		const card = firstCard();
		const title = card.querySelector(".ResultTitle");
		expect([...title.children].map(e => e.className || e.tagName)).toEqual(["A", "ResultSource"]);
		const feedback = card.querySelector(".ResultFeedback");
		expect([...feedback.children].map(e => e.className)).toEqual(["Votes"]);
		expect(within(card).queryByRole("button", { name: /Rate \d star/ })).toBeNull();
		expect(within(card).queryByText("unrated")).toBeNull();
	});

	test("describing a font", async () => {
		const { calls } = await loaded("alice", { "/api/font/describe": () => jsonResponse({ message: "Successful" }) }, { allowDescriptions: true });
		userEvent.click(within(firstCard()).getByRole("button", { name: "Describe" }));
		userEvent.type(within(firstCard()).getByLabelText("Font description"), "warm and friendly{enter}");
		await within(firstCard()).findByText("Thanks, description saved");
		expect(calls[calls.length - 1].body).toEqual({ fontKey: "google:Font 0", description: "warm and friendly" });
		expect(within(firstCard()).queryByLabelText("Font description")).toBeNull();
	});

	test("logging in re-fetches so the user's own votes show", async () => {
		const calls = installFetch({ "/api/font/query": () => jsonResponse({ ...makePage({ total: 1 }), results: [makeResult(0, { vote: queryCalls(calls).length > 1 ? 1 : 0 })] }) });
		const view = render(<SearchPage username={null} />);
		await search("serif");
		await screen.findAllByRole("img");
		expect(screen.getByRole("button", { name: "This font matched my query" })).toHaveAttribute("aria-pressed", "false");
		view.rerender(<SearchPage username="alice" />);
		await waitFor(() => expect(screen.getByRole("button", { name: "This font matched my query" })).toHaveAttribute("aria-pressed", "true"));
		expect(calls.filter(c => c.path === "/api/font/query")).toHaveLength(2);
	});
});
