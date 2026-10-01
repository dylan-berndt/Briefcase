import { render, screen, waitFor, within, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchPage, { pageWindow } from './index';
import { installFetch, jsonResponse, makePage, makeResult } from '../testUtils';

beforeEach(() => {
	window.history.replaceState({}, "", "/");
	window.HTMLElement.prototype.scrollIntoView = jest.fn();
});

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
		const calls = installFetch({ "/api/font/query": queryHandler(30, { tags: [{ tag: "serif", weight: 1 }, { tag: "bold", weight: -1 }] }) });
		render(<SearchPage username={null} />);
		await search("serif not bold");

		const images = await screen.findAllByRole("img");
		expect(images).toHaveLength(24);
		expect(images[0]).toHaveAttribute("src", "/api/font/specimen/0?v=abc");
		expect(images[0]).toHaveAttribute("alt", "Font 0 specimen");
		expect(images[0].closest("a")).toHaveAttribute("href", "https://example.com/font-0");
		expect(images[0].closest("a")).toHaveAttribute("target", "_blank");
		expect(images[0].closest("a").getAttribute("rel")).toContain("noopener");
		const chips = within(await screen.findByRole("list", { name: "Tags in your search" })).getAllByRole("listitem");
		expect(chips.map(c => c.textContent)).toEqual(["serif", "bold"]);
		expect(within(chips[1]).getByRole("button")).toHaveAttribute("aria-label", "bold: excluded");
		expect(calls[0].params).toEqual({ query: "serif not bold", page: "1", pageSize: "24" });
		expect(screen.getAllByText("Google Fonts")).toHaveLength(12);
		expect(screen.getAllByText("DaFont · Some Designer")).toHaveLength(12);
	});

	test("unrecognised words are reported; nothing recognised says so", async () => {
		installFetch({ "/api/font/query": () => jsonResponse(makePage({ total: 0, tags: [], unmatched: ["qwerty"] })) });
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
	const inferredAiry = { word: "airy", tags: ["thin", "feminine"], weight: 0.5 };
	const slimy = [
		{ tag: "grunge", via: "dirty", similarity: 0.68 },
		{ tag: "distressed", via: "dirty", similarity: 0.68 },
		{ tag: "horror", via: "creepy", similarity: 0.62 }];

	// A backend double that follows the real one: tags= force a sign (or add a tag), ignore= turns things off, and a
	// tag that is in the search is not suggested
	function backend() {
		return ({ params }) => {
			const forced = Object.fromEntries((params.tags || "").split(",").filter(Boolean)
				.map(t => t.startsWith("-") ? [t.slice(1), -1] : [t, 1]));
			const off = (params.ignore || "").split(",").filter(Boolean);
			const words = params.query.split(" ");
			const typed = ["serif", "bold", "thin"].filter(w => words.includes(w));
			const negated = words.map((w, i) => w !== "not" && words[i - 1] === "not" ? w : null).filter(Boolean);
			const names = [...new Set([...typed, ...Object.keys(forced).filter(n => !["airy"].includes(n))])].filter(n => !off.includes(n));
			const tags = names.map(n => ({ tag: n, weight: forced[n] ?? (negated.includes(n) ? -1 : 1) }));
			const guessed = words.includes("airy") && !off.includes("airy");
			const inferred = guessed ? [{ ...inferredAiry, weight: 0.5 * (forced.airy ?? 1) }] : [];
			const unmatched = words.filter(w => ["slimy", "airy", "qwerty"].includes(w) && !(w === "airy" && guessed));
			const suggested = unmatched.includes("slimy")
				? [{ word: "slimy", tags: slimy.filter(t => !names.includes(t.tag)) }] : [];
			return jsonResponse(makePage({ query: params.query, page: Number(params.page), total: 30, tags, unmatched, inferred, suggested }));
		};
	}
	const line = () => screen.getByRole("list", { name: "Tags in your search" });
	const box = (name, state) => within(line()).getByRole("button", { name: `${name}: ${state}` });
	const lastParams = (calls) => calls[calls.length - 1].params;

	test("each tag is a word with a ticked box, and an excluded one is crossed", async () => {
		installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("serif bold not thin");
		await within(await screen.findByRole("list", { name: "Tags in your search" })).findByText("serif");
		expect(within(line()).getAllByRole("listitem").map(i => i.textContent)).toEqual(["serif", "bold", "thin"]);
		expect(box("serif", "included")).toHaveClass("TagBox-on");
		expect(box("bold", "included")).toBeInTheDocument();
		expect(box("thin", "excluded")).toHaveClass("TagBox-neg");
		expect(screen.getByText("Searching for:")).toBeInTheDocument();
	});

	test("clicking steps tick -> empty -> cross -> empty -> tick", async () => {
		const calls = installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("serif bold");
		await within(await screen.findByRole("list", { name: "Tags in your search" })).findByText("serif");

		userEvent.click(box("serif", "included"));                     // tick -> empty
		await waitFor(() => expect(box("serif", "off")).toHaveClass("TagBox-off"));
		expect(lastParams(calls)).toMatchObject({ ignore: "serif", page: "1" });
		expect(lastParams(calls).tags).toBeUndefined();

		userEvent.click(box("serif", "off"));                          // empty -> cross
		await waitFor(() => expect(box("serif", "excluded")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ tags: "-serif" });
		expect(lastParams(calls).ignore).toBeUndefined();

		userEvent.click(box("serif", "excluded"));                     // cross -> empty
		await waitFor(() => expect(box("serif", "off")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ ignore: "serif" });
		expect(lastParams(calls).tags).toBeUndefined();

		userEvent.click(box("serif", "off"));                          // empty -> tick
		await waitFor(() => expect(box("serif", "included")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ tags: "serif" });
		expect(lastParams(calls).ignore).toBeUndefined();
		expect(box("bold", "included")).toBeInTheDocument();           // the other tag never moved
	});

	test("a box says what a click will do", async () => {
		installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("serif");
		await within(await screen.findByRole("list", { name: "Tags in your search" })).findByText("serif");
		expect(box("serif", "included")).toHaveAttribute("title", "Included. Click to turn off.");
		userEvent.click(box("serif", "included"));
		await waitFor(() => expect(box("serif", "off")).toHaveAttribute("title", "Off. Click to exclude."));
		userEvent.click(box("serif", "off"));
		await waitFor(() => expect(box("serif", "excluded")).toHaveAttribute("title", "Excluded. Click to turn off."));
		userEvent.click(box("serif", "excluded"));
		await waitFor(() => expect(box("serif", "off")).toHaveAttribute("title", "Off. Click to include."));
	});

	test("a tag typed with 'not' starts crossed and can be ticked", async () => {
		const calls = installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("not thin");
		await within(await screen.findByRole("list", { name: "Tags in your search" })).findByText("thin");
		userEvent.click(box("thin", "excluded"));
		await waitFor(() => expect(box("thin", "off")).toBeInTheDocument());
		userEvent.click(box("thin", "off"));                           // it came from a cross, so it goes to a tick
		await waitFor(() => expect(box("thin", "included")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ tags: "thin" });
	});

	test("a guessed word shows what it was guessed as and has the same box", async () => {
		const calls = installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("airy");
		const item = (await within(await screen.findByRole("list", { name: "Tags in your search" })).findByText("airy")).closest("li");
		expect(item).toHaveTextContent("airy→ thin, feminine");
		expect(item).toHaveAttribute("title", expect.stringContaining("Guessed"));
		userEvent.click(box("airy", "included"));
		await waitFor(() => expect(box("airy", "off")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ ignore: "airy" });
		userEvent.click(box("airy", "off"));
		await waitFor(() => expect(box("airy", "excluded")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ tags: "-airy" });
		expect(within(line()).getByText("→ thin, feminine")).toBeInTheDocument();
	});

	test("suggestions are on the same line with empty boxes, and the first click ticks them", async () => {
		const calls = installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("serif slimy");
		await within(await screen.findByRole("list", { name: "Tags in your search" })).findByText("grunge");
		expect(within(line()).getAllByRole("listitem").map(i => i.textContent))
			.toEqual(["serif", "similar to “slimy”:", "grunge", "distressed", "horror"]);
		expect(box("grunge", "off")).toHaveAttribute("title", "Off. Click to include.");
		expect(within(line()).getByText("grunge").closest("li")).toHaveAttribute("title", "Similar to “dirty” (0.68)");
		expect(calls[0].params.tags).toBeUndefined();

		userEvent.click(box("grunge", "off"));
		await waitFor(() => expect(box("grunge", "included")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ tags: "grunge" });
		// it is part of the search now, in the main group, and not offered twice
		expect(within(line()).getAllByText("grunge")).toHaveLength(1);
		expect(within(line()).getAllByRole("listitem").map(i => i.textContent))
			.toEqual(["serif", "grunge", "similar to “slimy”:", "distressed", "horror"]);
	});

	test("an added suggestion stays visible when turned off, and can then be crossed", async () => {
		const calls = installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("slimy");
		userEvent.click(await within(await screen.findByRole("list", { name: "Tags in your search" })).findByRole("button", { name: "grunge: off" }));
		await waitFor(() => expect(box("grunge", "included")).toBeInTheDocument());
		userEvent.click(box("grunge", "included"));
		await waitFor(() => expect(box("grunge", "off")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ ignore: "grunge" });
		userEvent.click(box("grunge", "off"));
		await waitFor(() => expect(box("grunge", "excluded")).toBeInTheDocument());
		expect(lastParams(calls)).toMatchObject({ tags: "-grunge" });
		expect(lastParams(calls).ignore).toBeUndefined();
	});

	test("paging keeps the choices, a new search starts clean", async () => {
		const calls = installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("serif airy");
		userEvent.click(await within(await screen.findByRole("list", { name: "Tags in your search" })).findByRole("button", { name: "airy: included" }));
		await waitFor(() => expect(box("airy", "off")).toBeInTheDocument());
		userEvent.click(await screen.findByRole("button", { name: "Next" }));
		await screen.findByText("Page 2 of 2 · 30 fonts");
		expect(lastParams(calls)).toMatchObject({ page: "2", ignore: "airy" });
		expect(window.location.search).toBe("?q=serif+airy&ignore=airy&page=2");

		userEvent.clear(screen.getByLabelText("Describe a font"));
		userEvent.type(screen.getByLabelText("Describe a font"), "serif airy{enter}");
		await waitFor(() => expect(box("airy", "included")).toBeInTheDocument());
		expect(lastParams(calls).ignore).toBeUndefined();
		expect(window.location.search).toBe("?q=serif+airy");
	});

	test("choices come back from a shared link, including tags that are off", async () => {
		window.history.replaceState({}, "", "/?q=serif+bold&tags=-bold&ignore=serif");
		const calls = installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await within(await screen.findByRole("list", { name: "Tags in your search" })).findByText("bold");
		expect(lastParams(calls)).toMatchObject({ tags: "-bold", ignore: "serif" });
		expect(box("bold", "excluded")).toBeInTheDocument();
		expect(box("serif", "off")).toBeInTheDocument();               // not in the response, kept from the link

		window.history.pushState({}, "", "/?q=serif+bold");
		act(() => { window.dispatchEvent(new PopStateEvent("popstate")); });
		await waitFor(() => expect(box("serif", "included")).toBeInTheDocument());
		expect(box("bold", "included")).toBeInTheDocument();
	});

	test("words that match nothing and have no suggestions are listed as not recognised", async () => {
		installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("serif qwerty slimy");
		expect(await screen.findByText("Not recognised: qwerty.")).toBeInTheDocument();
		expect(within(line()).getByText("similar to “slimy”:")).toBeInTheDocument();
	});

	test("nothing recognised says so", async () => {
		installFetch({ "/api/font/query": backend() });
		render(<SearchPage username={null} />);
		await search("qwerty");
		expect(await screen.findByText("No tags recognised in that description.")).toBeInTheDocument();
		expect(screen.getByText("Not recognised: qwerty.")).toBeInTheDocument();
	});

	test("an older backend without inferred/suggested still renders", async () => {
		installFetch({ "/api/font/query": () => jsonResponse((({ inferred, suggested, ...rest }) => rest)(makePage({ total: 5 }))) });
		render(<SearchPage username={null} />);
		await search("serif");
		await screen.findAllByRole("img");
		expect(within(line()).getAllByRole("listitem")).toHaveLength(1);
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
		expect(calls[0].params).toMatchObject({ query: "script", page: "3" });
		expect(screen.getByLabelText("Describe a font")).toHaveValue("script");
	});

	test("a link past the last page lands on the last page", async () => {
		window.history.replaceState({}, "", "/?q=script&page=99");
		const calls = installFetch({ "/api/font/query": queryHandler(50) });
		render(<SearchPage username={null} />);
		await screen.findByText("Page 3 of 3 · 50 fonts");
		expect(calls.map(c => c.params.page)).toEqual(["99", "3"]);
	});

	test("garbage in the URL is ignored", async () => {
		window.history.replaceState({}, "", "/?q=script&page=abc");
		const calls = installFetch({ "/api/font/query": queryHandler(50) });
		render(<SearchPage username={null} />);
		await screen.findAllByRole("img");
		expect(calls[0].params.page).toBe("1");
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
		userEvent.click(within(firstCard()).getByRole("button", { name: "Rate 3 stars" }));
		expect(calls.filter(c => c.init.method === "POST")).toHaveLength(0);
	});

	test("approve, flip, and clear", async () => {
		const answers = [];
		const { calls } = await loaded("alice", { "/api/font/approve": ({ body }) => { answers.push(body.vote); return jsonResponse({ message: "Successful", vote: body.vote }); } });
		const yes = () => within(firstCard()).getByRole("button", { name: "This font matched my query" });
		const no = () => within(firstCard()).getByRole("button", { name: "This font did not match my query" });

		userEvent.click(yes());
		await waitFor(() => expect(yes()).toHaveAttribute("aria-pressed", "true"));
		expect(calls[calls.length - 1].body).toEqual({ fontKey: "google:Font 0", query: "serif", vote: 1 });

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

	test("rating shows the server's average and can be cleared", async () => {
		const sent = [];
		await loaded("alice", { "/api/font/rate": ({ body }) => {
			sent.push(body.rating);
			return jsonResponse({ message: "Successful", rating: body.rating ? { average: 4, count: 3, mine: body.rating } : { average: 4.5, count: 2, mine: null } });
		} });
		expect(within(firstCard()).getByText("unrated")).toBeInTheDocument();
		userEvent.click(within(firstCard()).getByRole("button", { name: "Rate 4 stars" }));
		await within(firstCard()).findByText("4 (3)");
		expect(within(firstCard()).getByRole("button", { name: "Rate 4 stars" })).toHaveAttribute("aria-pressed", "true");
		userEvent.click(within(firstCard()).getByRole("button", { name: "Rate 4 stars" }));
		await within(firstCard()).findByText("4.5 (2)");
		expect(sent).toEqual([4, 0]);
	});

	test("an expired session asks to log in again", async () => {
		const { onNeedLogin } = await loaded("alice", { "/api/font/rate": () => jsonResponse({ message: "Session expired" }, 401) });
		userEvent.click(within(firstCard()).getByRole("button", { name: "Rate 2 stars" }));
		await within(firstCard()).findByText("Your session ended, please log in again");
		expect(onNeedLogin).toHaveBeenCalled();
	});

	test("server and network errors are shown on the card", async () => {
		await loaded("alice", {
			"/api/font/approve": () => jsonResponse({ message: "Font not found" }, 404),
			"/api/font/rate": () => Promise.reject(new Error("offline")),
		});
		userEvent.click(within(firstCard()).getByRole("button", { name: "This font matched my query" }));
		await within(firstCard()).findByText("Font not found");
		userEvent.click(within(firstCard()).getByRole("button", { name: "Rate 5 stars" }));
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

	test("the source sits under the font name, and the rating above the thumbs", async () => {
		await loaded("alice");
		const card = firstCard();
		const title = card.querySelector(".ResultTitle");
		expect([...title.children].map(e => e.className || e.tagName)).toEqual(["A", "ResultSource"]);
		const feedback = card.querySelector(".ResultFeedback");
		expect([...feedback.children].map(e => e.className)).toEqual(["Stars", "Votes"]);
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
		const calls = installFetch({ "/api/font/query": () => jsonResponse({ ...makePage({ total: 1 }), results: [makeResult(0, { vote: calls.length > 1 ? 1 : 0 })] }) });
		const view = render(<SearchPage username={null} />);
		await search("serif");
		await screen.findAllByRole("img");
		expect(screen.getByRole("button", { name: "This font matched my query" })).toHaveAttribute("aria-pressed", "false");
		view.rerender(<SearchPage username="alice" />);
		await waitFor(() => expect(screen.getByRole("button", { name: "This font matched my query" })).toHaveAttribute("aria-pressed", "true"));
		expect(calls.filter(c => c.path === "/api/font/query")).toHaveLength(2);
	});
});
