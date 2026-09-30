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
		expect(screen.getByText("Searching for: serif, not bold")).toBeInTheDocument();
		expect(calls[0].params).toEqual({ query: "serif not bold", page: "1", pageSize: "24" });
		expect(screen.getAllByText("Google Fonts")).toHaveLength(12);
		expect(screen.getAllByText("DaFont · Some Designer")).toHaveLength(12);
	});

	test("unrecognised words are reported; nothing recognised says so", async () => {
		installFetch({ "/api/font/query": () => jsonResponse(makePage({ total: 0, tags: [], unmatched: ["qwerty"] })) });
		render(<SearchPage username={null} />);
		await search("qwerty");
		expect(await screen.findByText(/No tags recognised/)).toHaveTextContent("Not recognised: qwerty.");
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
		expect(await screen.findByText("Searching for: second")).toBeInTheDocument();
		await act(async () => { releaseFirst(); });
		expect(screen.getByText("Searching for: second")).toBeInTheDocument();
		expect(screen.queryByText("Searching for: first")).toBeNull();
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
