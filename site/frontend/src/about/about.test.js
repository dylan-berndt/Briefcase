import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AboutPage, { renderMarkdown, slugify } from './index';

const text = (body) => () => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve(body) });

function serve(body, ok = true) {
	global.fetch = jest.fn(ok ? text(body) : () => Promise.resolve({ ok: false, status: 404, text: () => Promise.resolve("") }));
}

describe("renderMarkdown", () => {
	test("headings get ids, and ## and ### make the table of contents", () => {
		const { titleHtml, html, toc } = renderMarkdown("# Title\n\nintro\n\n## One\n\n### Sub part\n\n## Two");
		expect(toc).toEqual([
			{ id: "one", text: "One", depth: 2 },
			{ id: "sub-part", text: "Sub part", depth: 3 },
			{ id: "two", text: "Two", depth: 2 },
		]);
		expect(titleHtml).toContain('<h1 id="title">Title</h1>');
		expect(html).not.toContain("<h1");
		expect(html).toContain('<h3 id="sub-part">Sub part</h3>');
	});

	test("repeated headings get distinct ids", () => {
		const { toc } = renderMarkdown("## Same\n\n## Same\n\n## Same");
		expect(toc.map(t => t.id)).toEqual(["same", "same-2", "same-3"]);
		expect(slugify("Same", new Set(["same"]))).toBe("same-2");
	});

	test("formatting inside a heading is kept in the page and dropped from the contents", () => {
		const { html, toc } = renderMarkdown("## The `tag` *line*\n\n## B");
		expect(html).toContain("<code>tag</code>");
		expect(toc[0].text).toBe("The tag line");
		expect(toc[0].id).toBe("the-tag-line");
	});

	test("external links open in a new tab, internal ones do not", () => {
		const { html } = renderMarkdown("[out](https://example.com/a?b=1&c=2 \"Say \\\"hi\\\"\") and [in](#two)");
		expect(html).toMatch(/<a href="https:\/\/example.com\/a\?b=1&c=2"[^>]*target="_blank" rel="noopener noreferrer">out<\/a>/);
		expect(html).toContain('<a href="#two">in</a>');
	});
});

describe("About page", () => {
	const body = "# About\n\nHello **world**.\n\n## First\n\ntext\n\n### Nested\n\n## Second\n\nmore";

	test("renders the markdown file", async () => {
		serve(body);
		render(<AboutPage />);
		expect(await screen.findByRole("heading", { level: 1, name: "About" })).toBeInTheDocument();
		expect(screen.getByText("world").tagName).toBe("STRONG");
		expect(screen.getByRole("heading", { level: 2, name: "First" })).toHaveAttribute("id", "first");
		expect(global.fetch).toHaveBeenCalledTimes(1);
		expect(global.fetch.mock.calls[0][0]).toMatch(/about\.md$/);
	});

	test("a table of contents sits above the text and links to the headings", async () => {
		serve(body);
		render(<AboutPage />);
		const nav = await screen.findByRole("navigation", { name: "Table of contents" });
		const links = within(nav).getAllByRole("link");
		expect(links.map(l => l.textContent)).toEqual(["First", "Nested", "Second"]);
		expect(links.map(l => l.getAttribute("href"))).toEqual(["#first", "#nested", "#second"]);
		const title = screen.getByRole("heading", { level: 1 });
		const first = screen.getByRole("heading", { level: 2, name: "First" });
		expect(title.compareDocumentPosition(nav) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();    // title, then contents,
		expect(nav.compareDocumentPosition(first) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();    // then the text

		const scrolled = jest.fn();
		window.HTMLElement.prototype.scrollIntoView = scrolled;
		userEvent.click(within(nav).getByRole("link", { name: "Second" }));
		expect(scrolled).toHaveBeenCalledTimes(1);
		expect(scrolled.mock.instances[0]).toBe(document.getElementById("second"));
		expect(window.location.hash).toBe("");                 // the app's own address is left alone
	});

	test("no table of contents when there is nothing to list", async () => {
		serve("# Just a title\n\n## Only one section\n\ntext");
		render(<AboutPage />);
		await screen.findByRole("heading", { level: 2 });
		expect(screen.queryByRole("navigation")).toBeNull();
	});

	test("a failed load says so", async () => {
		serve("", false);
		render(<AboutPage />);
		expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the page");
	});

	test("the real about.md renders with a table of contents", async () => {
		const real = require("fs").readFileSync(require("path").join(__dirname, "about.md"), "utf8");
		serve(real);
		render(<AboutPage />);
		await waitFor(() => expect(screen.getByRole("navigation", { name: "Table of contents" })).toBeInTheDocument());
		expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
	});
});
