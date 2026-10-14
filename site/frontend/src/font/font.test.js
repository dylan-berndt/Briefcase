import { render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import FontPage, { fontPath, slugFromPath } from './index';
import { installFetch, jsonResponse } from '../testUtils';

const info = (overrides = {}) => ({
	slug: "Abril Fatface", key: "google:Abril Fatface", name: "Abril Fatface", source: "Google Fonts", creator: null,
	url: "https://fonts.google.com/specimen/Abril+Fatface", path: "/font/Abril%20Fatface",
	specimen: "/specimen/Abril%20Fatface.webp",
	tags: ["serif", "didone"], closeTags: ["classic", "vintage"], query: "serif didone",
	related: [{ slug: "Ganton", name: "Ganton", source: "DaFont", path: "/font/Ganton", specimen: "/specimen/Ganton.webp" }],
	about: ["First paragraph.", "Second paragraph."], title: "Abril Fatface font", description: "d",
	...overrides,
});

function open(address) {
	return render(<MemoryRouter initialEntries={[address]}>
		<Routes><Route path="/font/*" element={<FontPage />} /></Routes>
	</MemoryRouter>);
}

describe("addresses", () => {
	test("a name is encoded into the address and decoded back, slashes included", () => {
		for (const name of ["Abril Fatface", "13/5 Atom Sans", "What?#&", "Åkerbär"]) {
			expect(slugFromPath(fontPath(name))).toBe(name);
		}
		expect(fontPath("13/5 Atom")).toBe("/font/13%2F5%20Atom");
	});
});

describe("font page", () => {
	test("shows the font, its tags and close tags, and the similar fonts", async () => {
		installFetch({ "/api/font/info/Abril%20Fatface": () => jsonResponse(info()) });
		open("/font/Abril%20Fatface");

		const title = await screen.findByRole("heading", { level: 1 });
		const host = within(title).getByRole("link");
		expect(host).toHaveAttribute("href", "https://fonts.google.com/specimen/Abril+Fatface");
		expect(host).toHaveAttribute("target", "_blank");
		expect(host.getAttribute("rel")).toContain("noopener");
		expect(host.textContent).toBe("Abril Fatface");
		expect(host.querySelector("svg")).not.toBeNull();                   // the link-out icon is inside the link

		expect(screen.getByAltText("Abril Fatface specimen")).toHaveAttribute("src", "/specimen/Abril%20Fatface.webp");
		for (const tag of ["serif", "didone", "classic", "vintage"]) {
			expect(screen.getByRole("link", { name: tag })).toHaveAttribute("href", "/?q=" + tag);
		}
		expect(screen.getByRole("link", { name: /Ganton/ })).toHaveAttribute("href", "/font/Ganton");
		expect(screen.getByRole("link", { name: "Search for more fonts like this" })).toHaveAttribute("href", "/?q=serif%20didone");
	});

	test("asks for the font by its decoded name, even when it holds a slash", async () => {
		const calls = installFetch({ "/api/font/info/13%2F5%20Atom": () => jsonResponse(info({ name: "13/5 Atom" })) });
		open("/font/13%2F5%20Atom");
		await screen.findByRole("heading", { level: 1 });
		expect(calls[0].path).toBe("/api/font/info/13%2F5%20Atom");
	});

	test("an unknown font says so and links home", async () => {
		installFetch({ "/api/font/info/Nothing": () => jsonResponse({ message: "Not found" }, 404) });
		open("/font/Nothing");
		const home = await screen.findByRole("link", { name: /search for one/i });
		expect(home).toHaveAttribute("href", "/");
		expect(screen.queryByRole("heading", { level: 1 })).toBeNull();
	});

	test("the tab title is the font's, and goes back when the page is left", async () => {
		installFetch({ "/api/font/info/Abril%20Fatface": () => jsonResponse(info()) });
		document.title = "Font Search";
		const { unmount } = open("/font/Abril%20Fatface");
		await screen.findByRole("heading", { level: 1 });
		expect(document.title).toBe("Abril Fatface font");
		unmount();
		expect(document.title).toBe("Font Search");
	});
});
