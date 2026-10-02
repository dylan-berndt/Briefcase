import { render, screen, waitFor, within, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { installFetch, jsonResponse, makePage } from './testUtils';

jest.mock('@react-three/fiber', () => ({ extend: jest.fn(), useFrame: jest.fn(), Canvas: () => null }));
jest.mock('@react-three/drei', () => ({ shaderMaterial: () => function Material() { return null; } }));

beforeEach(() => window.history.replaceState({}, "", "/"));

// the real router, on whatever address the test has set
function renderApp(path) {
	if (path) window.history.replaceState({}, "", path);
	return render(<BrowserRouter><App /></BrowserRouter>);
}

function fill(username, password) {
	userEvent.type(screen.getByLabelText("Username:"), username);
	userEvent.type(screen.getByLabelText("Password:"), password);
}

test("anonymous visitor sees the search page and a login button", async () => {
	installFetch({ "/api/font/me": () => jsonResponse({ username: null }) });
	renderApp();
	expect(screen.getByRole("button", { name: "Login" })).toBeInTheDocument();
	expect(screen.getByLabelText("Describe a font")).toBeInTheDocument();
});

test("an existing session is picked up on load", async () => {
	installFetch({ "/api/font/me": () => jsonResponse({ username: "alice" }) });
	renderApp();
	expect(await screen.findByRole("button", { name: "alice" })).toBeInTheDocument();
});

test("login, then log out", async () => {
	const calls = installFetch({
		"/api/font/me": () => jsonResponse({ username: null }),
		"/api/font/login": () => jsonResponse({ message: "Logged in successfully", username: "alice" }),
		"/api/font/logout": () => jsonResponse({ message: "Logged out" }),
	});
	renderApp();
	userEvent.click(screen.getByRole("button", { name: "Login" }));
	fill("alice", "correct horse");
	userEvent.click(screen.getByRole("button", { name: "Submit" }));

	const loggedIn = await screen.findByRole("button", { name: "alice" });
	expect(calls.map(c => c.path)).toEqual(["/api/font/me", "/api/font/login"]);
	expect(Object.fromEntries(calls[1].body)).toEqual({ username: "alice", password: "correct horse" });

	userEvent.click(loggedIn);
	userEvent.click(screen.getByRole("button", { name: "Log out" }));
	expect(await screen.findByRole("button", { name: "Login" })).toBeInTheDocument();
	expect(calls[calls.length - 1].path).toBe("/api/font/logout");
});

test("a failed login shows the message and stays logged out", async () => {
	installFetch({
		"/api/font/me": () => jsonResponse({ username: null }),
		"/api/font/login": () => jsonResponse({ message: "Invalid username or password." }, 400),
	});
	renderApp();
	userEvent.click(screen.getByRole("button", { name: "Login" }));
	fill("alice", "nope nope nope");
	userEvent.click(screen.getByRole("button", { name: "Submit" }));
	expect(await screen.findByText("Invalid username or password.")).toBeInTheDocument();
	expect(screen.queryByRole("button", { name: "alice" })).toBeNull();
});

test("register creates the account first, then logs in", async () => {
	const order = [];
	installFetch({
		"/api/font/me": () => jsonResponse({ username: null }),
		"/api/font/register": () => { order.push("register"); return jsonResponse({ message: "Registered successfully" }); },
		"/api/font/login": () => { order.push("login"); return jsonResponse({ message: "Logged in successfully", username: "alice" }); },
	});
	renderApp();
	userEvent.click(screen.getByRole("button", { name: "Login" }));
	userEvent.click(screen.getByRole("button", { name: "Register" }));
	fill("alice", "correct horse");
	userEvent.click(screen.getByRole("button", { name: "Submit" }));
	await screen.findByRole("button", { name: "alice" });
	expect(order).toEqual(["register", "login"]);
});

test("a failed registration does not attempt to log in", async () => {
	const calls = installFetch({
		"/api/font/me": () => jsonResponse({ username: null }),
		"/api/font/register": () => jsonResponse({ message: "User already exists. Please login." }, 400),
	});
	renderApp();
	userEvent.click(screen.getByRole("button", { name: "Login" }));
	userEvent.click(screen.getByRole("button", { name: "Register" }));
	fill("alice", "correct horse");
	userEvent.click(screen.getByRole("button", { name: "Submit" }));
	expect(await screen.findByText("User already exists. Please login.")).toBeInTheDocument();
	expect(calls.map(c => c.path)).not.toContain("/api/font/login");
});

test("a vote while logged out opens the login box", async () => {
	installFetch({
		"/api/font/me": () => jsonResponse({ username: null }),
		"/api/font/query": () => jsonResponse(require('./testUtils').makePage({ total: 3 })),
	});
	renderApp();
	userEvent.type(screen.getByLabelText("Describe a font"), "serif{enter}");
	const yes = (await screen.findAllByRole("button", { name: "This font matched my query" }))[0];
	userEvent.click(yes);
	expect(await screen.findByLabelText("Username:")).toBeInTheDocument();
});


describe("pages are links", () => {
	const md = () => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve("# About text\n\n## One\n\nx\n\n## Two\n\ny") });
	const routes = () => installFetch({
		"/api/font/me": () => jsonResponse({ username: null }),
		"/about.md": md,
		"/api/font/query": ({ params }) => jsonResponse(makePage({ query: params.query, total: 30 })),
	});
	const nav = () => within(screen.getByRole("banner"));

	test("the header has real links to each page", () => {
		routes();
		renderApp();
		expect(nav().getByRole("link", { name: "Home" })).toHaveAttribute("href", "/");
		expect(nav().getByRole("link", { name: "Maps" })).toHaveAttribute("href", "/map");
		expect(nav().getByRole("link", { name: "About" })).toHaveAttribute("href", "/about");
	});

	test("a direct visit to a page shows that page", async () => {
		routes();
		renderApp("/about");
		expect(await screen.findByRole("heading", { name: "About text" })).toBeInTheDocument();
		expect(screen.queryByLabelText("Describe a font")).toBeNull();
		expect(nav().getByRole("link", { name: "About" })).toHaveClass("active");
	});

	test("the map page has its own address", () => {
		routes();
		renderApp("/map");
		expect(screen.getByTitle("mapLocation")).toBeInTheDocument();
		expect(nav().getByRole("link", { name: "Maps" })).toHaveClass("active");
	});

	test("clicking a link changes the address, and back and forward follow", async () => {
		routes();
		renderApp();
		userEvent.click(nav().getByRole("link", { name: "About" }));
		expect(await screen.findByRole("heading", { name: "About text" })).toBeInTheDocument();
		expect(window.location.pathname).toBe("/about");

		userEvent.click(nav().getByRole("link", { name: "Maps" }));
		expect(await screen.findByTitle("mapLocation")).toBeInTheDocument();
		expect(window.location.pathname).toBe("/map");

		act(() => window.history.back());
		expect(await screen.findByRole("heading", { name: "About text" })).toBeInTheDocument();
		userEvent.click(nav().getByRole("link", { name: "Home" }));
		expect(await screen.findByLabelText("Describe a font")).toBeInTheDocument();
		expect(window.location.pathname).toBe("/");
	});

	test("an unknown address goes to the search page", async () => {
		routes();
		renderApp("/no/such/page");
		expect(await screen.findByLabelText("Describe a font")).toBeInTheDocument();
		expect(window.location.pathname).toBe("/");
	});

	test("Home on the search page leaves the search and its ?q alone", async () => {
		routes();
		renderApp("/?q=serif");
		await screen.findAllByRole("img");
		userEvent.click(nav().getByRole("link", { name: "Home" }));
		expect(window.location.search).toBe("?q=serif");
		expect(screen.getByLabelText("Describe a font")).toHaveValue("serif");
	});

	test("a search's ?q survives going to another page and back with the browser buttons", async () => {
		routes();
		renderApp("/?q=serif");
		await screen.findAllByRole("img");
		userEvent.click(nav().getByRole("link", { name: "About" }));
		await screen.findByRole("heading", { name: "About text" });
		act(() => window.history.back());
		expect(await screen.findByLabelText("Describe a font")).toHaveValue("serif");
		expect(window.location.search).toBe("?q=serif");
	});
});
