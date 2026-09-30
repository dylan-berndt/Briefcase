import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import { installFetch, jsonResponse } from './testUtils';

jest.mock('@react-three/fiber', () => ({ extend: jest.fn(), useFrame: jest.fn(), Canvas: () => null }));
jest.mock('@react-three/drei', () => ({ shaderMaterial: () => function Material() { return null; } }));

beforeEach(() => window.history.replaceState({}, "", "/"));

function fill(username, password) {
	userEvent.type(screen.getByLabelText("Username:"), username);
	userEvent.type(screen.getByLabelText("Password:"), password);
}

test("anonymous visitor sees the search page and a login button", async () => {
	installFetch({ "/api/font/me": () => jsonResponse({ username: null }) });
	render(<App />);
	expect(screen.getByRole("button", { name: "Login" })).toBeInTheDocument();
	expect(screen.getByLabelText("Describe a font")).toBeInTheDocument();
});

test("an existing session is picked up on load", async () => {
	installFetch({ "/api/font/me": () => jsonResponse({ username: "alice" }) });
	render(<App />);
	expect(await screen.findByRole("button", { name: "alice" })).toBeInTheDocument();
});

test("login, then log out", async () => {
	const calls = installFetch({
		"/api/font/me": () => jsonResponse({ username: null }),
		"/api/font/login": () => jsonResponse({ message: "Logged in successfully", username: "alice" }),
		"/api/font/logout": () => jsonResponse({ message: "Logged out" }),
	});
	render(<App />);
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
	render(<App />);
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
	render(<App />);
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
	render(<App />);
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
	render(<App />);
	userEvent.type(screen.getByLabelText("Describe a font"), "serif");
	userEvent.click(screen.getByRole("button", { name: "Search" }));
	const yes = (await screen.findAllByRole("button", { name: "This font matches my search" }))[0];
	userEvent.click(yes);
	expect(await screen.findByLabelText("Username:")).toBeInTheDocument();
});
