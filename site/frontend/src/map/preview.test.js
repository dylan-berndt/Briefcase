import { previewMap } from './index';

// A stand-in for the map's iframe: a graph div that records the handlers it is given, in the page's own document.
function fakeMap({ touch = false } = {}) {
	const handlers = {};
	const images = [];            // every picture the page starts to load, in order
	const gd = document.createElement("div");
	gd.data = [{ text: ["Abel Regular", "Alice Regular", "Akt Regular"] }];
	gd.on = (name, fn) => { handlers[name] = fn; };
	document.body.appendChild(gd);
	const win = Object.assign(new EventTarget(), {
		Plotly: { restyle: jest.fn() },
		matchMedia: () => ({ matches: touch }),
		innerWidth: 1000,
		innerHeight: 800,
	});
	const frame = { contentDocument: { querySelector: () => gd, createElement: (t) => { const e = document.createElement(t); if (t === "img") images.push(e); return e; }, body: document.body }, contentWindow: win };
	// as in Plotly's 3D hover events there is no mouse event, only the point; the pointer is moved separately
	const move = (x, y) => win.dispatchEvent(new MouseEvent("pointermove", { clientX: x, clientY: y }));
	const hover = (point) => handlers.plotly_hover({ points: [{ pointNumber: point }] });
	return { frame, gd, win, handlers, hover, move, images };
}

const tip = () => document.querySelector("body > div[style*='position: fixed']");

beforeEach(() => jest.useFakeTimers());
afterEach(() => {
	jest.useRealTimers();
	document.body.innerHTML = "";
});

test("Plotly's own label is turned off, and the hover events are kept", () => {
	const { frame, win, handlers } = fakeMap();
	expect(previewMap(frame, "routes")).toBe(true);
	expect(win.Plotly.restyle).toHaveBeenCalledWith(expect.anything(), { hoverinfo: "none", hovertemplate: null });
	expect(handlers.plotly_hover).toBeDefined();
});

test("a touch screen is left alone", () => {
	const { frame, win, handlers } = fakeMap({ touch: true });
	expect(previewMap(frame, "routes")).toBe(false);
	expect(win.Plotly.restyle).not.toHaveBeenCalled();
	expect(handlers.plotly_hover).toBeUndefined();
});

test("resting on a point shows that point's picture, after a short wait", () => {
	const { frame, hover, images } = fakeMap();
	previewMap(frame, "routes");
	hover(1);
	jest.advanceTimersByTime(149);
	expect(images).toHaveLength(0);                          // nothing has been asked for yet
	jest.advanceTimersByTime(2);
	expect(images).toHaveLength(1);
	const picture = images[0];
	expect(picture.getAttribute("src")).toBe("/api/map/routes/1.webp");
	expect(picture.alt).toBe("Alice Regular");
	expect(picture.style.height).toBe("22px");
	expect(tip().style.display).toBe("none");                // and the box stays hidden until the picture is there
	picture.onload();
	expect(tip().style.display).toBe("block");
	expect(tip().contains(picture)).toBe(true);
});

test("sweeping across points asks for the one the pointer stopped on, not the others", () => {
	const { frame, hover, images } = fakeMap();
	previewMap(frame, "routes");
	hover(0);
	jest.advanceTimersByTime(60);
	hover(1);
	jest.advanceTimersByTime(60);
	hover(2);
	jest.advanceTimersByTime(200);
	expect(images).toHaveLength(1);
	expect(images[0].getAttribute("src")).toBe("/api/map/routes/2.webp");
});

test("a point with no picture shows its plain label in the same box", () => {
	const { frame, hover, images } = fakeMap();
	previewMap(frame, "routes");
	hover(2);
	jest.advanceTimersByTime(200);
	images[0].onerror();
	expect(tip().style.display).toBe("block");
	expect(tip().textContent).toBe("Akt Regular");
	expect(tip().querySelector("img")).toBeNull();
});

test("a picture that arrives late for an earlier point is dropped", () => {
	const { frame, hover, handlers, images } = fakeMap();
	previewMap(frame, "routes");
	hover(0);
	jest.advanceTimersByTime(200);
	handlers.plotly_unhover();
	images[0].onload();
	expect(tip().style.display).toBe("none");
});

test("leaving a point, or grabbing the map to rotate it, hides the box", () => {
	const { frame, hover, handlers, win, images } = fakeMap();
	previewMap(frame, "routes");
	hover(0);
	jest.advanceTimersByTime(200);
	images[0].onload();
	handlers.plotly_unhover();
	expect(tip().style.display).toBe("none");

	hover(1);
	jest.advanceTimersByTime(200);
	images[1].onload();
	expect(tip().style.display).toBe("block");
	win.dispatchEvent(new MouseEvent("pointerdown"));
	expect(tip().style.display).toBe("none");
});

test("the box sits just below and right of the pointer, follows it, and stays inside the frame", () => {
	const { frame, hover, move, images } = fakeMap();
	previewMap(frame, "routes");
	move(100, 200);
	hover(0);
	jest.advanceTimersByTime(200);
	images[0].onload();
	expect([tip().style.left, tip().style.top]).toEqual(["114px", "214px"]);
	move(300, 250);
	expect([tip().style.left, tip().style.top]).toEqual(["314px", "264px"]);       // it follows
	move(999, 799);
	expect(parseFloat(tip().style.left)).toBeLessThanOrEqual(996);                  // and does not leave the frame
	expect(parseFloat(tip().style.top)).toBeLessThanOrEqual(796);
});
