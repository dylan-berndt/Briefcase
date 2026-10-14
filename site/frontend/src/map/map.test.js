import { linkMap } from './index';

// A stand-in for the map's iframe: a Plotly graph div that records the handlers it is given.
function fakeFrame(labels) {
	const handlers = {};
	const gd = document.createElement("div");
	gd.data = [{ text: labels }];
	gd.on = (name, fn) => { handlers[name] = fn; };
	// Plotly follows a drag on the frame's window, so that is where the pointer events are heard
	const win = new EventTarget();
	const frame = { contentDocument: { querySelector: () => gd, defaultView: win } };
	const pointer = (type, x, y) => win.dispatchEvent(new MouseEvent(type, { clientX: x, clientY: y }));
	return { frame, gd, handlers, pointer };
}

test('a clicked point opens the page listed for its number', () => {
	const { frame, handlers } = fakeFrame(['a', 'b', 'c']);
	const opened = [];
	expect(linkMap(frame, ['font-a', null, 'font-c'], slug => opened.push(slug))).toBe(true);
	handlers.plotly_click({ points: [{ pointNumber: 2 }] });
	handlers.plotly_click({ points: [{ pointNumber: 1 }] });     // no page for this one
	handlers.plotly_click({ points: [] });
	expect(opened).toEqual(['font-c']);
});

test('the pointer shows only over points that open something', () => {
	const { frame, gd, handlers } = fakeFrame(['a', 'b']);
	linkMap(frame, ['font-a', null], () => { });
	handlers.plotly_hover({ points: [{ pointNumber: 0 }] });
	expect(gd.style.cursor).toBe('pointer');
	handlers.plotly_hover({ points: [{ pointNumber: 1 }] });
	expect(gd.style.cursor).toBe('');
});

test('a list that does not fit the map is not used', () => {
	const { frame, handlers } = fakeFrame(['a', 'b', 'c']);
	expect(linkMap(frame, ['font-a'], () => { })).toBe(false);
	expect(linkMap(frame, null, () => { })).toBe(false);
	expect(handlers.plotly_click).toBeUndefined();
});

describe('dragging to rotate is not a click', () => {
	const click = { points: [{ pointNumber: 0 }] };
	let now;

	beforeEach(() => {
		now = 1000;
		jest.spyOn(Date, 'now').mockImplementation(() => now);
	});
	afterEach(() => jest.restoreAllMocks());

	function setup() {
		const opened = [];
		const map = fakeFrame(['a']);
		linkMap(map.frame, ['font-a'], slug => opened.push(slug));
		return { ...map, opened, wait: (ms) => { now += ms; } };
	}

	// Plotly reports the click the moment the button goes down on a point
	describe('reported on the press', () => {
		test('a quick press that stays put opens the page when the button comes up, not before', () => {
			const m = setup();
			m.pointer('pointerdown', 100, 100);
			m.handlers.plotly_click(click);
			expect(m.opened).toEqual([]);
			m.wait(100);
			m.pointer('pointerup', 101, 100);
			expect(m.opened).toEqual(['font-a']);
		});

		test('a press that rotates the map never opens it', () => {
			const m = setup();
			m.pointer('pointerdown', 100, 100);
			m.handlers.plotly_click(click);
			m.pointer('pointermove', 140, 120);
			m.pointer('pointermove', 220, 150);
			m.wait(300);
			m.pointer('pointerup', 220, 150);
			expect(m.opened).toEqual([]);
		});

		test('nor does a drag that comes back to where it started', () => {
			const m = setup();
			m.pointer('pointerdown', 100, 100);
			m.handlers.plotly_click(click);
			m.pointer('pointermove', 200, 100);
			m.pointer('pointermove', 100, 100);
			m.pointer('pointerup', 100, 100);
			expect(m.opened).toEqual([]);
		});

		test('a press that is held down does not open it', () => {
			const m = setup();
			m.pointer('pointerdown', 100, 100);
			m.handlers.plotly_click(click);
			m.wait(900);
			m.pointer('pointerup', 100, 100);
			expect(m.opened).toEqual([]);
		});

		test('a cancelled press does not open it', () => {
			const m = setup();
			m.pointer('pointerdown', 100, 100);
			m.handlers.plotly_click(click);
			m.pointer('pointercancel', 100, 100);
			m.pointer('pointerup', 100, 100);
			expect(m.opened).toEqual([]);
		});

		test('the next click is judged on its own press, not the drag before it', () => {
			const m = setup();
			m.pointer('pointerdown', 10, 10);
			m.handlers.plotly_click(click);
			m.pointer('pointermove', 300, 300);
			m.pointer('pointerup', 300, 300);
			m.wait(1000);
			m.pointer('pointerdown', 50, 50);
			m.handlers.plotly_click(click);
			m.pointer('pointerup', 50, 50);
			expect(m.opened).toEqual(['font-a']);
		});
	});

	// ...or, in a build that does it the other way, as the button comes up
	describe('reported on the release', () => {
		test('a quick press opens the page, a drag does not', () => {
			const quick = setup();
			quick.pointer('pointerdown', 100, 100);
			quick.pointer('pointerup', 100, 100);
			quick.handlers.plotly_click(click);
			expect(quick.opened).toEqual(['font-a']);

			const drag = setup();
			drag.pointer('pointerdown', 100, 100);
			drag.pointer('pointermove', 220, 150);
			drag.pointer('pointerup', 220, 150);
			drag.handlers.plotly_click(click);
			expect(drag.opened).toEqual([]);
		});

		test('a click long after the last press is a script, not that press', () => {
			const m = setup();
			m.pointer('pointerdown', 10, 10);
			m.pointer('pointermove', 300, 300);
			m.pointer('pointerup', 300, 300);
			m.wait(5000);
			m.handlers.plotly_click(click);
			expect(m.opened).toEqual(['font-a']);
		});
	});

	test('a click with no press at all opens the page at once', () => {
		const m = setup();
		m.handlers.plotly_click(click);
		expect(m.opened).toEqual(['font-a']);
	});
});
