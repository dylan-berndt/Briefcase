import { linkMap } from './index';

// A stand-in for the map's iframe: a Plotly graph div that records the handlers it is given.
function fakeFrame(labels) {
	const handlers = {};
	const gd = { data: [{ text: labels }], style: {}, on: (name, fn) => { handlers[name] = fn; } };
	const frame = { contentDocument: { querySelector: () => gd } };
	return { frame, gd, handlers };
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
