import './main.css'
import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { fontPath } from '../font';

// Plotly leaves 80-100px of margin round a 3D plot, which on a phone is most of the width, and its default camera
// crops a portrait scene at the sides. The maps are same-origin, so the page can reach into the frame: zero the
// margins, and pull the camera back when the frame is narrower than it is tall. The camera is scaled relative to
// wherever the user has rotated it to, so resizing does not undo their view.
const PORTRAIT_ZOOM = 1.5;

export function fitMap(frame) {
    let win, doc;
    try {
        win = frame.contentWindow;
        doc = frame.contentDocument;
    } catch (e) {
        return false;
    }
    const gd = doc && doc.querySelector(".plotly-graph-div");
    if (!win || !win.Plotly || !gd || !gd._fullLayout || !gd._fullLayout.scene) {
        return false;
    }
    const want = frame.clientWidth < frame.clientHeight ? PORTRAIT_ZOOM : 1;
    const have = gd._briefcaseZoom || 1;
    const update = {};
    const m = gd._fullLayout.margin;
    if (m.l || m.r || m.t || m.b || m.pad) {
        update.margin = { l: 0, r: 0, t: 0, b: 0, pad: 0 };
    }
    if (want !== have) {
        const eye = gd._fullLayout.scene.camera.eye;
        const k = want / have;
        update["scene.camera.eye"] = { x: eye.x * k, y: eye.y * k, z: eye.z * k };
    }
    gd._briefcaseZoom = want;
    if (Object.keys(update).length) {
        win.Plotly.relayout(gd, update);
    }
    return true;
}

// A press is a click, not a drag to rotate, if the pointer never gets more than this many pixels from where it went down
// and the button is down no longer than this many milliseconds.
const CLICK_MOVE = 5;
const CLICK_HOLD = 500;
// a click Plotly reports this soon after the button came up still belongs to that press
const REPORTED_LATE = 250;

// Makes the points of the map in `frame` open font pages. The maps cannot be regenerated, so the page each point opens
// is listed beside the map (<map>.links.json, written by site/tools/linkMaps.py) in the order of the map's points; a
// point's number is its place in that list. A list that does not fit the map (the map was replaced and the list was
// not) is not used, rather than opening the wrong fonts. Returns whether the map was wired up.
export function linkMap(frame, slugs, onOpen) {
    let doc;
    try {
        doc = frame.contentDocument;
    } catch (e) {
        return false;
    }
    const gd = doc && doc.querySelector(".plotly-graph-div");
    if (!gd || !gd.on || !Array.isArray(gd.data)) return false;
    const trace = gd.data.find(t => Array.isArray(t.text));
    if (!trace || !Array.isArray(slugs) || slugs.length !== trace.text.length) return false;

    const slugOf = (event) => {
        const point = event && event.points && event.points[0];
        return point ? slugs[point.pointNumber ?? point.pointIndex] : null;
    };
    // Plotly reports a click as soon as the button goes down on a point, before it is known whether the user is clicking or
    // grabbing the map to rotate it. So the page opens when the button comes up, and only if the pointer stayed within a
    // few pixels the whole time and was not held. (A build that reports the click on release instead is handled too, and a
    // click with no press at all, from a script, opens at once.) The listeners are on the frame's window: once a drag
    // starts Plotly follows the pointer there, not on the graph.
    const win = doc.defaultView || doc;
    let press = null;       // while the button is down: where, when, and the farthest the pointer has been from there
    let waiting = null;     // the page Plotly reported during that press
    let released = null;    // the press that just ended: was it a click, and when
    const reach = (e) => { if (press) press.far = Math.max(press.far, Math.hypot(e.clientX - press.x, e.clientY - press.y)); };
    win.addEventListener("pointerdown", (e) => {
        press = { x: e.clientX, y: e.clientY, time: Date.now(), far: 0 };
        waiting = null;
        released = null;
    }, true);
    win.addEventListener("pointermove", reach, true);
    win.addEventListener("pointerup", (e) => {
        if (!press) return;
        reach(e);
        const ok = press.far <= CLICK_MOVE && Date.now() - press.time <= CLICK_HOLD;
        const slug = waiting;
        press = null;
        waiting = null;
        released = { ok, time: Date.now() };
        if (slug && ok) onOpen(slug);
    }, true);
    win.addEventListener("pointercancel", () => { press = null; waiting = null; }, true);
    gd.on("plotly_click", (event) => {
        const slug = slugOf(event);
        if (!slug) return;
        if (press) waiting = slug;
        else if (released && Date.now() - released.time < REPORTED_LATE) { if (released.ok) onOpen(slug); }
        else onOpen(slug);
    });
    // a pointer over the points that open something, so it is clear they can be clicked
    gd.on("plotly_hover", (event) => { gd.style.cursor = slugOf(event) ? "pointer" : ""; });
    gd.on("plotly_unhover", () => { gd.style.cursor = ""; });
    return true;
}

export default function MapPage() {
    const navigate = useNavigate();

    const [selectedMap, setSelectedMap] = useState("flower");
    const frameRef = useRef(null);

    // the plot is drawn by a script in the frame, which may not have finished when the frame reports it has loaded
    const handleLoad = () => {
        const frame = frameRef.current;
        if (!frame) return;
        const name = selectedMap;
        // the list of pages the points open; a map without one just is not clickable
        const links = fetch("/maps/" + name + ".links.json").then(r => r.ok ? r.json() : null).catch(() => null);
        let tries = 0;
        const attempt = () => {
            if (!frameRef.current) return;
            if (fitMap(frame)) {
                links.then(list => {
                    if (list && frameRef.current === frame) linkMap(frame, list.slugs, slug => navigate(fontPath(slug)));
                });
                return;
            }
            if (++tries > 50) return;
            setTimeout(attempt, 100);
        };
        attempt();
        try {
            frame.contentWindow.addEventListener("resize", () => fitMap(frame));
        } catch (e) { /* cross-origin: leave the map as it is */ }
    };

    const handleChange = (event) => {
        setSelectedMap(event.target.value);
    };

    return <div className="MapArea">
        <div className="MapDescription">
            <h>Font Maps</h>
            <br></br><br></br>
            <p>This is Font Maps, a section of the site dedicated to visualizing the relationships between fonts. 
                Essentially, each font in the dataset is encoded into a vector that represents the visual information 
                present in that font. Fonts that are close to each other in this space typically look similar, and ones
                that are far apart will look different. We can squish the vectors down into 6 dimensions with tSNE and show
                them as positions in a 3D space, with colors defined by the last 3 of the 6 dimensions. So, if two fonts have
                a similar color, they also share some other features with each other.
                <br></br><br></br>
                I've generated a couple different maps made up of the different font datasets I've used and all the different models I
                trained before completing the search project. The flower map is made up of the embeddings extracted from one of the first
                models I trained that was designed to take a letter and uppercase it. It also only uses fonts from the Google Fonts
                repository. The next map is routes, and it contains every font that is available to search on the website embedded with the
                latest model that I created using LeVJEPA. Click a point to open that font's page.</p>
            <br></br><br></br>
            <div height="4vh">
                <label htmlFor="options">Viewing: </label>
                <select id="options" value={selectedMap} onChange={handleChange}>
                    <option value="flower">Flower</option>
                    <option value="routes">Routes</option>
                </select>
            </div>
        </div>
        
        <iframe ref={frameRef} onLoad={handleLoad} title="mapLocation" src={"/maps/" + selectedMap + ".html"} style={{ border: "none" }}></iframe>
    </div>
}