import './main.css'
import React, { useState, useRef, useEffect } from 'react';

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

export default function MapPage() {

    const [selectedMap, setSelectedMap] = useState("flower");
    const frameRef = useRef(null);

    // the plot is drawn by a script in the frame, which may not have finished when the frame reports it has loaded
    const handleLoad = () => {
        const frame = frameRef.current;
        if (!frame) return;
        let tries = 0;
        const attempt = () => {
            if (!frameRef.current || fitMap(frame) || ++tries > 50) return;
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
                latest model that I created using LeVJEPA. </p>
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