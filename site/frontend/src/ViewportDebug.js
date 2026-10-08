import React, { useEffect, useState } from 'react';

// Only rendered with ?debug in the address: a readout of what the browser says about the viewport and the background
// shader's box, for debugging layouts on devices without a console (iOS Safari). "tint" paints the shader's box.
const unit = (value) => {
	const probe = document.createElement("div");
	probe.style.cssText = "position:fixed;left:0;top:0;width:1px;visibility:hidden;height:" + value;
	document.body.appendChild(probe);
	const h = probe.getBoundingClientRect().height;
	probe.remove();
	return Math.round(h * 10) / 10;
};

const box = (element) => {
	if (!element) return "none";
	const r = element.getBoundingClientRect();
	return "top " + Math.round(r.top) + " bottom " + Math.round(r.bottom) + " h " + Math.round(r.height);
};

export default function ViewportDebug() {
	const [text, setText] = useState("");
	const [tint, setTint] = useState(false);

	useEffect(() => {
		const update = () => {
			const shader = document.querySelector(".Shader");
			const canvas = shader && shader.querySelector("canvas");
			const style = shader && getComputedStyle(shader);
			const vv = window.visualViewport;
			setText([
				"inner " + window.innerWidth + "x" + window.innerHeight + "  screen " + window.screen.width + "x" + window.screen.height,
				"visualViewport h " + (vv ? Math.round(vv.height * 10) / 10 + " top " + Math.round(vv.offsetTop) : "n/a"),
				"svh " + unit("100svh") + "  lvh " + unit("100lvh") + "  dvh " + unit("100dvh"),
				"scrollY " + Math.round(window.scrollY) + "  doc h " + document.documentElement.scrollHeight,
				"shader " + box(shader),
				"  css bottom " + (style && style.bottom) + " h " + (style && style.height) + " bg " + (style && style.backgroundColor),
				"canvas " + box(canvas) + "  buffer " + (canvas ? canvas.width + "x" + canvas.height : "-"),
			].join("\n"));
		};
		update();
		const id = setInterval(update, 300);
		return () => clearInterval(id);
	}, []);

	return (
		<>
			{tint && <style>{".Shader { background: #0dd !important; } .Shader canvas { opacity: 0.5; }"}</style>}
			<div style={{ position: "fixed", top: 60, left: 0, right: 0, zIndex: 9999, background: "rgba(0,0,0,.85)", color: "#fff",
				font: "11px/1.25 ui-monospace, Menlo, monospace", whiteSpace: "pre", padding: "4px 8px" }}>
				{text}
				<div>
					<button type="button" onClick={() => setTint(!tint)} style={{ font: "inherit", padding: "3px 6px" }}>
						tint shader box: {tint ? "on" : "off"}
					</button>
				</div>
			</div>
		</>
	);
}
