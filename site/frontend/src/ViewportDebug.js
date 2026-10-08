import React, { useEffect, useState } from 'react';

// Only rendered with ?debug in the address: a readout of what the browser says about the viewport, the column and the
// background's box, for debugging layouts on devices without a console (iOS Safari).
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

	useEffect(() => {
		const update = () => {
			const shader = document.querySelector(".Shader");
			const column = document.querySelector(".Shadow");
			const canvas = shader && shader.querySelector("canvas");
			const vv = window.visualViewport;
			setText([
				"inner " + window.innerWidth + "x" + window.innerHeight + "  screen " + window.screen.width + "x" + window.screen.height,
				"visualViewport h " + (vv ? Math.round(vv.height * 10) / 10 + " top " + Math.round(vv.offsetTop) : "n/a"),
				"svh " + unit("100svh") + "  lvh " + unit("100lvh") + "  dvh " + unit("100dvh"),
				"window scrollY " + Math.round(window.scrollY) + "  doc h " + document.documentElement.scrollHeight,
				"column " + box(column) + "  scrollTop " + (column ? Math.round(column.scrollTop) + "/" + column.scrollHeight : "-"),
				"shader " + box(shader),
				"canvas " + box(canvas) + "  buffer " + (canvas ? canvas.width + "x" + canvas.height : "-"),
			].join("\n"));
		};
		update();
		const id = setInterval(update, 300);
		return () => clearInterval(id);
	}, []);

	return (
		<div style={{ position: "fixed", top: 60, left: 0, right: 0, zIndex: 9999, background: "rgba(0,0,0,.85)", color: "#fff",
			font: "11px/1.25 ui-monospace, Menlo, monospace", whiteSpace: "pre", padding: "4px 8px", pointerEvents: "none" }}>
			{text}
		</div>
	);
}
