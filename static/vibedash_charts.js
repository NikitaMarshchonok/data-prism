(() => {
    "use strict";

    const resizeChart = (container) => {
        const graph = container.querySelector(".plotly-graph-div");
        if (!graph || !window.Plotly || !window.Plotly.Plots) {
            return;
        }
        window.requestAnimationFrame(() => window.Plotly.Plots.resize(graph));
    };

    const initialiseCharts = () => {
        const containers = document.querySelectorAll(".chart-container");
        containers.forEach(resizeChart);

        if (!("ResizeObserver" in window)) {
            return;
        }
        const observer = new ResizeObserver((entries) => {
            entries.forEach((entry) => resizeChart(entry.target));
        });
        containers.forEach((container) => observer.observe(container));
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialiseCharts, {once: true});
    } else {
        initialiseCharts();
    }
})();
