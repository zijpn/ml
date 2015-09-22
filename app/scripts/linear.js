/*global d3, MathJax*/
'use strict';

var id = 1, converged, diverged, iteration,
    lr, theta0, theta1,
    svg, data;

var dataset1 = [
    { id: id++, values: { x: 1, y: 1 }},
    { id: id++, values: { x: 1.5, y: 2 }},
    { id: id++, values: { x: 0, y: 0.5 }},
    { id: id++, values: { x: 10, y: 9.5 }},
    { id: id++, values: { x: 8, y: 10 }},
    { id: id++, values: { x: 5, y: 3 }}
];

var dataset2 = [
    { id: id++, values: { x: 0, y: 0 }},
    { id: id++, values: { x: 1, y: 1 }},
    { id: id++, values: { x: 2, y: 2 }},
    { id: id++, values: { x: 3, y: 3 }},
    { id: id++, values: { x: 4, y: 4 }},
    { id: id++, values: { x: 10, y: 10 }}
];

function changeLr() {
    lr = +d3.select('#lr').node().value;
}

function changeTheta0() {
    theta0 = +d3.select('#theta0').node().value;
}

function changeTheta1() {
    theta1 = +d3.select('#theta1').node().value;
}

function getCoord(xs) {
    return xs.map(function(x) { return {x: x, y: theta1 * x + theta0}; });
}

function updateGraph() {
    var width = 600;
    var height = 600;

    var xscale = d3.scale.linear()
            .domain(d3.extent(data, function(d) { return d.values.x; }))
            .range([0, width]).nice(),
        yscale = d3.scale.linear()
            .domain(d3.extent(data, function(d) { return d.values.y; }))
            .range([height, 0]).nice();

    if(svg.select('g.x.axis').empty()) {
        svg.append('g').attr('class', 'x axis')
            .attr('transform', 'translate(0,' + height + ')');
        svg.append('g').attr('class', 'y axis');
    }

    svg.select('g.x.axis').call(d3.svg.axis().scale(xscale).orient('bottom'));
    svg.select('g.y.axis').call(d3.svg.axis().scale(yscale).orient('left'));

    var points = svg.selectAll('circle').data(data);
    points.enter().append('circle')
        .attr('r', 7)
        .style('fill', 'steelblue');

    points.attr('cx', function(d) { return xscale(d.values.x); })
        .attr('cy', function(d) { return yscale(d.values.y); })
        .on('mouseover', function(d) {
            var pred = theta1 * d.values.x + theta0;
            var p = svg.append('g').attr('class', 'pointer');
            p.append('line')
                .attr('marker-end', 'url(#arrow)')
                .style('stroke', 'steelblue')
                .style('stroke-width', 2)
                .attr('x1', xscale(d.values.x))
                .attr('y1', yscale(d.values.y))
                .attr('x2', xscale(d.values.x))
                .attr('y2', yscale(pred));
            p.append('text')
                .attr('text-anchor', 'middle')
                .attr('x', xscale(d.values.x))
                .attr('y', yscale(d.values.y) + (pred > d.values.y ? 25 : -15))
                .text('' + d3.round(d.values.y - pred, 4));
        })
        .on('mouseout', function() {
            svg.select('.pointer').remove();
        });

    points.exit().remove();

    var line = d3.svg.line()
                   .x(function(d) { return xscale(d.x); })
                   .y(function(d) { return yscale(d.y); });
    var path = svg.selectAll('path.reg').data([getCoord(xscale.ticks(100))]);
    path.enter().append('path').attr('class', 'reg');
    path.attr('d', line);
    path.exit().remove();
}

function update(sse)
{
    d3.select('#values')
        .html('Iteration = ' + iteration + (converged ? ' (converged)' : (diverged ? ' (diverged)' : '')) + '<br/>' +
              '\\(\\theta_0 = ' + d3.round(theta0, 4) + '\\)<br/>' +
              '\\(\\theta_1 = ' + d3.round(theta1, 4) + '\\)<br/>' +
              '\\(SSE = ' + d3.round(sse, 2) + '\\)<br/>');
    /*eslint-disable new-cap*/
    if (MathJax.Hub) {
      MathJax.Hub.Queue(['Typeset', MathJax.Hub, 'values']);
    }
    /*eslint-enable new-cap*/
    updateGraph();
}

function ginit() {
    changeLr();
    changeTheta0();
    changeTheta1();
    converged = false;
    diverged = false;
    iteration = 0;
    var sse = d3.sum(data, function(d) { return Math.pow(d.values.y - (theta1 * d.values.x + theta0), 2); });
    update(sse);
    d3.select('#step').attr('disabled', null);
    d3.select('#run').attr('disabled', null);
}

function gstep() {
    if (converged || diverged) {
       d3.select('#step').attr('disabled', 'true');
       d3.select('#run').attr('disabled', 'true');
       return;
    }

    iteration++;
    var theta0grad = d3.sum(data, function(d) { return (d.values.y - (theta1 * d.values.x + theta0)); });
    var theta0p = theta0 + lr * 2 * theta0grad;
    var theta1grad = d3.sum(data, function(d) { return (d.values.y - (theta1 * d.values.x + theta0)) * d.values.x; });
    var theta1p = theta1 + lr * 2 * theta1grad;

    var dtheta0 = Math.abs(theta0p - theta0);
    var dtheta1 = Math.abs(theta1p - theta1);

    var convergence = 1e-6;
    if (dtheta0 < convergence && dtheta1 < convergence) {
        converged = true;
    }
    var divergence = 1e+6;
    if (dtheta0 > divergence || dtheta1 > divergence) {
        diverged = true;
    }

    theta0 = theta0p;
    theta1 = theta1p;
    var sse = d3.sum(data, function(d) { return Math.pow(d.values.y - (theta1 * d.values.x + theta0), 2); });
    update(sse);
}

/*eslint-disable no-unused-vars*/
function grun() {
/*eslint-enable no-unused-vars*/
    while (!converged && !diverged) {
        gstep();
    }
    // disable the buttons
    gstep();
}

/*eslint-disable no-unused-vars*/
function changeDs() {
/*eslint-enable no-unused-vars*/
    var ds = document.getElementById('dataset').value;
    if (ds === 'data1') {
        data = dataset1;
    }
    else if (ds === 'data2') {
        data = dataset2;
    }
    ginit();
}

(function() {
    data = dataset1;

    svg = d3.select('#graph')
        .append('svg')
        .attr('preserveAspectRatio', 'xMinYMin meet')
        .attr('viewBox', '0 0 700 700')
        .append('g').attr('transform', 'translate(35,35)');

    svg.append('defs')
       .append('marker')
       .attr('id', 'arrow')
       .attr('viewBox', '0 -5 10 10')
       .attr('refX', 10)
       .attr('markerWidth', 6)
       .attr('markerHeight', 6)
       .attr('orient', 'auto')
       .append('path')
       .style('fill', 'steelblue')
       .style('stroke', 'none')
       .attr('d', 'M0,-5L10,0L0,5');

    ginit();
})();
