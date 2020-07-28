"use strict";

exports.__esModule = true;
var py = require("../dist/es5");
var dagre = require("dagre");
var Graph = require("graphlib").Graph;
var graphing = require("./Graph.js").Graph;
var dagreD3 = require("dagre-d3");
var d3 = require("d3");

var fs = require('fs');
var args = process.argv.slice(2);
var name = args[0];
var countLines = 0;
console.log(name);

//const path = './';
//const name = 'fold.py'
var contents = fs.readFileSync(name);
let jsondata = JSON.parse(contents);
//console.log(jsondata);

console.log("-----------------------------");

function readCode(jsondata) {
    var notebookCode = "\n";
    const rewriter = new py.MagicsRewriter();

    for (let cell of jsondata['cells']) {
        if (cell['cell_type'] == 'code') {
            var sourceCode = "";
            for (let line of cell['source']) {
                countLines += 1;
                if (line[0] == '%') {
                    line = rewriter.rewriteLineMagic(line);
                    //console.log("magic: " + line); HAHA ALL IT DID WAS COMMENT IT OUT FOR ME
                    // for some reason rewriteCellMagic doesn't work right now
                }
                sourceCode += line;
            }

            //let sourceCode = cell['source'].filter(str => str[0] != '%').join("");
            //console.log(sourceCode);
            notebookCode += sourceCode + '\n';

        }
    }
    //console.log(notebookCode);
    printDefUse(notebookCode);

}

readCode(jsondata);

/** TO DO
 * Find notebooks whose specs are actually included in the specs file ughhh
 * FIND SCOPE OF THE MAX EXTENT OF AN IMPORT STATEMENT -> when the subsequent def is no longer used beyond a certain point
 *
 * Curate notebooks so there's no "no spec for module ____" error
 * GRAPHVIZ to visualize the dataflow graphs (import statements)
 *
 *
 * "import" and "from" types are both import statements, idk why they separate them but ok
 *
 *
 *
 * Can print node types module -> see def_use relevant to those nodes only? = DONE
 * We're interested in the IMPORT -> DEF          def-use pairs     = DONE
 * How to get it to print only the definition line of code???   = DONE
 * Figure out how to slice import statements      = DONE
 * @type {string}
 */



var text = fs.readFileSync(name).toString();

function printDefUse(code){

    let tree = py.parse(code);
    //console.log(py.walk(tree).map(function (node) { return node.type; }));

    let cfg = new py.ControlFlowGraph(tree);
    //console.log(cfg.blocks);
    //console.log("-----");
    //console.log(cfg.blocks[0]);

    const analyzer = new py.DataflowAnalyzer();
    const flows = analyzer.analyze(cfg).dataflows;
    var importScope = {};
    var lineToCode = {};




    for (let flow of flows.items) {
        let fromNode = py.printNode(flow.fromNode).split("\n");
        let toNode = py.printNode(flow.toNode).split("\n");
        lineToCode[flow.fromNode.location.first_line] = fromNode[0];
        lineToCode[flow.fromNode.location.last_line] = fromNode[fromNode.length-1];
        lineToCode[flow.toNode.location.last_line] = toNode[toNode.length-1];
        lineToCode[flow.toNode.location.first_line] = toNode[0];

        if (flow.fromNode.type == "from" || flow.fromNode.type == "import") {
            importScope[flow.fromNode.location.first_line] = -1;
            // add to list of dictionaries
        }
        //g.setEdge(flow.fromNode.location.first_line.toString(), flow.toNode.location.first_line.toString());

    }
    var n = countLines;
    console.log("NOTEBOOK NAME: ", name);
    console.log("NUMBER OF LINES OF CODE ", n);
    // need graph size to be size of lineToCode, not number of edges
    var numgraph = new graphing(n+1);

    for (let flow of flows.items) {
        numgraph.addEdge(flow.fromNode.location.first_line, flow.toNode.location.first_line);

    }
    findImportScope(importScope, lineToCode, numgraph);
    // store the location of "import"  in a graph and do depth-first search = note last location
    // build a graph

}

function findImportScope(importScope, lineToCode, numgraph) {
    //console.log("edges", g.edges());
    //console.log("nodes", g.nodes());
    //console.log("import statements: ", importScope);
    //console.log("line to code: ", lineToCode);
    //console.log(lineToCode[11]);
    //console.log(numgraph.edge.length);
    //console.log(numgraph.edge);

    var imports = Object.keys(importScope);
    var scopes = {};

    for (let lineNum of imports) {
        var result = numgraph.findLongestPathSrc(numgraph.edge.length, parseInt(lineNum))
        scopes[lineNum] = result[0];
        var order = result[1];
        console.log(lineToCode[lineNum]);
        console.log("START: ", lineNum.toString(), " END: ", scopes[lineNum]);

        //console.log(result[1]);
        //generateEdges(order);
        //labelNodeColor(order[0], order, lineToCode);
    }
    //console.log(scopes);


}

// TO DO: Convert "order" into the _ -> _ format, with everything following an import statement being the colour of that import statement
// generate colour map txt file for each notebook
// then go to pycharm and run cindy_graph.py


function generateEdges(order) {
    var i;
    for (i=0; i<order.length-1;i++) {
        var writeData = order[i].toString() + '->' + order[i+1].toString() + '\n'
        fs.appendFileSync('C:/Users/grace/PycharmProjects/model-card-generator/graphviz/' + name + '.txt', writeData);
    }
}

function labelNodeColor(node, order, lineToCode) {
    var nodeCode = lineToCode[node];
    var filePathColor = 'C:/Users/grace/PycharmProjects/model-card-generator/graphviz/' + name + '.color.txt';
    //var filePathColor = './' + name + '.color.txt';
    var filePathCode = 'C:/Users/grace/PycharmProjects/model-card-generator/graphviz/'  + name + '.code.txt';
    //var filePathCode = './' + name + '.code.txt';
    var label = '';
    var color = '';


    if (nodeCode.includes('datasets')) {
        // everything in order should be
        label = "Data Cleaning";
    } else if(nodeCode.includes('.impute')) {
        label = "Data Cleaning";
    } else if(nodeCode.includes('.feature_extraction')) {
        label = "Data Cleaning";
    } else if(nodeCode.includes('.utils')) {
        label = "Data Cleaning";
    } else if(nodeCode.includes('.preprocessing')) {
        label = "Preprocessing";
    }else if(nodeCode.includes('.clustering')) {
        label = "Preprocessing";
    }else if(nodeCode.includes('.feature_selection')) {
        label = "Preprocessing";
    }else if(nodeCode.includes('.pipeline')) {
        label = "Preprocessing";
    }else if(nodeCode.includes('.linear_model')) {
        label = "Model Training";
    }else if(nodeCode.includes('.multi_class')) {
        label = "Model Training";
    }else if(nodeCode.includes('.naive_bayes')) {
        label = "Model Training";
    }else if(nodeCode.includes('.neighbors')) {
        label = "Model Training";
    }else if(nodeCode.includes('.neural_network')) {
        label = "Model Training";
    }else if(nodeCode.includes('.svm')) {
        label = "Model Training";
    }else if(nodeCode.includes('.semi_supervised')) {
        label = "Model Training";
    }else if(nodeCode.includes('.tree')) {
        label = "Model Training";
    }else if(nodeCode.includes('model_selection')) {
        label = "Model Evaluation";
    }else if(nodeCode.includes('metrics')) {
        label = "Model Evaluation";
    }else {
        label = "Null";
    }

    if (label === "Data Cleaning") {
        color = "green";
    }else if (label === "Preprocessing") {
        color = "blue";
    }else if (label === "Model Training") {
        color = "orange";
    }else if (label === "Model Evaluation") {
        color = "red";
    }else {
        color = "white";
    }

    for (let vertex of order) {
        //console.log("vertex", vertex, color);
        var writeData = vertex.toString() + '->' + color + '\n'
        var writeCodeData = vertex.toString() + '->' + lineToCode[vertex] + '\n';
        fs.appendFileSync(filePathColor, writeData);
        fs.appendFileSync(filePathCode, writeCodeData);
    }

}





/**
 Data cleaning
 datasets
 impute
 feature_extraction
 utils

 Preprocessing
 preprocessing
 clustering
 feature_selection
 pipeline

 Model Training
 linear_model
 multi_class
 naive_bayes
 neighbors
 neural_network
 svm
 semi_supervised
 tree

 Model Evaluation
 model_selection
 metrics
 **/






//console.log(text + "\n")
//printDefUse(text)


/**
 var Graph = require("graphlib").Graph;

 // Create a new directed graph
 var g = new Graph();

 // Add node "a" to the graph with no label
 g.setNode("a");

 g.hasNode("a");
 // => true
// g.setEdge("c", "d", { k: 456 });
 // Add node "b" to the graph with a String label
 g.setNode("b", "b's value");

 // Get the label for node b
 g.node("b");
 // => "b's value"

 // Add node "c" to the graph with an Object label
 g.setNode("c", { k: 123 });

 **/