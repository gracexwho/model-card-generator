// TO DO
// Parse model name
// *** get import + library descriptions
// Use Program-Builder to get Cell Data


"use strict";
exports.__esModule = true;
var py = require("../../python-program-analysis/dist/es5");
var dagre = require("dagre");
var Graph = require("graphlib").Graph;
var graphing = require("./Graph.js").Graph;
var dagreD3 = require("dagre-d3");
var d3 = require("d3");
var fs = require('fs');



var args = process.argv.slice(2);
var filename = args[0];
var countLines = 0;



function readCode(filename) {
    var contents = fs.readFileSync(filename.toString());
    let jsondata = JSON.parse(contents);
    var notebookCode = "\n";
    const rewriter = new py.MagicsRewriter();

    for (let cell of jsondata['cells']) {
        if (cell['cell_type'] == 'code') {
            var sourceCode = "";
            for (let line of cell['source']) {
                countLines += 1;
                if (line[0] == '%') {
                    line = rewriter.rewriteLineMagic(line);
                }
                sourceCode += line;
            }
            //let sourceCode = cell['source'].filter(str => str[0] != '%').join("");
            notebookCode += sourceCode + '\n';
        }
    }
    return notebookCode;

}

function generateModelName(filename) {
    // either filename or first line of markdown cell
    // parse from end to previous /
    
    console.log("------------------MODEL CARD--------------------");
    console.log("## NOTEBOOK NAME ##")
    console.log("File Name: ", filename);

}


function printLineDefUse(code){
    let tree = py.parse(code);
    let cfg = new py.ControlFlowGraph(tree);

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
        }
        //g.setEdge(flow.fromNode.location.first_line.toString(), flow.toNode.location.first_line.toString());

    }
    var n = countLines;
    console.log();
    console.log("## NUMBER OF LINES OF CODE ##");
    console.log(n);


    // need graph size to be size of lineToCode, not number of edges
    var numgraph = new graphing(n+1);

    for (let flow of flows.items) {
        numgraph.addEdge(flow.fromNode.location.first_line, flow.toNode.location.first_line);
    }
    findImportScope(importScope, lineToCode, numgraph);

}

function findImportScope(importScope, lineToCode, numgraph) {
    var importCode = Object.keys(importScope);
    var scopes = {};
    var imports = {};

    for (let lineNum of importCode) {
        var result = numgraph.findLongestPathSrc(numgraph.edge.length, parseInt(lineNum))
        scopes[lineNum] = result[1];
        var order = result[1];
        //console.log(lineToCode[lineNum]);
        //console.log("START: ", lineNum.toString(), " END: ", scopes[lineNum]);
        imports[lineToCode[lineNum]] = "START:" + lineNum.toString() + "\t" + " END:" + scopes[lineNum];
    }
    console.log();
    generateLibraryInfo(imports);
}

function generateLibraryInfo(imports) {
    let library_defs = JSON.parse(fs.readFileSync("source/library_defs.json"));
    console.log("## Libraries Used ##");
    var libraries = {"pandas":[], "numpy":[], "matplotlib":[], "sklearn":[], "tensorflow":[], "pytorch":[], "OTHER":[]};

    for (let im of Object.keys(imports)) {
        if (im.includes("pandas")){
            libraries["pandas"].push(im);
        } else if (im.includes("numpy")) {
            libraries["numpy"].push(im);
        } else if(im.includes("matplotlib")) {
            libraries["matplotlib"].push(im);
        } else if(im.includes("sklearn")) {
            libraries["sklearn"].push(im);
        } else if (im.includes("tensorflow")) {
            libraries["tensorflow"].push(im);
        } else if (im.includes("pytorch")) {
            libraries["pytorch"].push(im);
        } else {
            libraries["OTHER"].push(im);
        }
    }

    for (let lib of Object.keys(libraries)) {
        if (libraries[lib].length > 0) {
            console.log("### From the library ", lib, " ###");
            console.log(library_defs[lib]["description"]);
            libraries[lib].forEach(element => console.log(element, "\t", imports[element]));
            console.log("--");
        }
    }

}



let notebookCode = readCode(filename);
generateModelName(filename);
printLineDefUse(notebookCode);






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





