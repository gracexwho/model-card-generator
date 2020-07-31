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
var tc = require("./testcell");


var args = process.argv.slice(2);
var filename = args[0];
var countLines = 0;

class ModelCard {
    constructor() {
        this.JSONSchema = {
            modelname:{title:""},
            authorinfo:{title:"Author Info"},
            dataset: {title: "Dataset", description:"", link:""},
            references: {title:"References"},
            other:{title:"Other", cells:[], lineNumbers:[], source:"", imports:[], functions:"", figures:"", description:""},
            datacleaning:{title:"Data Cleaning", cells:[], lineNumbers:[], source:"", imports:[], functions:"", figures:"", description:""},
            preprocessing:{title:"Preprocessing", cells:[], lineNumbers:[], source:"", imports:[], functions:"", figures:"", description:""},
            hyperparameters:{title:"Hyperparameters", cells:[], lineNumbers:[], source:"", values:""},
            modeltraining:{title:"Model Training", cells:[], lineNumbers:[], source:"", imports:[], functions:"", figures:"", description:""},
            modelevaluation:{title:"Evaluation", cells:[], lineNumbers:[], source:"", imports:[], functions:"", figures:"", description:""}};
    }

    getStageLineNumbers(stage_name) {
        return this.JSONSchema[stage_name]["lineNumbers"];
    }

    getDCLineNumbers() {
        return this.JSONSchema["datacleaning"]["lineNumbers"];
    }
    getPPLineNumbers() {
        return this.JSONSchema["preprocessing"]["lineNumbers"];
    }
    getMTLineNumbers() {
        return this.JSONSchema["modeltraining"]["lineNumbers"];
    }
    getMELineNumbers() {
        return this.JSONSchema["modelevaluation"]["lineNumbers"];
    }
}



function readCells(filename) {
    var contents = fs.readFileSync(filename.toString());
    let jsondata = JSON.parse(contents);

    let id = 1;

    for (let cell of jsondata['cells']) {

    }

}


function readCode(filename) {
    var contents = fs.readFileSync(filename.toString());
    let jsondata = JSON.parse(contents);
    var notebookCode = "\n";
    var notebookMarkdown = "\n";
    const rewriter = new py.MagicsRewriter();
    var currStage = "other";
    var model_card = new ModelCard();

    // Added Manual labels # Data Cleaning, # Preprocessing, # Model Training, # Model Evaluation

    for (let cell of jsondata['cells']) {
        if (cell['cell_type'] === "code") {
            var sourceCode = "";
            if (cell['source'][0].includes("Data Cleaning")) {
                currStage = "datacleaning";
            } else if (cell['source'][0].includes("Preprocessing")) {
                currStage = "preprocessing";

            }else if (cell['source'][0].includes("Model Training")) {
                currStage = "modeltraining";

            }else if (cell['source'][0].includes("Model Evaluation")) {
                currStage = "modelevaluation";
            }
            for (let line of cell['source']) {
                //console.log("LINE: ", line, " COUNTLINES: ", countLines);
                countLines += 1;
                model_card.JSONSchema[currStage]["lineNumbers"].push(countLines);
                //console.log(model_card.JSONSchema[currStage]["lineNumbers"]);
                if (line[0] === "%") {
                    line = rewriter.rewriteLineMagic(line);
                }
                sourceCode += line;
            }
            //let sourceCode = cell['source'].filter(str => str[0] != '%').join("");

            model_card.JSONSchema[currStage]["cells"].push(cell["execution_count"]);
            model_card.JSONSchema[currStage]["source"] += sourceCode;
            notebookCode += sourceCode + '\n';

        } else if (cell['cell_type'] === "markdown"){
            for (let line of cell['source']) {
                notebookMarkdown += line;
            }
            notebookMarkdown += '\n';
        }
    }
    return [notebookCode, notebookMarkdown, model_card];

}

function generateModelName(notebookMarkdown) {
    // either filename or first line of markdown cell
    // parse from end to previous /


    //console.log(notebookMarkdown);
    console.log("------------------MODEL CARD--------------------");
    console.log("## NOTEBOOK NAME ##")
    console.log("File Name: ", filename);

}

function printModelCard(model_card) {
    console.log(model_card.JSONSchema);
}

function printLineDefUse(code, model_card){
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
    findImportScope(importScope, lineToCode, numgraph, model_card);

}

function findImportScope(importScope, lineToCode, numgraph, model_card) {
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
        if (model_card.getDCLineNumbers().includes(parseInt(lineNum))) {
            model_card.JSONSchema["datacleaning"]["imports"].push(lineToCode[lineNum]);
        } else if (model_card.getPPLineNumbers().includes(parseInt(lineNum))) {
            model_card.JSONSchema["preprocessing"]["imports"].push(lineToCode[lineNum]);
        }else if (model_card.getMTLineNumbers().includes(parseInt(lineNum))) {
            model_card.JSONSchema["modeltraining"]["imports"].push(lineToCode[lineNum]);
        }else if (model_card.getMELineNumbers().includes(parseInt(lineNum))) {
            model_card.JSONSchema["modelevaluation"]["imports"].push(lineToCode[lineNum]);
        }

    }
    console.log();
    generateLibraryInfo(imports);
}

function generateLibraryInfo(imports) {
    let library_defs = JSON.parse(fs.readFileSync("library_defs.json"));
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


function getOutput() {
    // look at "output_type" of json"
    const TestCell = require("../../python-program-analysis/dist/es5/testcell");
    var hello = new TestCell(text, executionCount, executionEventId);
}


let res = readCode(filename);
let notebookCode = res[0];
let notebookMarkdown = res[1];
let model_card = res[2];
generateModelName(notebookMarkdown);
printLineDefUse(notebookCode, model_card);
printModelCard(model_card);





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





