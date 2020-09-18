"use strict";
exports.__esModule = true;

// COMMAND: node main.js ../assets/News_Categorization_MNB.ipynb

var py = require("../lib/python-program-analysis/dist/es5");
var graphing = require("./Graph.js").Graph;
var fs = require('fs');
var path = require('path');
var ic = require("./infocell");
var child = require('child_process');
var dep = require("./cell_deps.js");


var args = process.argv.slice(2);
var filePath = args[0];
var labels = args[1];
var countLines = 0;

var markdown_contents = "";




class ModelCard {
    constructor() {
        this.JSONSchema = {
            modelname:{Model_Name:""},
            authorinfo:{title:"Author Info"},
            dataset: {title: "Dataset", description:"", link:""},
            references: {title:"References", link:[]},
            libraries:{title:"Libraries Used"},
            pre:{title:"Pre", markdown:""},
            other:{title:"Other", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:"", figures:[], description:""},
            datacleaning:{title:"Data Cleaning", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:"", figures:[], description:""},
            preprocessing:{title:"Preprocessing", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:"", figures:[], description:""},
            hyperparameters:{title:"Hyperparameters", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", values:""},
            modeltraining:{title:"Model Training", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:"", figures:[], description:""},
            modelevaluation:{title:"Evaluation", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:"", figures:[], description:""}
        }
        this.outputs = {};
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
var model_card = new ModelCard();


function createCell(text, executionCount, output) {
    return new ic.InfoCell(text, executionCount, output);
}




function readCells(filePath) {
    var contents = fs.readFileSync(path.resolve(__dirname, filePath));
    let jsondata = JSON.parse(contents);
    var notebookCode = "\n";
    var notebookMarkdown = "\n";
    const rewriter = new py.MagicsRewriter();
    var currStage = "pre";
    let id_count = -1;
    let programbuilder = new py.ProgramBuilder();
    model_card.JSONSchema["modelname"]["Model_Name"] = filePath;

    for (let cell of jsondata['cells']) {
        let sourceCode = "";
        if (cell['cell_type'] === 'markdown') {
            model_card.JSONSchema[currStage]["markdown"] += cell['source'];
        } else if (cell['source'][0] != undefined){
            // it's code
            id_count += 1;
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
                if (line[0] === "%") {
                    line = rewriter.rewriteLineMagic(line);
                }
                countLines += 1;
                model_card.JSONSchema[currStage]["lineNumbers"] += countLines;
                sourceCode += line;
            }
            notebookCode += sourceCode + '\n';
            let code_cell = createCell(sourceCode, cell['execution_count'], cell['outputs'][0]);
            //console.log(ic.printInfoCell(code_cell));
            //console.log("OUTPUT: ", cell["outputs"]);
            if (cell["outputs"].length != 0) {
                //console.log("OUTPUT : ", cell["outputs"][0]['output_type']);
                model_card.outputs[code_cell.persistentId] = cell["outputs"][0];
                if (cell["outputs"][0]['output_type'] == 'display_data') {
                    var bitmap = new Buffer.from(cell["outputs"][0]['data']['image/png'], 'base64');
                    fs.writeFileSync(__dirname + "/../example/" + code_cell.persistentId + ".jpg", bitmap);
                    var image = "![Hello World](data:image/png;base64," + cell["outputs"][0]['data']['image/png'];
                    //console.log(model_card.JSONSchema);
                    model_card.JSONSchema[currStage]["figures"] += image;
                }

            }

            programbuilder.add(code_cell)
            model_card.JSONSchema[currStage]["cells"] += JSON.stringify(code_cell);
            //console.log(code_cell);
            //console.log(model_card.JSONSchema[currStage]["cells"]);
            model_card.JSONSchema[currStage]["source"] += sourceCode;
            model_card.JSONSchema[currStage]["cell_ids"] += id_count;
        }
    }
    // id_count = persistentId
    let code = programbuilder.buildTo("id" + id_count.toString()).text;
    //console.log(model_card);
    //console.log("NOTEBOOK CODE : " + notebookCode);
    return [notebookCode, notebookMarkdown, model_card];

}


function generateModelName(notebookMarkdown) {
    var nbname = filePath.replace(/^.*[\\\/]/, '')
    //console.log("------------------MODEL CARD--------------------");
    //console.log("## NOTEBOOK NAME ##")
    //console.log("File Name: ", nbname);

    var matches = notebookMarkdown.match(/\bhttps?:\/\/\S+/gi);
    model_card.JSONSchema["modelname"]['Model_Name'] = nbname;
    model_card.JSONSchema["references"]["link"] = matches;

    console.log(notebookMarkdown);

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

        if (flow.fromNode.type === "from" || flow.fromNode.type === "import") {
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
    //console.log(model_card.JSONSchema["preprocessing"]["imports"]);
    generateLibraryInfo(imports);
}

function generateLibraryInfo(imports) {
    let library_defs = JSON.parse(fs.readFileSync(__dirname + "/../assets/library_defs.json"));
    //console.log("## Libraries Used ##");
    markdown_contents += "## Libraries Used ##" + "\n";
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
            //console.log("### From the library ", lib, " ###");
            //console.log(library_defs[lib]["description"]);
            markdown_contents += "#### From the library " + lib + " ####" + "\n";
            for (let element of libraries[lib]) {
                markdown_contents += element + "    " + imports[element] + "\n" + "\n";
            }
            //libraries[lib].forEach(element => console.log(element, "\t", imports[element]));
            //console.log("--");
        }
    }

}

function printCellsOfStage(stage_name, model_card) {
    for (let cell in model_card.JSONSchema[stage_name]["cells"]) {
        console.log(ic.printInfoCell(cell));
    }
}

function getOutput() {
    // look at "output_type" of json"
    var hello = new InfoCell(text, executionCount, executionEventId);
}

function printModelCard(model_card) {
    console.log(model_card.JSONSchema);
}


function generateMarkdown(model_card) {

    var keys = Object.keys( model_card.JSONSchema );
    for( var i = 0,length = keys.length; i < length; i++ ) {
        if (keys[i] == 'libraries') {
            printLineDefUse(notebookCode, model_card);
        } else {
            var stageKeys = Object.keys(model_card.JSONSchema[keys[i]]);
            for (let stageKey of stageKeys) {
                if (stageKey == 'title') {
                    markdown_contents += "## " + model_card.JSONSchema[keys[i]][stageKey] + " ##" + "\n";
                } else {
                    if (stageKey == 'source') {
                        markdown_contents += "### " + stageKey + " ###" + "\n";
                        markdown_contents += "``` " + "\n" + model_card.JSONSchema[keys[i]][stageKey] + "\n" + " ```" + "\n";

                    } else if (stageKey == "outputs") {
                        markdown_contents += "### " + stageKey + " ###" + "\n";
                        markdown_contents += model_card.JSONSchema[keys[i]][stageKey] + "\n";
                        //var image = document.createElement('img');
                        //image.src = "data:image/png;base64," + base64JsonData;
                    } else if (stageKey == "imports" || stageKey == "markdown") {
                        continue;
                    } else if (stageKey == "figures") {
                        markdown_contents += markdown_contents += "### " + stageKey + " ###" + "\n";
                        for (let image of model_card.JSONSchema[keys[i]][stageKey]) {
                            markdown_contents += image + "\n";
                        }
                    } else {
                        markdown_contents += "### " + stageKey + " ###" + "\n";
                        markdown_contents += model_card.JSONSchema[keys[i]][stageKey] + "\n";
                    }
                }
            }
        }


    }
    fs.writeFile('ModelCard2.md', markdown_contents, (err) => {
        if (err) throw err;
        console.log('Model card saved');
    });

}



function main() {

    //analyze_notebooks();      //Cindy's code
    //convertColorToLabel(filePath);

    var res = readCells(filePath);
    var notebookCode = res[0];
    var notebookMarkdown = res[1];
    generateModelName(notebookMarkdown);
    generateMarkdown(model_card);


    //printModelCard(model_card);
    //Stage("datacleaning", model_card);

    //printCellsOfStage("preprocessing", model_card);
    //printCellsOfStage("modeltraining", model_card);
    //printCellsOfStage("modelevaluation", model_card);



    //graphvisual();
    //convert_nb();
    //let { stdout } = await sh('node analyze_notebooks ' + '../assets/');
    //console.log("STDOUT:", stdout);
    //let stdout = await sh('node convert_nb.js ' + args[0] + ' ' + args[0].split('.ipynb')[0] + '_analysis.txt');
    //console.log("STDOUT", stdout);
}


main();









