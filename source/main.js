"use strict";
exports.__esModule = true;

/**
 * @TODO
 * Merge Code into Sixian's Repo
 * ### LInk to relevant sklearn documentation/pull hyperparameter descriptions from lale
 * **/

// COMMAND: node main.js ../assets/News_Categorization_MNB.ipynb

//var py = require("../lib/python-program-analysis/dist/es5");
var graphing = require("./Graph.js").Graph;
var py = require("modified-python-program-analysis/dist/es5");
var fs = require('fs');
var path = require('path');
var ic = require("./infocell");
var dep = require("./cell_deps.js");


var args = process.argv.slice(2);
var filePath = args[0];
var countLines = 0;


class ModelCard {
    constructor() {
        this.JSONSchema = {
            modelname:{title:"", Filename:"", cell_ids:[]},
            author:{title:"Author"},
            datasets: {title: "Datasets", description:"", links:"", cell_ids:[]},
            references: {title:"References", source:"", links:[], cell_ids:[]},
            libraries:{title:"Libraries Used", lib:{}, info:{}, cell_ids:[]},
            hyperparameters:{title:"Hyperparameters", cell_ids:[], lineNumbers:[], source:"", values:[], description:""},
            misc:{title:"Miscellaneous", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:[], figures:[], description:"", outputs:[]},
            plotting:{title:"Plotting", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:[], figures:[], description:"", outputs:[]},
            datacleaning:{title:"Data Cleaning", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:[], figures:[], description:"", outputs:[]},
            preprocessing:{title:"Preprocessing", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:[], figures:[], description:"", outputs:[]},
            modeltraining:{title:"Model Training", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:[], figures:[], description:"", outputs:[]},
            modelevaluation:{title:"Evaluation", cell_ids:[], cells:[], lineNumbers:[], source:"", markdown:"", imports:[], functions:[], figures:[], description:"", outputs:[]}
        }
        this.line_to_cell = {};
        this.markdown = "";
        this.intended_use = "";
        this.ethical_considerations = "";
        this.developer_comments = "";
        this.hyperparamschemas = {};
    }

    getStageLineNumbers(stage_name) {
        return this.JSONSchema[stage_name]["lineNumbers"];
    }
    getPLineNumbers() {
        return this.JSONSchema["plotting"]["lineNumbers"];
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

function convertColorToLabel(filePath) {
    // data collection -> red
    // data cleaning -> yellow
    // data labeling -> green
    // feature engineering -> lightblue
    // training -> purple
    // evaluation -> orange
    // model deployment -> pink

    var color_map = dep.printLabels(filePath);

    //var colourFile = fs.readFileSync(path.resolve(__dirname, filePath.split(".ipynb")[0] + "_deps_and_labels_new.txt"), "utf8");
    var mapObj = {red:"Data collection",yellow:"Data cleaning",
        green:"Data labelling", "lightblue":"Plotting", "blue":"Feature Engineering",
        purple:"Training", orange:"Evaluation", pink:"Model deployment"};

    var re = new RegExp(Object.keys(mapObj).join("|"),"gi");
    color_map = color_map.replace(re, function(matched){
        return mapObj[matched];
    });

    fs.writeFile((__dirname + "/../assets/" + filePath.split(".ipynb")[0] + '_labels.txt'), color_map,
        function (err) {
            if (err) throw err;
            //console.log('Labels file saved!');
        });

    color_map = color_map.split("\n");
    var new_color_map = {};

    for (let element of color_map) {
        element = element.split("->");
        new_color_map[element[0]] = element[1];
    }


    const testFolder = '/../lib/lale/sklearn/';
    var schemas = {};
    var filenames = fs.readdirSync(__dirname + testFolder);
    filenames.forEach(file => {
        var newname = file.replace("_", "");
        newname = newname.replace(".py", "");
        schemas[newname] = file;
    });
    model_card.hyperparamschemas = schemas;
    //console.log(schemas);

    return new_color_map;
}


function readCells(filePath, new_color_map) {
    // ## Section  in Markdown
    var content = fs.readFileSync(path.resolve(__dirname, filePath));
    let jsondata = JSON.parse(content);
    var notebookCode = "\n";
    var notebookMarkdown = "";
    const rewriter = new py.MagicsRewriter();
    var currStage = "misc";
    let id_count = 0;
    let flag = true;
    let programbuilder = new py.ProgramBuilder();
    model_card.JSONSchema["modelname"]["Filename"] = filePath.split("/").slice(-1).toString();
    console.log();
    fs.mkdirSync("../example/" + model_card.JSONSchema["modelname"]["Filename"], { recursive: true })

    for (let cell of jsondata['cells']) {
        let sourceCode = "";
        if (cell['cell_type'] === 'markdown') {
            model_card.JSONSchema[currStage]["markdown"] += "\n" + cell['source'];
            for (let mdline of cell['source']) {
                var matches = mdline.match(/\bhttps?:\/\/[\S][^)]+/gi);
                if (matches !== null) {
                    model_card.JSONSchema["references"]["cell_ids"].push(id_count);
                    model_card.JSONSchema["references"]["links"] = model_card.JSONSchema["references"]["links"].concat(matches);
                }
            }
            if (id_count == 0 && flag) {
                flag = false;
                model_card.JSONSchema["modelname"]["title"] = cell['source'][0];
                model_card.JSONSchema["modelname"]["cell_ids"] = id_count;
            }
            id_count += 1;
            notebookMarkdown += cell["source"];

        } else if (cell['source'][0] != undefined){
            id_count += 1;
            var key = cell['execution_count'].toString();
            if (key in new_color_map) {
                var stage = new_color_map[key];
                if (stage == "Data collection" || stage == "Data cleaning" || stage == "Data labelling") {
                    currStage = "datacleaning";
                } else if (stage == "Feature Engineering") {
                    currStage = "preprocessing";
                } else if (stage == "Training") {
                    currStage = "modeltraining";
                } else if (stage == "Evaluation") {
                    currStage = "modelevaluation";
                } else if (stage == "Plotting") {
                    currStage = "plotting";
                }
            }

            for (let line of cell['source']) {
                if (line[0] === "%") {
                    line = rewriter.rewriteLineMagic(line);

                }
                countLines += 1;
                model_card.JSONSchema[currStage]["lineNumbers"].push(countLines);
                model_card.line_to_cell[countLines] = id_count;
                sourceCode += line;
            }
            notebookCode += sourceCode + '\n';
            let code_cell = createCell(sourceCode, cell['execution_count'], cell['outputs'][0]);
            //console.log(ic.printInfoCell(code_cell));
            //console.log("OUTPUT: ", cell["outputs"]);
            if (cell["outputs"].length != 0) {
                for (let output in cell["outputs"]) {
                    //model_card.outputs[code_cell.persistentId] += output;
                    if (cell["outputs"][output]['output_type'] == 'display_data') {
                        var bitmap = new Buffer.from(cell["outputs"][output]['data']['image/png'], 'base64');
                        fs.writeFileSync(__dirname + "/../example/" + model_card.JSONSchema["modelname"]["Filename"] + "/" + code_cell.persistentId + ".jpg", bitmap);
                        var image = "![Hello World](data:image/png;base64," + cell["outputs"][output]['data']['image/png'];
                        model_card.JSONSchema[currStage]["figures"].push(code_cell.persistentId + ".jpg");
                    } else if (cell["outputs"][output]['output_type'] == 'stream') {
                        var info = cell["outputs"][output]["text"];
                        model_card.JSONSchema[currStage]["outputs"].push(info);
                    }
                }
            }

            programbuilder.add(code_cell)
            model_card.JSONSchema[currStage]["cells"] += code_cell;
            //console.log(code_cell);
            //console.log(model_card.JSONSchema[currStage]["cells"]);
            model_card.JSONSchema[currStage]["source"] += sourceCode;
            model_card.JSONSchema[currStage]["cell_ids"].push(id_count);
        }
    }
    // id_count = persistentId
    //let code = programbuilder.buildTo("id" + id_count.toString()).text;
    model_card.markdown += notebookMarkdown;
    printLineDefUse(notebookCode, model_card);
    return [notebookCode, notebookMarkdown, model_card];
}


function printLineDefUse(code, model_card){
    let tree = py.parse(code);
    let cfg = new py.ControlFlowGraph(tree);
    const analyzer = new py.DataflowAnalyzer();
    const flows = analyzer.analyze(cfg).dataflows;

    var importScope = {};
    var lineToCode = {};
    var pLines = model_card.getPLineNumbers();
    var dcLines = model_card.getDCLineNumbers();
    var ppLines = model_card.getPPLineNumbers();
    var mtLines = model_card.getMTLineNumbers();
    var meLines = model_card.getMELineNumbers();


    for (let flow of flows.items) {
        let fromNode = py.printNode(flow.fromNode).split("\n");
        let toNode = py.printNode(flow.toNode).split("\n");

        lineToCode[flow.fromNode.location.first_line] = fromNode[0];
        lineToCode[flow.fromNode.location.last_line] = fromNode[fromNode.length-1];
        lineToCode[flow.toNode.location.last_line] = toNode[toNode.length-1];
        lineToCode[flow.toNode.location.first_line] = toNode[0];

        if (flow.fromNode.type === "from" || flow.fromNode.type === "import") {
            if (fromNode[0].includes("sklearn.datasets")) {
                model_card.JSONSchema["datasets"]["source"] += fromNode[0];
                model_card.JSONSchema["datasets"]["cell_ids"].push(model_card.line_to_cell[flow.fromNode.location.first_line]);
            }

            //Check Hyperparameters
            var input = fromNode[0].toLowerCase();
            var match = "";
            var hyperparam_descriptions = {};

            Object.keys(model_card.hyperparamschemas).forEach(function(key) {
                if (input.includes(key)) {
                    var hcontents = fs.readFileSync(__dirname + "/../lib/lale/sklearn/" + model_card.hyperparamschemas[key], "utf8");
                    var hflag = false;
                    var pflag = false;
                    var hyperflag = false
                    var hyperparams = "";
                    var hproperties = "";
                    var openbrackets = 0;

                    for (let hline of hcontents.split("\n")) {
                        if (hline.includes("_hyperparams_schema =")) {
                            hyperflag = true;
                        }
                        if (hyperflag) {
                            if (hline.includes("'properties':")) {
                                //console.log(hline);
                                pflag = true;
                            }
                            openbrackets += (hline.match(/{/g)||[]).length
                            openbrackets -= (hline.match(/}/g)||[]).length
                            //console.log(line);
                            //console.log(brackets);
                            //console.log();
                        }

                        if (hline.includes("relevantToOptimizer")) {
                            hflag = true;
                        }
                        if (hflag) {
                            hyperparams += hline;
                        }
                        if (hline.includes("],")) {
                            hflag = false;
                        }
                        if (pflag == true && hyperflag == true) {
                            hproperties = hproperties + hline + "\n";
                        }
                        if (hyperflag && openbrackets == 0) {
                            break;
                        }
                    }
                    //console.log(hproperties);

                    hyperparams = hyperparams.substr(hyperparams.indexOf('[')+1);
                    hyperparams = hyperparams.split("]")[0];
                    hyperparams = hyperparams.split(",");
                    var parameters = [];
                    for (let s of hyperparams) {
                        s = s.replace(/['"]+/g, "");
                        s = s.trim();
                        if (s) {
                            parameters.push(s);
                        }
                    }

                    pflag = false;
                    openbrackets=0;
                    var desc = "";
                    var substring = null;
                    var param = "";

                    function containsAny(str, substrings) {
                        for (var i = 0; i != substrings.length; i++) {
                            var substring = "'" + substrings[i] + "'";
                            if (str.indexOf(substring) != - 1) {
                                return substring;
                            }
                        }
                        return null;
                    }


                    for (let line of hproperties.split("\n")) {
                        if (!pflag) {
                            substring = containsAny(line, parameters);
                            if (substring != null) {
                                //parameters.some(function(v) {hyperparam_descriptions["'" + v + "'"] = ""; return line.indexOf("'" + v + "'") >= 0; })
                                pflag = true;
                                param = substring;
                            }
                        }
                        if (pflag) {
                            if (line.includes("{")) {
                                openbrackets += 1;
                            }
                            if (line.includes("}")) {
                                openbrackets -=1;
                            }
                            desc = desc + line + "\n";

                            if (openbrackets <= 0) {
                                pflag = false;
                                //console.log(desc);
                                //console.log("#####")
                                hyperparam_descriptions[input] += desc;
                                //console.log(desc);
                                desc = "";
                            }

                        }
                    }

                    //console.log(hyperparam_descriptions);


                    model_card.JSONSchema["hyperparameters"]["values"].concat(parameters);
                    model_card.JSONSchema["hyperparameters"]["lineNumbers"].push(flow.fromNode.location.first_line);
                    model_card.JSONSchema["hyperparameters"]["cell_ids"].push(model_card.line_to_cell[flow.fromNode.location.first_line]);
                    model_card.JSONSchema["hyperparameters"]["source"] += fromNode[0];
                    model_card.JSONSchema["hyperparameters"]["description"] = hyperparam_descriptions;
                }
            });

            importScope[flow.fromNode.location.first_line] = -1;
            model_card.JSONSchema["libraries"]["cell_ids"].push(model_card.line_to_cell[flow.fromNode.location.first_line]);

        } else if (flow.fromNode.type === "def") {
            if (flow.fromNode.location.first_line in pLines) {
                model_card.JSONSchema["plotting"]["functions"].push(py.printNode(flow.fromNode));
            } else if (flow.fromNode.location.first_line in dcLines) {
                model_card.JSONSchema["datacleaning"]["functions"].push(py.printNode(flow.fromNode));
            } else if (flow.fromNode.location.first_line in ppLines) {
                model_card.JSONSchema["preprocessing"]["functions"].push(py.printNode(flow.fromNode));
            } else if (flow.fromNode.location.first_line in mtLines) {
                model_card.JSONSchema["modeltraining"]["functions"].push(py.printNode(flow.fromNode));
            } else if (flow.fromNode.location.first_line in meLines) {
                model_card.JSONSchema["modelevaluation"]["functions"].push(py.printNode(flow.fromNode));s
            }
        }

    }
    var n = countLines;
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
    model_card.JSONSchema["libraries"]["lib"] = libraries;
    model_card.JSONSchema["libraries"]["info"] = library_defs;
    //console.log(library_defs);
    /**
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
     **/

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
    console.log(JSON.stringify(model_card.JSONSchema));
}


function generateMarkdown(model_card, notebookname="") {
    var markdown_contents = "";
    var keys = Object.keys( model_card.JSONSchema );

    for( var i = 0,length = keys.length; i < length; i++ ) {
        var stageKeys = Object.keys(model_card.JSONSchema[keys[i]]);
        for (let stageKey of stageKeys) {
            if (stageKey == 'title') {
                markdown_contents += "## " + model_card.JSONSchema[keys[i]][stageKey] + " ##" + "\n";
            } else {
                if (stageKey == 'source') {
                    markdown_contents += "### " + stageKey + " ###" + "\n";
                    markdown_contents += "``` " + "\n" + model_card.JSONSchema[keys[i]][stageKey] + "\n" + " ```" + "\n";

                }else if (stageKey == "description" && keys[i] == "hyperparameters") {
                    markdown_contents += "### " + stageKey + " ###" + "\n";
                    markdown_contents += JSON.stringify(model_card.JSONSchema[keys[i]][stageKey]) + "\n";

                }else if (stageKey == "outputs") {
                    markdown_contents += "### " + stageKey + " ###" + "\n";
                    markdown_contents += model_card.JSONSchema[keys[i]][stageKey] + "\n";
                    //var image = document.createElement('img');
                    //image.src = "data:image/png;base64," + base64JsonData;

                } else if (stageKey == "imports" || stageKey == "markdown") {
                    continue;
                } else if (stageKey == "figures") {
                    markdown_contents += "### " + stageKey + " ###" + "\n";
                    for (let image of model_card.JSONSchema[keys[i]][stageKey]) {
                        //![id5](./image/id5.jpg)
                        markdown_contents += "![" + image + "](" + "../example/" +
                            model_card.JSONSchema["modelname"]["Filename"] + "/" + image + ")" + "\n";
                    }
                } else if (keys[i] == "references" && stageKey == "links") {
                    for (let link of model_card.JSONSchema['references']['links']) {
                        markdown_contents += link + "\n";
                    }
                }else {
                    markdown_contents += "### " + stageKey + " ###" + "\n";
                    markdown_contents += JSON.stringify(model_card.JSONSchema[keys[i]][stageKey]) + "\n";
                }
            }
        }


    }
    //console.log(markdown_contents);
    fs.writeFile('../assets/model_cards/' + 'ModelCard' + notebookname + '.md', markdown_contents, (err) => {
        if (err) throw err;
        //console.log('Model card saved!');
        //console.log(model_card);
    });

}

function getExt(filename){
    return filename.substring(filename.lastIndexOf('.')+1, filename.length);
}


function bulk_run() {
    var fpath = "../assets/"
    fs.readdirSync(fpath).forEach(file => {
        var filePath = "";
        if (getExt(file) === "ipynb"){
            console.log('Currently processing:');
            console.log(fpath + file + '\n');
            filePath = fpath + file;

            var new_color = convertColorToLabel(filePath);
            var res = readCells(filePath, new_color);
            var notebookCode = res[0];
            var notebookMarkdown = res[1];
            var MC = res[2];

            generateMarkdown(MC, "_" + file.split(".")[0]);

        }

    });
}

function main() {
    bulk_run()

    /**
     *
     var new_color = convertColorToLabel(filePath);
     var res = readCells(filePath, new_color);
     var notebookCode = res[0];
     var notebookMarkdown = res[1];
     var MC = res[2];

     generateMarkdown(MC, notebookCode);
     */

}


main();









