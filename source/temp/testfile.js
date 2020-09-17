var py = require("../../../python-program-analysis/dist/es5");
var tc = require("../testcell");
var fs = require("fs");

let filename = "News_Categorization_MNB.ipynb";

function createCell(text, executionCount, output) {
    return new tc.TestCell(text, executionCount, output || "");
}

function readCells(filename) {
    let programbuilder = new py.ProgramBuilder();
    var contents = fs.readFileSync(filename.toString());
    let jsondata = JSON.parse(contents);
    var notebookCode = "\n";
    var notebookMarkdown = "\n";
    const rewriter = new py.MagicsRewriter();
    var currStage = "other";
    let id_count = 1;
    // Added Manual labels # Data Cleaning, # Preprocessing, # Model Training, # Model Evaluation
    for (let cell of jsondata['cells']) {
        let sourceCode = "";

        if (cell['cell_type'] == 'markdown') {
            // no output
            //programbuilder.add(createCell(cell['source'], 0, cell['output']));
        } else {
            for (let line of cell['source']) {
                //console.log("LINE: ", line, " COUNTLINES: ", countLines);
                if (line[0] === "%") {
                    line = rewriter.rewriteLineMagic(line);
                }
                sourceCode += line;
            }
            let code_cell = createCell(sourceCode, cell['execution_count'], cell['output']);
            programbuilder.add(code_cell);
            console.log(tc.printTestCell(code_cell));
        }
    }
    //let code = programbuilder.buildTo("id8").text;
    //console.log(code);
}


readCells(filename);