"use strict";
exports.__esModule = true;

// COMMAND: node main.js ../assets/News_Categorization_MNB.ipynb

var py = require("../lib/python-program-analysis/dist/es5");
var fs = require('fs');
var path = require('path');
var child = require('child_process');
var dep = require("./cell_deps.js");


var args = process.argv.slice(2);
var directory = args[0];
var labels = args[1];
var countLines = 0;


function analyze_notebooks() {
    function sh(cmd) {
        return new Promise(function (resolve, reject) {
            child.exec(cmd, (err, stdout, stderr) => {
                if (err) {
                    reject(err);
                } else {
                    resolve({ stdout, stderr });
                }
            });
        });
    }

    sh('node analyze_notebooks ' + directory).then(
        function(result) {
            console.log(result['stdout']); // "initResolve"
            graphvisual();
            return "normalReturn";
        }
    )
        .catch(function(result) {
            console.log(result);
            return;
        })

}

function graphvisual() {
    let {PythonShell} = require('python-shell');

    fs.readdirSync(directory).forEach(file => {

        if (getExt(file) === "ipynb") {
            console.log('Currently processing:');
            console.log(directory + file + '\n');

            let options = {
                mode: 'text',
                pythonPath: 'C:\\Program Files\\Python38\\python',
                pythonOptions: ['-u'], // get print results in real-time
                scriptPath: './',
                args: [directory + file.split(".ipynb")[0] + "_deps_and_labels_new.txt"]
            };

            console.log(options["args"]);
            PythonShell.run('graph_visual-4.py', options,
                function (err, results) {
                    if (err) throw err;
                    console.log(results);
                    convert_nb();
                });

        }
    });

}

function convert_nb() {
    function sh(cmd) {
        return new Promise(function (resolve, reject) {
            child.exec(cmd, (err, stdout, stderr) => {
                if (err) {
                    reject(err);
                } else {
                    resolve({ stdout, stderr });
                }
            });
        });
    }

    fs.readdirSync(directory).forEach(file => {
        var wholefile = directory + file;

        if ((getExt(file) === "ipynb") && (file.indexOf('_clean') == -1)) {
            console.log('Currently processing:');
            console.log(wholefile + '\n');

            sh('node convert_nb.js ' + wholefile + ' ' + wholefile.split('.ipynb')[0] + '_analysis.txt').then(
                function (result) {
                    console.log(result['stdout']); // "initResolve"
                    return "normalReturn";
                }
            )
                .catch(function (result) {
                    console.log(result);
                    return;
                })
        }
    });
}

function getExt(filename){
    return filename.substring(filename.lastIndexOf('.')+1, filename.length);
}

function convertColorToLabel(directory) {
    // data collection -> red
    // data cleaning -> yellow
    // data labeling -> green
    // feature engineering -> blue
    // training -> purple
    // evaluation -> orange
    // model deployment -> pink

    // ADD A WRITEFILESYNC TO CELL_DEPS WHERE YOU RECORD THIS THING

    //fs.writeFileSync(__dirname + "/../assets/" + new_name + '_colormap.txt', res_color_map, function (err) {
    //   if (err) throw err;
    //});

    dep.calculateCells(filePath, printMode);

    var colourFile = fs.readFileSync(path.resolve(__dirname, filePath.split(".ipynb")[0] + "_deps_and_labels_new.txt"), "utf8");
    var mapObj = {red:"Data collection",yellow:"Data cleaning",
        green:"Data labelling", blue:"Feature engineering",
        Purple:"Training", Orange:"Evaluation", pink:"Model deployment"};

    var re = new RegExp(Object.keys(mapObj).join("|"),"gi");
    colourFile = colourFile.replace(re, function(matched){
        return mapObj[matched];
    });


    fs.writeFile((__dirname + "/../assets/" + filePath.split(".ipynb")[0] + '_labels.txt'), colourFile,
        function (err) {
            if (err) throw err;
            console.log('Labels file saved!');
        });
}


function main() {
    analyze_notebooks();      //Cindy's code
//convertColorToLabel(directory);
}

main();