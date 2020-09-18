var fs = require('fs');

var args = process.argv.slice(2);
var schemaname = args[0];
var filename = args[1]

var schema = JSON.parse(fs.readFileSync(schemaname.toString()));
var file = fs.readFileSync(filename.toString());

console.log(schema);
console.log("Hyperparameters: ", Object.keys(schema['allOf'][0]['properties']));
