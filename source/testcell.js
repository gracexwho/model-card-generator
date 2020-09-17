"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TestCell = void 0;
var TestCell = /** @class */ (function () {
    function TestCell(text, executionCount, output, persistentId) {
        //if (hasError === void 0) { hasError = false; }
        let currid = genid();
        this.text = text;
        this.executionCount = executionCount;
        //this.hasError = hasError;
        //this.executionEventId = executionEventId || genid();
        this.executionEventId = currid;
        this.persistentId = persistentId || currid;
        this.output = output;

    }
    TestCell.prototype.deepCopy = function () { return this; }; // not used for testing
    return TestCell;
}());

var ID = 0;
function genid() {
    return 'id' + (ID++);
}

function printTestCell(cell) {
    let info = {"persistentId":cell.persistentId, "code":cell.text};
    try {
        info["output"] = output.toJSON();
    } catch {
        info["output"] = "";
    }
    return info
}

exports.TestCell = TestCell;
exports.printTestCell = printTestCell;

//# sourceMappingURL=testcell.js.map