(function (factory) {
    if (typeof module === "object" && typeof module.exports === "object") {
        var v = factory(require, exports);
        if (v !== undefined) module.exports = v;
    }
    else if (typeof define === "function" && define.amd) {
        define(["require", "exports", "mocha", "assert", "../dist"], factory);
    }
})(function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    require("mocha"); // need to include this for all test files
    var assert = require("assert");
    var dist_1 = require("../dist");
    // specify to get code from dist, or else it won't run for your tests
    // hello
    describe("index", function () {
        it("should say 'hello world'", function () {
            dist_1.SmokeTest.HelloWorld();
            assert.ok(true);
        });
    });
});
//# sourceMappingURL=index.js.map