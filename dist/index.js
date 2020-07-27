(function (factory) {
    if (typeof module === "object" && typeof module.exports === "object") {
        var v = factory(require, exports);
        if (v !== undefined) module.exports = v;
    }
    else if (typeof define === "function" && define.amd) {
        define(["require", "exports"], factory);
    }
})(function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.SmokeTest = void 0;
    var SmokeTest;
    (function (SmokeTest) {
        function HelloWorld() {
            console.log("Hello World");
        }
        SmokeTest.HelloWorld = HelloWorld;
    })(SmokeTest = exports.SmokeTest || (exports.SmokeTest = {}));
    var Logger = /** @class */ (function () {
        function Logger() {
        }
        Logger.prototype.log = function (message) {
            console.log(message);
        };
        return Logger;
    }());
    var logger = new Logger();
    logger.log("hello world");
});
//# sourceMappingURL=index.js.map