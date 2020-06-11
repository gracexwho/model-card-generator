import "mocha"; // need to include this for all test files
import * as assert from "assert";
import {SmokeTest} from "../dist";

// specify to get code from dist, or else it won't run for your tests

describe("index",()=>{
    it("should say 'hello world'",()=> {
        SmokeTest.HelloWorld();
        assert.ok(true);
    })
})